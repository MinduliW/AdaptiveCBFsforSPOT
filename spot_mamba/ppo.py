"""
CleanRL-style PPO loop with the pure-PyTorch Mamba2 actor-critic, ported/adapted
from the reference mamba2_cleanrl/ppo.py.

The loop OWNS the recurrent state: it steps the agent token-by-token through the
VecEnv, zeroes each env's (conv,ssm) state on `done`, and marks the next step's
episode_start. The update re-runs the full-sequence parallel scan (with burn-in)
so rollout and update see the identical recurrence (verified in mamba_torch).
Works on CPU (pure-torch Mamba) or CUDA.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .agent import Mamba2ActorCritic
from .buffer import RolloutBuffer


def _to_t(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class MambaPPO:
    def __init__(self, env, config, device=None):
        self.env = env
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.N = env.num_envs
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.act_low = _to_t(env.action_space.low, self.device)
        self.act_high = _to_t(env.action_space.high, self.device)
        self.agent = Mamba2ActorCritic(
            obs_dim, act_dim, d_model=config.d_model, layers=config.layers,
            nodes=config.nodes, log_std_init=config.log_std_init,
            mamba_kw=config.mamba_kw()).to(self.device)
        self.opt = Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buf = RolloutBuffer(config.n_steps, self.N, obs_dim, act_dim,
                                 config.gamma, config.gae_lambda,
                                 burn_in=config.burn_in, device=self.device)
        self.num_timesteps = 0

    def _reset_states_on_done(self, states, done):
        keep_c = (1.0 - done).view(-1, 1, 1)
        keep_s = (1.0 - done).view(-1, 1, 1, 1)
        out = []
        for conv, ssm in states:
            out.append((conv * keep_c, ssm * keep_s))
        return out

    def collect_rollout(self, obs, states, episode_start):
        self.buf.reset()
        for _ in range(self.cfg.n_steps):
            action, logp, value, states = self.agent.step(obs, states)
            clipped = torch.clamp(action, self.act_low, self.act_high)
            next_obs, reward, done, _ = self.env.step(clipped.cpu().numpy())
            self.buf.add(obs, action, logp, _to_t(reward, self.device), value, episode_start)
            obs = _to_t(next_obs, self.device)
            done_t = _to_t(done, self.device)
            states = self._reset_states_on_done(states, done_t)   # zero state where episode ended
            episode_start = done_t                                # next step begins a new episode
            self.num_timesteps += self.N
        last_value, _ = self.agent.get_value(obs, states)
        self.buf.compute_gae(last_value, episode_start)
        self.buf.stash_context()
        return obs, states, episode_start

    def update(self):
        cfg = self.cfg
        stats = {}
        for epoch in range(cfg.n_epochs):
            kls = []
            for mb in self.buf.get(cfg.batch_envs):
                values, logp, entropy = self.agent.evaluate_actions(
                    mb["obs"], mb["actions"], mb["episode_starts"],
                    mb.get("context_obs"), mb.get("context_es"))
                adv = mb["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(logp - mb["logprobs"])
                pg = torch.max(-adv * ratio,
                               -adv * torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)).mean()
                v_clip = mb["values"] + torch.clamp(values - mb["values"], -cfg.clip_range, cfg.clip_range)
                v_loss = 0.5 * torch.max((values - mb["returns"]) ** 2,
                                         (v_clip - mb["returns"]) ** 2).mean()
                ent = entropy.mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kls.append(((ratio - 1) - (logp - mb["logprobs"])).mean().item())
                stats = dict(pg_loss=float(pg), v_loss=float(v_loss), entropy=float(ent))
            if cfg.target_kl is not None and np.mean(kls) > cfg.target_kl:
                break
        stats["approx_kl"] = float(np.mean(kls))
        return stats

    def learn(self, total_timesteps, callback=None):
        obs = _to_t(self.env.reset(), self.device)
        states = self.agent.initial_states(self.N, self.device)
        episode_start = torch.ones(self.N, device=self.device)
        n_updates = max(1, total_timesteps // (self.cfg.n_steps * self.N))
        for it in range(n_updates):
            obs, states, episode_start = self.collect_rollout(obs, states, episode_start)
            stats = self.update()
            if callback is not None:
                callback(self, it, stats)
        return self
