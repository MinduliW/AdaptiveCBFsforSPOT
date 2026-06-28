"""
Mamba2 actor-critic for the SPOT docking PPO loop -- ported from the reference
`mamba2_cleanrl/agent.py`, with the CUDA `mamba_ssm.Mamba2` swapped for our
pure-PyTorch `Mamba2` (mamba_torch.py).

Stack (separate instances/state for actor and critic, no shared recurrence):
    obs -> proj(Linear obs_dim->d_model) -> Mamba2 -> MLP([nodes]*layers, Tanh)
        -> action_mean(Linear->act_dim) [actor]  /  value_head(Linear->1) [critic]
Actions are a diagonal Gaussian (mean, exp(log_std)); log_std is a learnable
Parameter clamped at -4 before exp. The PPO loop owns the recurrent state and
threads it through step()/evaluate_actions().
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .mamba_torch import Mamba2


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _mlp(d_in, hidden, act=nn.Tanh):
    layers, d = [], d_in
    for h in hidden:
        layers += [_ortho(nn.Linear(d, h), np.sqrt(2)), act()]
        d = h
    return nn.Sequential(*layers), d


class _Branch(nn.Module):
    """proj -> Mamba2 -> MLP. Carries (conv_state, ssm_state)."""
    def __init__(self, obs_dim, d_model, hidden, mamba_kw):
        super().__init__()
        self.proj = _ortho(nn.Linear(obs_dim, d_model), 1.0)
        self.mamba = Mamba2(d_model=d_model, **mamba_kw)
        self.mlp, self.d_out = _mlp(d_model, hidden)

    def initial_state(self, batch, device):
        return self.mamba.allocate_state(batch, device)

    def step(self, obs, state):
        conv, ssm = state
        h = self.proj(obs)
        h, conv, ssm = self.mamba.step(h, conv, ssm)
        return self.mlp(h), (conv, ssm)

    def scan(self, obs_seq, episode_starts):
        h = self.proj(obs_seq)
        h = self.mamba(h, episode_starts=episode_starts)
        return self.mlp(h)


class Mamba2ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=64, layers=2, nodes=128,
                 log_std_init=float(np.log(0.4)),
                 mamba_kw=None):
        super().__init__()
        mamba_kw = mamba_kw or dict(d_state=16, d_conv=4, expand=2, headdim=64, ngroups=1)
        hidden = [nodes] * layers
        self.actor = _Branch(obs_dim, d_model, hidden, mamba_kw)
        self.critic = _Branch(obs_dim, d_model, hidden, mamba_kw)
        self.action_mean = _ortho(nn.Linear(self.actor.d_out, act_dim), 0.01)
        self.value_head = _ortho(nn.Linear(self.critic.d_out, 1), 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    # ---- recurrent state ----
    def initial_states(self, batch, device):
        return [self.actor.initial_state(batch, device),
                self.critic.initial_state(batch, device)]

    def _dist(self, mean):
        std = torch.exp(self.log_std.clamp(min=-4.0))
        return Normal(mean, std)

    # ---- rollout (single timestep) ----
    @torch.no_grad()
    def step(self, obs, states, deterministic=False):
        a_state, c_state = states
        a_h, a_state = self.actor.step(obs, a_state)
        c_h, c_state = self.critic.step(obs, c_state)
        mean = self.action_mean(a_h)
        dist = self._dist(mean)
        action = mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.value_head(c_h).squeeze(-1)
        return action, logp, value, [a_state, c_state]

    @torch.no_grad()
    def get_value(self, obs, states):
        c_h, c_state = self.critic.step(obs, states[1])
        return self.value_head(c_h).squeeze(-1), [states[0], c_state]

    # ---- training (batched sequence) ----
    def evaluate_actions(self, obs_seq, actions, episode_starts,
                         context_obs=None, context_es=None):
        """obs_seq (B,T,obs_dim), actions (B,T,act_dim), episode_starts (B,T).
        Optional burn-in context_obs (B,K,obs_dim)/context_es (B,K) is PREPENDED to
        warm the scan, then its first K outputs are discarded.
        Returns values (B,T), log_probs (B,T), entropy (B,T)."""
        if context_obs is not None and context_obs.shape[1] > 0:
            K = context_obs.shape[1]
            obs_full = torch.cat([context_obs, obs_seq], dim=1)
            es_full = torch.cat([context_es, episode_starts], dim=1)
            a_h = self.actor.scan(obs_full, es_full)[:, K:]
            c_h = self.critic.scan(obs_full, es_full)[:, K:]
        else:
            a_h = self.actor.scan(obs_seq, episode_starts)
            c_h = self.critic.scan(obs_seq, episode_starts)
        dist = self._dist(self.action_mean(a_h))
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.value_head(c_h).squeeze(-1)
        return values, logp, entropy
