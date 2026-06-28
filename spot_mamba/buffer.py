"""
Time-major rollout buffer with GAE and burn-in, for the Mamba PPO loop.

Storage is (n_steps, n_envs, *). episode_starts[t] = done at the START of step t
(i.e. the previous step terminated), so the Mamba scan resets its state there.
get() yields minibatches over ENVIRONMENTS (not transitions) -- each minibatch is
a set of full-length (T) sequences (B_envs, T, *) for the parallel scan. Burn-in
prepends the previous rollout's last `burn_in` obs/episode_starts so the training
scan warms its (zero-init) state before the real chunk, mirroring the reference.
"""
import torch


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, act_dim, gamma, gae_lambda,
                 burn_in=0, device="cpu"):
        self.T, self.N = n_steps, n_envs
        self.gamma, self.lam, self.burn_in = gamma, gae_lambda, burn_in
        self.device = device
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs = z(self.T, self.N, obs_dim)
        self.actions = z(self.T, self.N, act_dim)
        self.logprobs = z(self.T, self.N)
        self.rewards = z(self.T, self.N)
        self.values = z(self.T, self.N)
        self.episode_starts = z(self.T, self.N)
        # rolling context for burn-in (last `burn_in` obs/episode_starts of prev rollout)
        self.ctx_obs = z(max(burn_in, 1), self.N, obs_dim)
        self.ctx_es = z(max(burn_in, 1), self.N)
        self.pos = 0

    def reset(self):
        self.pos = 0

    def add(self, obs, action, logprob, reward, value, episode_start):
        i = self.pos
        self.obs[i] = obs; self.actions[i] = action; self.logprobs[i] = logprob
        self.rewards[i] = reward; self.values[i] = value; self.episode_starts[i] = episode_start
        self.pos += 1

    @torch.no_grad()
    def compute_gae(self, last_value, last_episode_start):
        """Standard GAE; next-step nonterminal mask uses episode_starts[t+1]."""
        adv = torch.zeros_like(self.rewards)
        last = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                nonterm = 1.0 - last_episode_start
                next_v = last_value
            else:
                nonterm = 1.0 - self.episode_starts[t + 1]
                next_v = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_v * nonterm - self.values[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        self.advantages = adv
        self.returns = adv + self.values

    def stash_context(self):
        """Keep this rollout's tail as burn-in context for the next rollout's update."""
        if self.burn_in > 0:
            self.ctx_obs = self.obs[-self.burn_in:].clone()
            self.ctx_es = self.episode_starts[-self.burn_in:].clone()

    def get(self, batch_envs):
        """Yield env-major minibatches: each a dict of (B_envs, [K+]T, *) tensors."""
        perm = torch.randperm(self.N, device=self.device)
        tm = lambda x, e: x[:, e].transpose(0, 1)        # (T,n,*) -> (n,T,*)
        for s in range(0, self.N, batch_envs):
            e = perm[s:s + batch_envs]
            mb = dict(
                obs=tm(self.obs, e), actions=tm(self.actions, e),
                logprobs=tm(self.logprobs, e), advantages=tm(self.advantages, e),
                returns=tm(self.returns, e), values=tm(self.values, e),
                episode_starts=tm(self.episode_starts, e),
            )
            if self.burn_in > 0:
                mb["context_obs"] = tm(self.ctx_obs, e)
                mb["context_es"] = tm(self.ctx_es, e)
            yield mb
