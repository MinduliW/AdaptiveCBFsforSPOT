"""
Pure-PyTorch Mamba-2 (SSD) block -- a CPU/Mac-runnable drop-in for
`mamba_ssm.modules.mamba2.Mamba2`, so the SPOT Mamba RL path needs no CUDA
kernels (mamba-ssm / causal-conv1d).

It mirrors the reference Mamba2 data flow:
    u -> in_proj -> [z | xBC | dt]
       xBC -> causal depthwise conv1d -> SiLU -> split into x, B, C
       selective SSM scan (input-dependent A,B,C,dt) over the time axis
       y = scan + D*x  ->  gated RMSNorm(y, z)  ->  out_proj
and exposes the two paths PPO needs:
    forward(u, episode_starts)         parallel/training scan over a sequence
    step(u, conv_state, ssm_state)     one-timestep stateful rollout

The selective scan here is an explicit sequential recurrence (no CUDA selective-
scan kernel), so step() applied token-by-token is numerically identical to
forward() on the same sequence -- which is the correctness test we run.

State shapes (for d_model=64, expand=2, headdim=64, d_state=16, d_conv=4):
    conv_state : (B, conv_dim=160, d_conv=4)
    ssm_state  : (B, nheads=2, headdim=64, d_state=16)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """Mamba-2 gated RMSNorm:  rmsnorm(x * silu(z))."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x, z):
        x = x * F.silu(z)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class Mamba2(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=64, ngroups=1,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0, "d_model*expand must be divisible by headdim"
        self.nheads = self.d_inner // headdim
        self.conv_dim = self.d_inner + 2 * ngroups * d_state

        # in_proj -> [z (d_inner) | xBC (conv_dim) | dt (nheads)]
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias)
        # depthwise causal conv over xBC
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=d_conv,
                                groups=self.conv_dim, padding=d_conv - 1, bias=conv_bias)

        # dt bias (inverse-softplus of a log-uniform dt), A = -exp(A_log), D skip
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))                 # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ---- state allocation ----
    def allocate_state(self, batch, device=None, dtype=torch.float32):
        device = device or self.in_proj.weight.device
        conv_state = torch.zeros(batch, self.conv_dim, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state

    def _gbroad(self, t):
        """(B,*,ngroups,d_state) -> (B,*,nheads,d_state) by repeating groups across heads."""
        return t.repeat_interleave(self.nheads // self.ngroups, dim=-2)

    def _conv_reset(self, xBC, episode_starts):
        """Causal depthwise conv that does NOT read across episode boundaries -- kernel
        taps whose source token belongs to an earlier episode are masked out (the
        pure-torch analogue of mamba-ssm's seq_idx conv reset)."""
        B, L, C = xBC.shape
        xpad = F.pad(xBC.transpose(1, 2), (self.d_conv - 1, 0))            # (B,C,L+k-1)
        win = xpad.unfold(-1, self.d_conv, 1)                              # (B,C,L,d_conv)
        w = self.conv1d.weight.squeeze(1)                                  # (C,d_conv)
        si = torch.cumsum(episode_starts, dim=1) - 1.0                     # (B,L) episode index
        si_pad = F.pad(si, (self.d_conv - 1, 0), value=-2.0)               # invalid pad never matches
        mask = (si_pad.unfold(-1, self.d_conv, 1) == si.unsqueeze(-1)).to(xBC.dtype)  # (B,L,d_conv)
        out = (win * w.view(1, C, 1, self.d_conv) * mask.unsqueeze(1)).sum(-1)        # (B,C,L)
        if self.conv1d.bias is not None:
            out = out + self.conv1d.bias.view(1, C, 1)
        return out.transpose(1, 2)

    # ---- training / sequence path ----
    def forward(self, u, episode_starts=None):
        """u: (B, L, d_model);  episode_starts: (B, L) 1.0 where a new episode begins.
        Returns y: (B, L, d_model). Scans from a zero state, resetting it at boundaries."""
        B, L, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)
        if episode_starts is None:                                        # plain causal depthwise conv
            xBC = self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2)
        else:                                                             # reset conv across boundaries
            xBC = self._conv_reset(xBC, episode_starts)
        xBC = F.silu(xBC)
        x, Bm, Cm = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state,
                                      self.ngroups * self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)                                 # (B,L,nheads)
        A = -torch.exp(self.A_log)                                         # (nheads,)
        x = x.view(B, L, self.nheads, self.headdim)
        Bm = self._gbroad(Bm.view(B, L, self.ngroups, self.d_state))        # (B,L,nheads,d_state)
        Cm = self._gbroad(Cm.view(B, L, self.ngroups, self.d_state))
        dA = torch.exp(dt * A)                                             # (B,L,nheads)

        state = u.new_zeros(B, self.nheads, self.headdim, self.d_state)
        ys = []
        for t in range(L):
            if episode_starts is not None:
                keep = (1.0 - episode_starts[:, t]).view(B, 1, 1, 1)        # zero state at boundaries
                state = state * keep
            dBx = (dt[:, t].view(B, self.nheads, 1, 1)
                   * x[:, t].unsqueeze(-1)                                  # (B,h,p,1)
                   * Bm[:, t].unsqueeze(2))                                 # (B,h,1,n) -> (B,h,p,n)
            state = dA[:, t].view(B, self.nheads, 1, 1) * state + dBx
            ys.append((state * Cm[:, t].unsqueeze(2)).sum(-1))             # (B,h,p)
        y = torch.stack(ys, dim=1)                                         # (B,L,h,p)
        y = y + x * self.D.view(1, 1, self.nheads, 1)
        y = y.reshape(B, L, self.d_inner)
        y = self.norm(y, z)
        return self.out_proj(y)

    # ---- single-timestep rollout path ----
    def step(self, u, conv_state, ssm_state):
        """u: (B, d_model). Advances conv_state/ssm_state by one token. Returns (y, conv_state, ssm_state)."""
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state = conv_state.clone()
        conv_state[:, :, -1] = xBC
        w = self.conv1d.weight.squeeze(1)                                  # (conv_dim, d_conv)
        xBC = (conv_state * w).sum(-1)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = F.silu(xBC)
        x, Bm, Cm = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state,
                                      self.ngroups * self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)
        x = x.view(-1, self.nheads, self.headdim)
        Bm = self._gbroad(Bm.view(-1, self.ngroups, self.d_state))
        Cm = self._gbroad(Cm.view(-1, self.ngroups, self.d_state))
        dA = torch.exp(dt * A)
        dBx = dt.unsqueeze(-1).unsqueeze(-1) * x.unsqueeze(-1) * Bm.unsqueeze(2)
        ssm_state = dA.view(-1, self.nheads, 1, 1) * ssm_state + dBx
        y = (ssm_state * Cm.unsqueeze(2)).sum(-1)
        y = y + x * self.D.view(1, self.nheads, 1)
        y = y.reshape(-1, self.d_inner)
        y = self.norm(y, z)
        return self.out_proj(y), conv_state, ssm_state
