"""Config for the SPOT Mamba PPO -- mirrors our SB3 LSTM settings + the reference Mamba dims."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SpotMambaConfig:
    # Mamba2 core (reference docking preset; d_model*expand must be divisible by headdim)
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    # MLP heads
    layers: int = 2
    nodes: int = 128
    log_std_init: float = float(np.log(0.4))
    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.9997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.05
    n_epochs: int = 10
    n_steps: int = 256          # rollout chunk length (<< episode; burn-in warms the scan)
    burn_in: int = 64
    batch_envs: int = 4         # minibatch is in ENVIRONMENTS; needs num_env > batch_envs
    # run
    num_env: int = 16
    total_timesteps: int = int(20e6)

    def mamba_kw(self):
        return dict(d_state=self.d_state, d_conv=self.d_conv, expand=self.expand,
                    headdim=self.headdim, ngroups=self.ngroups)
