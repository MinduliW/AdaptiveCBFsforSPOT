"""
Standalone numpy reference for the SPOT docking MLP policy.
No PyTorch / Stable-Baselines3 needed -- only numpy + scipy (for the .mat).

This IS the ground-truth pipeline to reproduce in Simulink:

    raw obs (28) --[normalize]--> --[MLP]--> action (10) --[decode]--> gains (3x3) + k_dock

All weights and constants live in spot_policy.mat. Run this file to self-check
against reference_io.csv.
"""
import os
import numpy as np
import scipy.io as sio

_HERE = os.path.dirname(os.path.abspath(__file__))
_D = sio.loadmat(os.path.join(_HERE, "spot_policy.mat"))


def normalize(obs):
    """Stage 2 -- VecNormalize: z-score with saved mean/std, then clip to +-clip_obs."""
    obs = np.asarray(obs, float).reshape(-1, 1)
    return np.clip((obs - _D["obs_mean"]) / _D["obs_std"], -float(_D["clip_obs"]), float(_D["clip_obs"]))


def mlp(obs_norm):
    """Stage 3 -- the trained network: Linear(28,128)->tanh->Linear(128,128)->tanh->Linear(128,10)."""
    h = np.tanh(_D["W1"] @ obs_norm + _D["b1"])
    h = np.tanh(_D["W2"] @ h + _D["b2"])
    return _D["W3"] @ h + _D["b3"]                      # (10,1) action mean (UNclipped)


def action(obs):
    """raw obs (28,) -> action (10,)  [stages 2+3]."""
    return mlp(normalize(obs)).reshape(-1)


def decode(a):
    """Stage 4 -- residual decode: action (10,) -> gains g[3x3] (rows tar,obs,los;
    cols a0,a1,a2) and the CLF decay k_dock."""
    a = np.clip(np.asarray(a, float).reshape(-1), -1.0, 1.0)
    grid = a[0:9].reshape(3, 3)
    base = _D["BASE"].reshape(-1)
    band = _D["BAND"].reshape(-1)
    g = base[None, :] + band[None, :] * grid
    g[:, 0:2] = np.clip(g[:, 0:2], 0.0, float(_D["ACOEF_HI"]))
    g[:, 2] = np.clip(g[:, 2], float(_D["HSLACK_LO"]), float(_D["HSLACK_HI"]))
    k_dock = max(0.1, float(_D["KDOCK"]) + float(_D["KBAND"]) * a[9])
    return g, k_dock


def gains(obs):
    """Full pipeline: raw obs (28,) -> gains g[3x3], k_dock  [stages 2+3+4]."""
    return decode(action(obs))


if __name__ == "__main__":
    ref = np.loadtxt(os.path.join(_HERE, "reference_io.csv"), delimiter=",", skiprows=1)
    obs, act_ref = ref[:, :28], ref[:, 28:38]
    err = max(np.abs(action(o) - a).max() for o, a in zip(obs, act_ref))
    print("reference_io.csv: %d rows;  max|action - reference| = %.2e" % (len(ref), err))
    assert err < 1e-4, "MISMATCH -- the network or constants don't match the reference"
    g, k = gains(obs[0])
    print("OK. example gains(obs[0]):\n  tar=%s obs=%s los=%s  k_dock=%.3f"
          % (np.round(g[0], 3), np.round(g[1], 3), np.round(g[2], 3), k))
