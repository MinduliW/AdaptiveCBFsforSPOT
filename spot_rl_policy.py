"""
MATLAB <-> Python bridge for the SPOT RL gain policy (pyenv co-execution).

Mirrors trc_mpc.trcmpc.solve_trcmpc: the SPOT emulator (MATLAB/Simulink) calls
get_gains(x_red, x_black, x_obstacle, holding_radius) every control step and gets
back the RL-chosen class-K gains (3x3) + k_dock, which YOUR CBF-CLF QP then uses.

Internally it runs the SAME pipeline as training (no QP here -- that stays in MATLAB):
    states -> obs (28) -> VecNormalize -> MLP -> gains (3x3) + k_dock

MATLAB side (call_python_policy.m) and Run_Initializer.m setup are alongside this file.
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import main_train_MLP as M
from spot_env import SpotDockEnv

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DIR = os.environ.get("SPOT_RL_MODEL", os.path.join(
    _ROOT, "TrainedModels",
    "PPO_adaptive_l2_n128_lr0.0003D_std0.2_ne10_g0.9997_ent0.01_scratch_rand_anneal_resid_prop_tof"))

_model = _venv = _env = None


def _load():
    """Load the policy + normalizer + env ONCE (cached, like the MATLAB persistent)."""
    global _model, _venv, _env
    if _model is None:
        M.HIDDEN_OBS = False; M.HIDDEN_RL2 = False; M.RESIDUAL_GAINS = True
        _venv = VecNormalize.load(os.path.join(_DIR, "vecnormalize.pkl"),
                                  DummyVecEnv([M.make_env(0)]))
        _venv.training = False
        _model = PPO.load(os.path.join(_DIR, "best_model.zip"))
        _env = SpotDockEnv(randomize=False, hidden_obs=False, residual_gains=True)
        _env.reset(seed=0)                 # sets nominal KOZ / FOV / sensor params
    return _model, _venv, _env


def get_gains(x_red, x_black, x_obstacle, holding_radius=None):
    """One time step: states -> the RL-chosen class-K gains (NO QP -- gains only).
    x_*: 6-vectors [x y theta dx dy dtheta] (lab frame).
    holding_radius: scalar or [a,b] current target keep-out size (optional).
    Returns a dict: gains (3,3) rows=tar,obs,los cols=a0,a1,a2 ; k_dock (float) ; success."""
    model, venv, env = _load()
    env.xR = np.asarray(x_red, float).ravel()[:6].copy()
    env.xB = np.asarray(x_black, float).ravel()[:6].copy()
    env.xU = np.asarray(x_obstacle, float).ravel()[:6].copy()
    if holding_radius is not None:
        hr = np.asarray(holding_radius, float).ravel()
        env.rkoz_tar = (hr[:2] if hr.size >= 2 else np.array([hr[0], hr[0]])).copy()

    obs = env._obs()
    nobs = venv.normalize_obs(obs.reshape(1, -1))
    action, _ = model.predict(nobs, deterministic=True)
    gains, k_dock = env._decode(np.asarray(action, float).ravel())

    return dict(gains=np.asarray(gains, float), k_dock=float(k_dock), success=True)


if __name__ == "__main__":          # smoke test (run as a normal python script)
    from spot_env import TEST_CASES
    xR, xB, xU = TEST_CASES[2]
    out = get_gains(xR, xB, xU)
    print("k_dock = %.3f" % out["k_dock"])
    print("gains (rows tar,obs,los ; cols a0,a1,a2) =\n", np.round(out["gains"], 3))
