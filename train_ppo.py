"""Train an RL policy to tune the ICCBF class-K gains for the SPOT docking env.

The policy maps the 28-D observation -> 10 action dims (per-constraint class-K
coefs a0,a1 + barrier-decay slack a2, for tar-KOZ / obs-KOZ / LOS, plus one CLF
Lslack). The CLF-CBF-QP inside the env turns those into safe thrust. Each
episode the env randomizes the *hidden* task geometry (KOZ sizes, FOV, sensor
offsets, target spin & approach), so a recurrent policy can meta-learn to infer
it -- mirroring MetaRL_for_ICCBFs/src/docking/train_DCRNN.py.

Usage:
    python train_ppo.py                 # MLP PPO (default)
    python train_ppo.py --rnn           # LSTM RecurrentPPO (meta-RL, sb3-contrib)
    python train_ppo.py --smoke         # quick end-to-end check
    python train_ppo.py --rnn --smoke   # quick RNN check
"""
import sys, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from spot_env import SpotDockEnv, T_MAX, DT

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None


def make_env(seed=0):
    def _init():
        env = Monitor(SpotDockEnv())
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init


def build_model(venv, rnn, smoke):
    # shared on-policy hyper-params (match the MetaRL docking setup)
    common = dict(verbose=1, gae_lambda=0.95, gamma=0.995,
                  ent_coef=0.003, learning_rate=3e-4)
    if rnn:
        if RecurrentPPO is None:
            raise ImportError("sb3-contrib is required for --rnn: pip install sb3-contrib")
        return RecurrentPPO(
            "MlpLstmPolicy", venv,
            n_steps=256 if smoke else 1024, batch_size=128,
            policy_kwargs=dict(net_arch=[128], lstm_hidden_size=128,
                               enable_critic_lstm=True),
            **common), "rppo_spot_gains"
    return PPO(
        "MlpPolicy", venv,
        n_steps=512 if smoke else 4096, batch_size=128,
        policy_kwargs=dict(net_arch=[128, 128]),
        **common), "ppo_spot_gains"


def main(smoke=False, rnn=False, n_envs=4):
    n_envs = 1 if smoke else n_envs
    VecCls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    venv = VecCls([make_env(i) for i in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model, tag = build_model(venv, rnn, smoke)
    model.learn(total_timesteps=3000 if smoke else 1_000_000)
    model.save(tag); venv.save("vecnorm.pkl")

    rollout(model, rnn)


def rollout(model, rnn):
    """Roll out the trained policy and report safety margins + settled gains."""
    env = SpotDockEnv(); obs, _ = env.reset(seed=1)
    state = None; episode_start = np.ones(1, dtype=bool)
    hmin = [1e9, 1e9, 1e9]; coef_log = []
    for _ in range(int(T_MAX / DT)):
        if rnn:
            a, state = model.predict(obs, state=state,
                                     episode_start=episode_start, deterministic=True)
            episode_start = np.zeros(1, dtype=bool)
        else:
            a, _ = model.predict(obs, deterministic=True)
        g, k_dock = env._decode(a)
        coef_log.append(np.append(g.flatten(), k_dock))
        obs, _, term, trunc, info = env.step(a)
        hmin[0] = min(hmin[0], info['h_tar'])
        hmin[1] = min(hmin[1], info['h_obs'])
        hmin[2] = min(hmin[2], info['h_los'])
        if term or trunc:
            print(f"rollout end t={env.t:.1f} docked={info['docked']} V={info['V']:.3f}")
            break
    g = np.mean(coef_log, axis=0)
    print("min margins  h_tar=%.3f h_obs=%.3f h_los=%.3f" % tuple(hmin))
    print("mean learned gains (rows: tarKOZ, obsKOZ, LOS ; cols a0 a1 a2):")
    print(np.round(g[:9].reshape(3, 3), 3))
    print("mean CLF decay k_dock = %.3f" % g[9])


if __name__ == "__main__":
    main(smoke=("--smoke" in sys.argv), rnn=("--rnn" in sys.argv))
