"""
RL bandit for CONSTANT per-test-case ICCBF gains (SPOT docking).

A 1-step contextual bandit: each episode samples one of the 5 named test cases,
the observation is that case's initial config (states of all 3 bodies + barriers
h_tar/h_obs/h_los + the docking Lyapunov V), and the policy outputs ONE constant
10-D gain vector held for the whole rollout. Reward is tune_gains.score() over the
rollout (the +1000 dock bonus is direct -- 1-step, no discounting). Trained across
the 5 cases, the policy learns the best constant gains for each; evaluation rolls
out all 5 and saves the best policy.

Same harness format as main_train_adaptive.py (trainON/modelLoad flags,
training_name, TensorBoard, LR schedule, policy_kwargs, SubprocVecEnv + Monitor).
"""
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
sys.path.append('/Users/minduli/miniconda3/lib/python3.12/site-packages/')

import os
import gc
import warnings
import multiprocessing
from typing import Callable

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.modules import activation
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from spot_env import (SpotDockEnv, koz_row, los_row, clf_rows, mRED, IRED,
                      sample_pose_config)
from tune_gains import rollout_metrics, score

N_CASES = 5


# ------------------------- config observation ------------------------------
def config_obs(env):
    """Observation = states of all 3 bodies + barrier values h + Lyapunov V (22-D)."""
    xR, xB, xU = env.xR, env.xB, env.xU
    _, _, h_tar = koz_row(xR, xB, env.rkoz_tar, 1, 1, 1, mRED)
    _, _, h_obs = koz_row(xR, xU, env.r_koz_obs, 1, 1, 1, mRED)
    _, _, h_los = los_row(xR, xB, 1, 1, 1, IRED, 1,
                          fov=env.fov, sens_off=env.sens_off, sens_tgt=env.sens_tgt)
    _, _, V, _ = clf_rows(xR, xB, 1)
    return np.concatenate([xR, xB, xU, [h_tar, h_obs, h_los, V]]).astype(np.float32)


def case_obs(tc):
    env = SpotDockEnv(randomize=False, test_case=tc); env.reset(seed=0)
    return config_obs(env)


# ------------------------------- bandit ------------------------------------
class CaseGainBandit(gym.Env):

    def __init__(self, random_states=False):
        super().__init__()
        self.random_states = random_states
        self.action_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float32)
        self._probe = SpotDockEnv(randomize=False)   # to build the obs for random configs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_states:
            self.cfg = sample_pose_config(self.np_random); self.tc = None
            self._probe.ic = self.cfg; self._probe.reset(seed=0)
            self.obs = config_obs(self._probe)
        else:
            self.tc = int(self.np_random.integers(0, N_CASES)); self.cfg = None
            self.obs = case_obs(self.tc)
        return self.obs, {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        m = (rollout_metrics(a, ic=self.cfg) if self.random_states
             else rollout_metrics(a, test_case=self.tc))
        return self.obs, float(score(m)), True, False, m


# ------------------------------ helpers ------------------------------------
def make_env(rank=0, seed=0, random_states=False):
    def _init():
        env = CaseGainBandit(random_states=random_states)
        env.action_space.seed(seed + rank)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def linear_schedule(initial_value: float, min_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, min_value)
    return func


def constant_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value
    return func


def is_debug_mode():
    return sys.gettrace() is not None


# ------------------------- evaluation / reporting --------------------------
class FiveCaseEval(BaseCallback):
    """Every eval_freq steps: predict gains for each of the 5 cases, save best by score."""
    def __init__(self, save_dir, eval_freq=4000, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir; self.eval_freq = eval_freq
        self.best = -1e18; self._last_eval = 0

    def _on_step(self):
        # eval_freq counts ENV TIMESTEPS (not vec-steps) so it's n_envs-agnostic
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        total = 0.0; ndock = 0
        for tc in range(N_CASES):
            a, _ = self.model.predict(self.training_env.normalize_obs(case_obs(tc)),
                                      deterministic=True)
            m = rollout_metrics(np.asarray(a).reshape(-1), test_case=tc)
            total += score(m); ndock += int(m["docked"])
        self.logger.record("eval/docked", ndock)
        self.logger.record("eval/sum_score", total)
        new = total > self.best
        if new:
            self.best = total
            self.model.save(os.path.join(self.save_dir, "best_model.zip"))
            self.training_env.save(os.path.join(self.save_dir, "vecnormalize.pkl"))
        if self.verbose:
            tag = "NEW BEST" if new else "best %.0f" % self.best
            print("[eval @ %d]  docked %d/5  sum_score=%.0f  (%s)"
                  % (self.num_timesteps, ndock, total, tag))
        return True


def report(model, venv, save_dir=None):
    
    """Print the learned constant gains + metrics for each of the 5 cases, and
    (if save_dir) write them to <save_dir>/gains.npz + gains.txt."""
    
    print("\nlearned constant gains per case:")
    
    thetas, gains, kdocks, lines = [], [], [], []
    for tc in range(N_CASES):
        a, _ = model.predict(venv.normalize_obs(case_obs(tc)), deterministic=True)
        theta = np.asarray(a).reshape(-1)
        g, k = SpotDockEnv()._decode(theta); m = rollout_metrics(theta, test_case=tc)
        line = ("case %d: docked=%s t_dock=%.1fs fuel=%.2f  tar=%s obs=%s los=%s k_dock=%.2f"
                % (tc, m["docked"], m["t_dock"], m["fuel"],
                   np.round(g[0], 3), np.round(g[1], 3), np.round(g[2], 3), k))
        print("  " + line)
        thetas.append(theta); gains.append(g); kdocks.append(k); lines.append(line)
    if save_dir is not None:
        np.savez(os.path.join(save_dir, "gains.npz"),
                 theta=np.array(thetas), gains=np.array(gains), k_dock=np.array(kdocks))
        with open(os.path.join(save_dir, "gains.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("saved gains ->", os.path.join(save_dir, "gains.npz"))


# --------------------------------- main ------------------------------------
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    # ---- run flags ----
    trainON   = True       # train a new policy
    trainLoad = False      # warm-start from the saved best_model
    modelLoad = True       # after training (or instead), load best + report gains
    RANDOM_STATES = False   # True: train on random pose layouts; False: the 5 named cases
                           # (evaluation/best-model is ALWAYS on the 5 named cases)

    num_env = 1 if is_debug_mode() else 16
    if not trainON:
        num_env = 1

    # ---- algorithm / hyper-params (1-step bandit) ----
    MLPtype       = 'PPO'        # 'PPO' or 'RPPO'
    rnn           = MLPtype == 'RPPO'
    total_timesteps = int(4e5)   # bandit episodes (each = one full rollout)
    learning_rate = 3e-4
    lr_type       = 'D'
    gamma         = 0.0          # 1-step episodes: return == immediate reward (no bootstrap)
    gae_lambda    = 0.0
    clip_range    = 0.2
    ent_coef      = 0.01
    n_epochs      = 10
    batch_size    = 64
    n_steps       = batch_size * 4
    layers, nodes = 2, 64
    std           = 0.5
    activation_fn = activation.Tanh

    lr_schedule = (linear_schedule(learning_rate, learning_rate/100)
                   if lr_type == 'D' else constant_schedule(learning_rate))
    policy_kwargs = dict(activation_fn=activation_fn, ortho_init=True,
                         log_std_init=float(np.log(std)), optimizer_class=Adam,
                         net_arch=dict(pi=[nodes]*layers, vf=[nodes]*layers))

    training_name = (f"{MLPtype}_casegains_l{layers}_n{nodes}_lr{learning_rate}{lr_type}"
                     f"_std{std}_ne{n_epochs}_ent{ent_coef}")
    print('Training:', training_name)

    # one self-contained run folder: TB logs + networks + vecnorm + gains data
    OUTPUT_ROOT = 'runs/'
    run_dir = os.path.join(OUTPUT_ROOT, training_name)
    os.makedirs(run_dir, exist_ok=True)

    VecCls = SubprocVecEnv if num_env > 1 else DummyVecEnv
    train_env = VecNormalize(VecCls([make_env(i, random_states=RANDOM_STATES)
                                     for i in range(num_env)]),
                             norm_obs=True, norm_reward=True, clip_obs=10.0)

    Algo = RecurrentPPO if rnn else PPO
    policy = "MlpLstmPolicy" if rnn else "MlpPolicy"
    model = Algo(policy, train_env, verbose=1, tensorboard_log=run_dir,
                 learning_rate=lr_schedule, n_steps=n_steps, batch_size=batch_size,
                 n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
                 clip_range=clip_range, ent_coef=ent_coef, normalize_advantage=True,
                 policy_kwargs=policy_kwargs)

    if trainLoad:
        print('Warm-starting from best_model...')
        model = Algo.load(os.path.join(run_dir, 'best_model.zip'),
                          env=train_env, tensorboard_log=run_dir)

    if trainON:
        print('--- STARTING LEARNING ---  episodes:', total_timesteps, ' ->', run_dir)
        callback = FiveCaseEval(run_dir)
        model.learn(total_timesteps=total_timesteps, callback=callback,
                    tb_log_name='tb', progress_bar=True)
        model.save(os.path.join(run_dir, 'final_model.zip'))
        train_env.save(os.path.join(run_dir, 'vecnormalize.pkl'))
        print('--- DONE LEARNING ---')

    if modelLoad:
        print('Loading best model...')
        venv = VecNormalize.load(os.path.join(run_dir, 'vecnormalize.pkl'),
                                 DummyVecEnv([make_env(0)]))
        venv.training = False; venv.norm_reward = False
        model = Algo.load(os.path.join(run_dir, 'best_model.zip'))
        report(model, venv, run_dir)
