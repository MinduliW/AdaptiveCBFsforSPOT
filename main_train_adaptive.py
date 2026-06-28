"""
Per-step ADAPTIVE ICCBF gain tuning with RL (SPOT docking).

The policy runs INSIDE the 20 Hz control loop: every step it reads the live 28-D
state and outputs a fresh 10-D gain vector for the CLF-CBF-QP, so the gains adapt
through the maneuver. Plain PPO on SpotDockEnv (multi-step episodes, the env's
dense per-step reward); each episode randomizes the body poses (geometry fixed).

Structured like the lab's RL trainers: trainON/modelLoad flags, descriptive
training_name, TensorBoard under TrainedModels/, LR schedule, EvalCallback on the
5 named cases (saves best_model.zip), SubprocVecEnv + Monitor.

    python train_adaptive.py            # train (set trainON below) or validate
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

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from spot_env import SpotDockEnv, DT, Rmat, DOCK_OFF, wrap, propellant, N_CASES
from tune_gains import score

EPISODE_T = 150.0       # episode horizon [s] (docking can take ~100 s)

# POMDP / recurrent meta-RL variant: hide the target & obstacle so the policy sees
# ONLY the chaser's own state and an LSTM must INFER the hidden geometry over time.
# Use with MLPtype='RPPO' (RecurrentPPO). rl2 also feeds prev action+reward to the LSTM.
HIDDEN_OBS = True       # True: chaser-only observation (target/obstacle hidden)
HIDDEN_RL2 = True       # True: also feed previous action+reward (RL^2) to the LSTM
RESIDUAL_GAINS = True   # True: action -> constant baseline (1,0.5,0.25) +/- a small band
                        #   (action=0 == the 5/5 const gains; the policy only fine-tunes)


# ------------------------------ helpers ------------------------------------
def make_env(rank=0, seed=0, random_states=True):
    """Monitor-wrapped per-step env (geometry fixed).
    random_states=True  -> random pose layouts each episode;
    random_states=False -> sample one of the 5 named cases each episode."""
    def _init():
        env = SpotDockEnv(randomize=False, t_max=EPISODE_T,
                          pose_random=random_states, case_random=not random_states,
                          hidden_obs=HIDDEN_OBS, rl2=HIDDEN_RL2, residual_gains=RESIDUAL_GAINS)
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


# ------------------------- rollout / evaluation ----------------------------
def rollout_policy(model, venv, test_case=4, rnn=False):
    """Roll the per-step policy through a named case; log time-varying gains + metrics."""
    env = SpotDockEnv(randomize=False, test_case=test_case, t_max=EPISODE_T,
                      hidden_obs=HIDDEN_OBS, rl2=HIDDEN_RL2, residual_gains=RESIDUAL_GAINS)
    obs, _ = env.reset(seed=0)
    state = None; episode_start = np.ones(1, dtype=bool)
    keys = ("t", "xR", "xB", "xU", "rk_tar", "rk_obs",
            "h_tar", "h_obs", "h_los", "V", "u", "fov", "soff")
    log = {k: [] for k in keys}; gains_t = []
    mh = [1e9, 1e9, 1e9]; docked = False; t_dock = EPISODE_T; fuel = 0.0
    for _ in range(int(EPISODE_T / DT)):
        nobs = venv.normalize_obs(obs)
        if rnn:
            a, state = model.predict(nobs, state=state,
                                     episode_start=episode_start, deterministic=True)
            episode_start = np.zeros(1, dtype=bool)
        else:
            a, _ = model.predict(nobs, deterministic=True)
        g, k = env._decode(np.asarray(a).reshape(-1))
        gains_t.append(np.append(g.flatten(), k))
        log["t"].append(env.t)
        log["xR"].append(env.xR.copy()); log["xB"].append(env.xB.copy())
        log["xU"].append(env.xU.copy())
        log["rk_tar"].append(env.rkoz_tar.copy()); log["rk_obs"].append(env.r_koz_obs.copy())
        log["fov"].append(env.fov); log["soff"].append(env.sens_off.copy())
        obs, r, term, trunc, info = env.step(a)
        for kk in ("h_tar", "h_obs", "h_los", "V", "u"):
            log[kk].append(info[kk])
        mh[0] = min(mh[0], info["h_tar"]); mh[1] = min(mh[1], info["h_obs"])
        mh[2] = min(mh[2], info["h_los"]); fuel += propellant(info["u"]) * DT
        if info["docked"] and not docked:
            docked = True; t_dock = env.t
        if term or trunc:
            break
    for kk in log:
        log[kk] = np.asarray(log[kk])
    r_des = env.xB[:2] + Rmat(env.xB[2]) @ DOCK_OFF[:2]; th_des = wrap(env.xB[2] + DOCK_OFF[2])
    m = dict(docked=docked, t_dock=t_dock, t_end=env.t, fuel=fuel,
             mh_tar=mh[0], mh_obs=mh[1], mh_los=mh[2],
             dock_err=float(np.hypot(*(env.xR[:2] - r_des))),
             att_err=float(abs(wrap(env.xR[2] - th_des))))
    return log, np.asarray(gains_t), m


class FiveCaseEval(BaseCallback):
    """Every eval_freq steps: roll on cases 0-4, save best_model.zip by summed score."""
    def __init__(self, save_dir, rnn=False, eval_freq=25000, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir; self.rnn = rnn
        self.eval_freq = eval_freq; self.best = -1e18; self._last_eval = 0
        # Don't clobber a good best_model on restart: seed self.best from disk so a
        # fresh run's early (bad) evals can't overwrite a previously-saved 5/5 model.
        self._best_file = os.path.join(save_dir, "best_score.txt")
        if os.path.exists(self._best_file):
            try:
                self.best = float(open(self._best_file).read().strip())
                print("[eval] resuming best_score = %.0f (won't overwrite below this)" % self.best)
            except Exception:
                pass

    def _on_step(self):
        # eval_freq counts ENV TIMESTEPS (not vec-steps) so it's n_envs-agnostic
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        total = 0.0; ndock = 0
        for tc in range(N_CASES):
            _, _, m = rollout_policy(self.model, self.training_env, tc, self.rnn)
            total += score(m); ndock += int(m["docked"])
        self.logger.record("eval/docked", ndock)
        self.logger.record("eval/sum_score", total)
        new = total > self.best
        if new:
            self.best = total
            self.model.save(os.path.join(self.save_dir, "best_model.zip"))
            self.training_env.save(os.path.join(self.save_dir, "vecnormalize.pkl"))
            with open(self._best_file, "w") as fh:
                fh.write("%.4f" % self.best)
        if self.verbose:
            tag = "NEW BEST" if new else "best %.0f" % self.best
            print("[eval @ %d]  docked %d/%d  sum_score=%.0f  (%s)"
                  % (self.num_timesteps, ndock, N_CASES, total, tag))
        return True


class ExplorationAnneal(BaseCallback):
    """Shrink exploration over training so the policy can COMMIT once it finds a
    good region.

    Two coupled schedules over the first `anneal_steps` env-steps (then held):
      - ent_coef:  ent0 -> ent1   (remove the entropy bonus that was out-voting a
                   weak reward gradient and inflating std to ~1.8).
      - log_std cap: std_max0 -> std_max1, applied as a hard clamp on policy.log_std
                   (allow high early exploration that discovers docking, then force
                   sigma down so the deterministic eval stops thrashing).
    """
    def __init__(self, ent0=0.01, ent1=5e-4, std_max0=1.0, std_max1=0.3,
                 anneal_steps=10_000_000, verbose=0):
        super().__init__(verbose)
        self.ent0, self.ent1 = ent0, ent1
        self.std_max0, self.std_max1 = std_max0, std_max1
        self.anneal_steps = max(1, int(anneal_steps))

    def _on_step(self):
        frac = min(1.0, self.num_timesteps / self.anneal_steps)
        self.model.ent_coef = self.ent0 + frac * (self.ent1 - self.ent0)
        cap = self.std_max0 + frac * (self.std_max1 - self.std_max0)
        with torch.no_grad():                      # hard cap on exploration std
            self.model.policy.log_std.clamp_(max=float(np.log(cap)))
        if self.verbose and self.num_timesteps % 500000 < self.model.n_steps:
            self.logger.record("explore/ent_coef", self.model.ent_coef)
            self.logger.record("explore/std_cap", cap)
        return True


def report_rollout(model, venv, rnn=False):
    print("\nper-step adaptive policy on the 5 named cases:")
    for tc in range(N_CASES):
        log, gains_t, m = rollout_policy(model, venv, tc, rnn)
        spread = gains_t.std(axis=0)[:9].reshape(3, 3).mean(axis=1)   # gain motion during run
        print("  case %d: docked=%s t_dock=%.1fs fuel=%.2f  min h tar/obs/los=%+.2f/%+.2f/%+.2f"
              "  gain-var tar/obs/los=%.2f/%.2f/%.2f"
              % (tc, m["docked"], m["t_dock"], m["fuel"],
                 m["mh_tar"], m["mh_obs"], m["mh_los"], *spread))


# --------------------------- warm-start (BC) -------------------------------
# DE-optimal per-case gain vectors (actions) -- behavioral-cloning / baseline targets.
# REVERTED to the score-optimal version (case 0: k_dock=5.33, docks ~85 s, tar=[2.13,2.14,0.09]).
# Cases 0-3 are that version; case 4 kept as-is (that run covered 0-3 only; case 4 unused at N_CASES=3).
TARGET_THETA = {
    0: [0.4165, 0.4254, -0.837, 0.6396, -0.6167, 0.4087, -0.6947, -0.3537, -0.73, 0.7332],
    1: [0.5088, 0.4674, -0.1553, -0.2683, 0.4248, -0.0754, -0.0425, -0.9703, -0.9264, 0.6856],
    2: [0.703, 0.7391, -0.4738, -0.5837, -0.2028, 0.0871, 0.2974, -0.6883, -0.381, 0.8876],
    3: [0.1893, -0.3268, -0.1132, 0.9507, -0.967, -0.8558, -0.1675, 0.3631, 0.5141, -0.3458],
    4: [0.8059, -0.0911, -0.8733, -0.8061, 0.2774, -0.3306, 0.2187, -0.8016, -0.6056, 0.1777],
}


def collect_demos():
    """Roll each of the 5 cases with its DE-optimal CONSTANT gains; record
    (obs, action) pairs as behavioral-cloning demonstrations."""
    OBS, TGT = [], []
    for tc, theta in TARGET_THETA.items():
        env = SpotDockEnv(randomize=False, test_case=tc, t_max=EPISODE_T)
        obs, _ = env.reset(seed=0); a = np.array(theta, dtype=np.float32)
        for _ in range(int(EPISODE_T / DT)):
            OBS.append(obs.copy()); TGT.append(a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.asarray(OBS, np.float32), np.asarray(TGT, np.float32)


def pretrain(model, venv, epochs=300, batch=256, lr=1e-3):
    """Behavioral cloning: make the policy output the DE gains BEFORE PPO.
    Seeds VecNormalize obs stats on the demos, then MSE-fits the policy mean so
    the agent starts docking-capable and PPO refines from there."""
    obs_np, tgt_np = collect_demos()
    venv.obs_rms.update(obs_np)                          # seed obs normalization
    nobs = venv.normalize_obs(obs_np)
    dev = model.device
    obs_t = torch.as_tensor(nobs, dtype=torch.float32, device=dev)
    tgt_t = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    model.policy.train(); n = obs_t.shape[0]
    print("BC warm-start on %d demo steps ..." % n)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i+batch]
            mean = model.policy.get_distribution(obs_t[b]).distribution.mean
            loss = torch.mean((mean - tgt_t[b])**2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(b)
        if ep % 50 == 0 or ep == epochs-1:
            print("  BC epoch %d/%d  mse=%.4f" % (ep, epochs, tot/n))


# --------------------------------- main ------------------------------------
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)
    gc.collect()

    # ---- run flags ----
    trainON   = True       # train a new policy
    trainLoad = False       # RESUME training from the saved best_model (finish the 3229 run)
    modelLoad = True       # after training (or instead), load best + validate
    RANDOM_STATES = True   # True: random pose layouts; False: the 5 named cases only
                           # (evaluation/best-model is ALWAYS on the 5 named cases)
    WARMSTART = False       # behavioral-clone the DE gains before PPO (docking-capable start)

    num_env = 1 if is_debug_mode() else 16
    if not trainON:
        num_env = 1

    # ---- algorithm / hyper-params ----
    MLPtype       = 'RPPO'       # 'PPO' or 'RPPO' (RecurrentPPO/LSTM)
    rnn           = MLPtype == 'RPPO'
    total_timesteps = int(80e6)
    learning_rate = 3e-4
    lr_type       = 'D'          # C=constant, D=decreasing
    gamma         = 0.9997      # ~165 s effective horizon (terminal dock bonus must propagate)
    gae_lambda    = 0.95
    clip_range    = 0.2
    ent_coef      = 0.01        # INITIAL entropy bonus; annealed by ExplorationAnneal
    ANNEAL_EXPLORE = True       # decay ent_coef + cap log_std so the policy converges
    ent_coef_final = 5e-4       # entropy bonus at the end of the anneal window
    std_cap0, std_cap1 = 1.0, 0.3   # log_std cap: high early (explore) -> low late (commit)
    anneal_steps   = int(15e6)  # env-steps over which to anneal (then held); at ~7.5M
                                # (where the old run found 4/5) cap~0.65 -> still exploring,
                                # but no runaway to std~1.8; fully committed (cap 0.3) by 15M
    n_epochs      = 10
    batch_size    = 256
    n_steps       = 256 * num_env
    layers, nodes = 2, 128
    std           = 0.2
    activation_fn = activation.Tanh

    lstm_size     = 128         # LSTM hidden size (RPPO / hidden-obs inference)

    if rnn:   # the ONLY LSTM config that escaped 1/5 (reached 3229) -- resume & finish it.
        learning_rate = 1e-4    #   (every calibrated lr/n_steps/lstm variant stalled at 1/5)
        n_epochs      = 10
        n_steps       = 256 * num_env
        lstm_size     = 128     #   MUST match the saved 3229 model for trainLoad resume
        # (the MLP-PPO settings above are preserved for PPO runs)

    lr_schedule = (linear_schedule(learning_rate, learning_rate/100)
                   if lr_type == 'D' else constant_schedule(learning_rate))
    policy_kwargs = dict(activation_fn=activation_fn, ortho_init=True,
                         log_std_init=float(np.log(std)), optimizer_class=Adam,
                         net_arch=dict(pi=[nodes]*layers, vf=[nodes]*layers))
    if rnn:   # RecurrentPPO/MlpLstmPolicy: the LSTM is the belief-state that infers
        policy_kwargs.update(lstm_hidden_size=lstm_size, n_lstm_layers=1,
                             enable_critic_lstm=True)   # the hidden target/obstacle

    training_name = (f"{MLPtype}_adaptive_l{layers}_n{nodes}_lr{learning_rate}{lr_type}"
                     f"_std{std}_ne{n_epochs}_g{gamma}_ent{ent_coef}"
                     f"_{'warm' if WARMSTART else 'scratch'}"
                     f"_{'rand' if RANDOM_STATES else 'cases'}"
                     f"{'_anneal' if ANNEAL_EXPLORE else ''}"
                     f"{'_hidden' if HIDDEN_OBS else ''}{'rl2' if HIDDEN_RL2 else ''}"
                     f"{'_resid' if RESIDUAL_GAINS else ''}"
                     f"_prop_tof")   # propellant (torque-weighted) + one-sided TOF cap -> fresh dir
    print('Training:', training_name)

    tb_root = 'TrainedModels/'
    eval_log_dir = os.path.join(tb_root, training_name)
    os.makedirs(eval_log_dir, exist_ok=True)

    # ---- parallel, normalized envs ----
    VecCls = SubprocVecEnv if num_env > 1 else DummyVecEnv
    train_env = VecNormalize(VecCls([make_env(i, random_states=RANDOM_STATES)
                                     for i in range(num_env)]),
                             norm_obs=True, norm_reward=True, clip_obs=10.0)

    Algo = RecurrentPPO if rnn else PPO
    policy = "MlpLstmPolicy" if rnn else "MlpPolicy"
    model = Algo(policy, train_env, verbose=1, tensorboard_log=tb_root,
                 learning_rate=lr_schedule, n_steps=n_steps, batch_size=batch_size,
                 n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
                 clip_range=clip_range, ent_coef=ent_coef, normalize_advantage=True,
                 policy_kwargs=policy_kwargs)

    if trainLoad:
        print('Resuming from saved best_model + normalizer...')
        vp = os.path.join(eval_log_dir, 'vecnormalize.pkl')
        if os.path.exists(vp):                       # restore obs/ret running stats so the
            train_env = VecNormalize.load(vp, train_env.venv)   # resumed policy sees matching obs
            train_env.training = True; train_env.norm_reward = True
        model = Algo.load(os.path.join(eval_log_dir, 'best_model.zip'),
                          env=train_env, tensorboard_log=tb_root)

    if trainON:
        if WARMSTART and not trainLoad:
            print('--- BEHAVIORAL-CLONING WARM-START (DE gains) ---')
            pretrain(model, train_env)
        print('--- STARTING LEARNING ---  steps:', total_timesteps,
              ' batch:', batch_size*num_env, ' n_steps:', n_steps*num_env)
        callback = [FiveCaseEval(eval_log_dir, rnn=rnn)]   # eval every 25k steps (default)
        if ANNEAL_EXPLORE:
            callback.append(ExplorationAnneal(ent0=ent_coef, ent1=ent_coef_final,
                                              std_max0=std_cap0, std_max1=std_cap1,
                                              anneal_steps=anneal_steps, verbose=1))
        model.learn(total_timesteps=total_timesteps, callback=callback,
                    tb_log_name=training_name, progress_bar=True,
                    reset_num_timesteps=not trainLoad)   # resume: continue the step counter (+anneal)
        model.save(os.path.join(eval_log_dir, 'final_model.zip'))
        train_env.save(os.path.join(eval_log_dir, 'vecnormalize.pkl'))
        print('--- DONE LEARNING ---')

    if modelLoad:
        print('Loading best model for validation...')
        venv = VecNormalize.load(os.path.join(eval_log_dir, 'vecnormalize.pkl'),
                                 DummyVecEnv([make_env(0)]))
        venv.training = False; venv.norm_reward = False
        model = Algo.load(os.path.join(eval_log_dir, 'best_model.zip'))
        report_rollout(model, venv, rnn)
