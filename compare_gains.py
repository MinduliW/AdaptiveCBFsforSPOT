"""
Compare four controllers on the 5 named cases:
  - MLP        : adaptive gains, full obs
  - LSTM       : adaptive gains, blind/chaser-only obs
  - Constant   : fixed a0,a1,a2 = 1,0.5,0.25 on every CBF, k_dock=K_DOCK
  - DE-optimized: per-case constant gains found by differential evolution (TARGET_THETA)
All four optimise/score the SAME env reward, so gains and reward are directly comparable.

Writes:
  figures/adaptive_gains_MLP_vs_LSTM.png    (gains: MLP | LSTM | Constant | DE-opt)
  figures/reward_MLP_vs_LSTM.png            (per-step + cumulative reward, 4 lines)

    python compare_gains.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import main_train_adaptive as M
from spot_env import SpotDockEnv, DT, K_DOCK
from main_train_adaptive import EPISODE_T, TARGET_THETA

LABELS = ["tar_a0", "tar_a1", "tar_a2", "obs_a0", "obs_a1", "obs_a2",
          "los_a0", "los_a1", "los_a2", "k_dock"]
MLP_DIR  = "BestModels/adaptive_5of5_4457_FROMSCRATCH"
LSTM_DIR = "BestModels/adaptive_lstm_5of5_4284_blind"
CONST_GAINS = (1.0, 0.5, 0.25)


def roll_one(model, venv, tc, rnn):
    env = SpotDockEnv(randomize=False, test_case=tc, t_max=EPISODE_T,
                      hidden_obs=M.HIDDEN_OBS, rl2=M.HIDDEN_RL2)
    obs, _ = env.reset(seed=0)
    state = None; es = np.ones(1, dtype=bool)
    ts, gains, rews = [], [], []; docked = False
    for _ in range(int(EPISODE_T / DT)):
        nobs = venv.normalize_obs(obs)
        if rnn:
            a, state = model.predict(nobs, state=state, episode_start=es, deterministic=True)
            es = np.zeros(1, dtype=bool)
        else:
            a, _ = model.predict(nobs, deterministic=True)
        g, k = env._decode(np.asarray(a).reshape(-1))
        ts.append(env.t); gains.append(np.append(g.flatten(), k))
        obs, r, term, trunc, info = env.step(a)
        rews.append(r); docked = docked or info["docked"]
        if term or trunc:
            break
    return np.asarray(ts), np.asarray(gains), np.asarray(rews), docked


def roll_fixed(tc, gains_flat, action, setconst, const_gains=None):
    """Roll a constant-gain policy. setconst=True ignores action and uses const_gains
    (the hand-set baseline); otherwise feeds a fixed `action` theta each step (DE-opt)."""
    env = SpotDockEnv(randomize=False, test_case=tc, t_max=EPISODE_T,
                      setconst=setconst, const_gains=const_gains or CONST_GAINS, const_kdock=K_DOCK)
    env.reset(seed=0)
    ts, gseq, rews = [], [], []; docked = False
    for _ in range(int(EPISODE_T / DT)):
        ts.append(env.t); gseq.append(gains_flat)
        obs, r, term, trunc, info = env.step(action)
        rews.append(r); docked = docked or info["docked"]
        if term or trunc:
            break
    return np.asarray(ts), np.asarray(gseq), np.asarray(rews), docked


def roll_const(tc):
    g_flat = np.array(list(CONST_GAINS) * 3 + [K_DOCK])
    return roll_fixed(tc, g_flat, np.zeros(10), setconst=True)


def roll_deopt(tc):
    a = np.clip(np.asarray(TARGET_THETA[tc], float), -1, 1)
    g, k = SpotDockEnv()._decode(a)
    return roll_fixed(tc, np.append(g.flatten(), k), a.astype(np.float32), setconst=False)


def roll_all(model_dir, rnn, hidden, rl2):
    M.HIDDEN_OBS = hidden; M.HIDDEN_RL2 = rl2
    venv = VecNormalize.load(model_dir + "/vecnormalize.pkl", DummyVecEnv([M.make_env(0)]))
    venv.training = False
    model = (RecurrentPPO if rnn else PPO).load(model_dir + "/best_model.zip")
    return [roll_one(model, venv, tc, rnn) for tc in range(5)]


mlp   = roll_all(MLP_DIR,  rnn=False, hidden=False, rl2=False)
lstm  = roll_all(LSTM_DIR, rnn=True,  hidden=True,  rl2=True)
const = [roll_const(tc) for tc in range(5)]
deopt = [roll_deopt(tc) for tc in range(5)]
COLS = [(mlp, "MLP (full obs)"), (lstm, "LSTM (blind)"),
        (const, "Constant 1,0.5,0.25"), (deopt, "DE-optimized")]

# ---------------- Figure 1: gains, MLP | LSTM | Constant | DE-opt ----------------
fig, axes = plt.subplots(5, 4, figsize=(26, 16))
for tc in range(5):
    for c, (data, name) in enumerate(COLS):
        t, g, r, dk = data[tc]
        gv = g[:, :9].std(axis=0).mean()
        ax = axes[tc, c]
        for j in range(10):
            ax.plot(t, g[:, j], lw=1.0, label=LABELS[j])
        ax.set_title("case %d   %s   docked=%s   gain-var=%.2f" % (tc, name, dk, gv), fontsize=9)
        ax.set_ylabel("gain"); ax.grid(alpha=0.3); ax.set_ylim(-0.2, 6.2)
axes[0, 0].legend(fontsize=6, ncol=5, loc="upper right")
for c in range(4):
    axes[-1, c].set_xlabel("time [s]")
fig.suptitle("Adaptive ICCBF gains:  MLP  vs  LSTM (blind)  vs  Constant  vs  DE-optimized", fontsize=13)
fig.tight_layout(); fig.savefig("figures/adaptive_gains_MLP_vs_LSTM.png", dpi=120); plt.close(fig)

# ---------------- Figure 2: reward (per-step + cumulative), 4 lines ----------------
COLORS = {"MLP": "C0", "LSTM": "C3", "Const": "C2", "DE-opt": "C1"}
fig, axes = plt.subplots(5, 2, figsize=(14, 16))
for tc in range(5):
    series = [("MLP", mlp[tc]), ("LSTM", lstm[tc]), ("Const", const[tc]), ("DE-opt", deopt[tc])]
    for nm, (t, _, r, dk) in series:
        axes[tc, 0].plot(t, r, lw=0.8, color=COLORS[nm], label=nm)
        axes[tc, 1].plot(t, np.cumsum(r), lw=1.3, color=COLORS[nm],
                         label="%s (tot %.0f, dock=%s)" % (nm, r.sum(), dk))
    axes[tc, 0].set_title("case %d  per-step reward" % tc, fontsize=9)
    axes[tc, 0].set_ylabel("r(t)"); axes[tc, 0].grid(alpha=0.3); axes[tc, 0].set_ylim(-5, 12)
    axes[tc, 1].set_title("case %d  cumulative reward" % tc, fontsize=9)
    axes[tc, 1].set_ylabel("Σr"); axes[tc, 1].grid(alpha=0.3); axes[tc, 1].legend(fontsize=7)
axes[0, 0].legend(fontsize=8, loc="upper right")
for c in range(2):
    axes[-1, c].set_xlabel("time [s]")
fig.suptitle("Per-step (left) and cumulative (right) reward:  MLP vs LSTM vs Constant vs DE-optimized", fontsize=11)
fig.tight_layout(); fig.savefig("figures/reward_MLP_vs_LSTM.png", dpi=120); plt.close(fig)

print("wrote figures/adaptive_gains_MLP_vs_LSTM.png and figures/reward_MLP_vs_LSTM.png\n")
print(" case | dock M/L/C/D | gain-var M/L (C/D=0) | total reward  MLP / LSTM / Const / DE-opt")
for tc in range(5):
    gm = mlp[tc][1][:, :9].std(axis=0).mean(); gl = lstm[tc][1][:, :9].std(axis=0).mean()
    print("  %d   | %d/%d/%d/%d  |   %.2f / %.2f       |   %.0f / %.0f / %.0f / %.0f"
          % (tc, mlp[tc][3], lstm[tc][3], const[tc][3], deopt[tc][3], gm, gl,
             mlp[tc][2].sum(), lstm[tc][2].sum(), const[tc][2].sum(), deopt[tc][2].sum()))
