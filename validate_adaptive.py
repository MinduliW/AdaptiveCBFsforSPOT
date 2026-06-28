"""
Validate a trained per-step ADAPTIVE policy.

Loads best_model.zip + vecnormalize.pkl from a run dir, rolls the policy on the 5
named cases, and produces:
  - a per-case metrics table (dock time, fuel, min barrier margins, gain-variation),
  - one video per case (videos/adaptive_case<N>.mp4),
  - a figure of the time-varying gains (figures/adaptive_gains.png) -- this shows
    HOW the policy adapts the gains through each maneuver.

    python validate_adaptive.py [run_dir] [--rnn] [--no-video]
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import main_train_adaptive as M
import run_matlab_case as RC
from tune_gains import score

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

DEFAULT_RUN = "TrainedModels/PPO_adaptive_l2_n128_lr0.0001D_std0.1_ne10_g0.9997_ent0.0001"
LABELS = ["tar_a0", "tar_a1", "tar_a2", "obs_a0", "obs_a1", "obs_a2",
          "los_a0", "los_a1", "los_a2", "k_dock"]


def validate(run_dir=DEFAULT_RUN, rnn=False, render=True):
    venv = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"),
                             DummyVecEnv([M.make_env(0)]))
    venv.training = False; venv.norm_reward = False
    Algo = RecurrentPPO if rnn else PPO
    model = Algo.load(os.path.join(run_dir, "best_model.zip"))

    print("validating:", run_dir)
    print("case | docked | t_dock | fuel | min h  tar / obs / los | gain-var tar/obs/los")
    fig, axes = plt.subplots(5, 1, figsize=(10, 13))
    tot = 0.0; nd = 0
    for tc in range(5):
        log, gains_t, m = M.rollout_policy(model, venv, tc, rnn)
        s = score(m); tot += s; nd += int(m["docked"])
        var = gains_t.std(axis=0)[:9].reshape(3, 3).mean(axis=1)   # mean per-CBF gain motion
        print("  %d  |  %-4s | %5.1fs | %.2f | %+.2f / %+.2f / %+.2f | %.2f / %.2f / %.2f"
              % (tc, m["docked"], m["t_dock"], m["fuel"],
                 m["mh_tar"], m["mh_obs"], m["mh_los"], *var))
        t = log["t"]
        for j in range(10):
            axes[tc].plot(t, gains_t[:, j], lw=1.1, label=LABELS[j])
        axes[tc].set_title("case %d   docked=%s  t_dock=%.0fs  fuel=%.2f"
                           % (tc, m["docked"], m["t_dock"], m["fuel"]), fontsize=10)
        axes[tc].set_ylabel("gain"); axes[tc].grid(alpha=0.3)
        if render:
            RC.animate(log, tc, out="adaptive_case%d.mp4" % tc)

    axes[0].legend(fontsize=7, ncol=5, loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Per-step adaptive gains over each maneuver", fontsize=12)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/adaptive_gains.png", dpi=120); plt.close(fig)

    print("\nSUM_SCORE = %.1f   docked %d/5" % (tot, nd))
    print("wrote figures/adaptive_gains.png" + ("  +  videos/adaptive_case0..4.mp4" if render else ""))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = args[0] if args else DEFAULT_RUN
    validate(run, rnn="--rnn" in sys.argv, render="--no-video" not in sys.argv)
