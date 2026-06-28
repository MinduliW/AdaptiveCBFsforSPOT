"""Plot residual MLP (full obs) vs residual LSTM (blind) vs const baseline:
gain trajectories and cumulative fuel, on the 5 cases."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import main_train_adaptive as M
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from spot_env import SpotDockEnv, DT, K_DOCK

MLP_DIR  = "TrainedModels/PPO_adaptive_l2_n128_lr0.0003D_std0.4_ne10_g0.9997_ent0.01_scratch_rand_anneal_resid"
LSTM_DIR = "TrainedModels/RPPO_adaptive_l2_n128_lr0.0003D_std0.4_ne10_g0.9997_ent0.01_scratch_rand_anneal_hiddenrl2_resid"
LABELS = ["tar_a0", "tar_a1", "tar_a2", "obs_a0", "obs_a1", "obs_a2", "los_a0", "los_a1", "los_a2", "k_dock"]
M.RESIDUAL_GAINS = True


def roll_model(d, rnn, hidden, rl2):
    M.HIDDEN_OBS = hidden; M.HIDDEN_RL2 = rl2
    venv = VecNormalize.load(d + "/vecnormalize.pkl", DummyVecEnv([M.make_env(0)])); venv.training = False
    model = (RecurrentPPO if rnn else PPO).load(d + "/best_model.zip")
    out = []
    for tc in range(5):
        log, gains_t, m = M.rollout_policy(model, venv, tc, rnn)
        fuel = np.cumsum(np.linalg.norm(log["u"], axis=1)) * DT
        out.append((log["t"], gains_t, fuel))
    return out


def roll_const():
    out = []
    g_flat = np.array([1., 0.5, 0.25] * 3 + [K_DOCK])
    for tc in range(5):
        env = SpotDockEnv(randomize=False, test_case=tc, t_max=150.0, residual_gains=True)
        env.reset(seed=0); ts, us, gains = [], [], []
        for _ in range(int(150 / DT)):
            ts.append(env.t); gains.append(g_flat)
            _, _, term, trunc, info = env.step(np.zeros(10)); us.append(info["u"])
            if term or trunc:
                break
        out.append((np.array(ts), np.array(gains), np.cumsum(np.linalg.norm(np.array(us), axis=1)) * DT))
    return out


mlp, lstm, cst = roll_model(MLP_DIR, False, False, False), roll_model(LSTM_DIR, True, True, True), roll_const()
COLS = [(mlp, "MLP-resid (full obs)"), (lstm, "LSTM-resid (blind)"), (cst, "const baseline")]

# ---- gains ----
fig, ax = plt.subplots(5, 3, figsize=(20, 16))
for tc in range(5):
    for c, (data, nm) in enumerate(COLS):
        t, g, fuel = data[tc]
        for j in range(10):
            ax[tc, c].plot(t, g[:, j], lw=1.0, label=LABELS[j])
        ax[tc, c].set_title("case %d   %s   gain-var=%.2f" % (tc, nm, g[:, :9].std(0).mean()), fontsize=9)
        ax[tc, c].grid(alpha=0.3); ax[tc, c].set_ylim(-0.5, 6.5); ax[tc, c].set_ylabel("gain")
ax[0, 0].legend(fontsize=6, ncol=5, loc="upper right")
fig.suptitle("Residual gains: MLP vs LSTM vs const", fontsize=13)
fig.tight_layout(); fig.savefig("figures/resid_gains.png", dpi=120); plt.close(fig)

# ---- cumulative fuel ----
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for tc in range(5):
    a = ax[tc // 3, tc % 3]
    for (data, nm), col in zip(COLS, ["C0", "C3", "C2"]):
        t, g, fuel = data[tc]
        a.plot(t, fuel, color=col, lw=1.5, label="%s (%.2f)" % (nm.split()[0], fuel[-1]))
    a.set_title("case %d  cumulative fuel" % tc, fontsize=10); a.set_xlabel("time [s]")
    a.set_ylabel("fuel"); a.grid(alpha=0.3); a.legend(fontsize=8)
ax[1, 2].axis("off")
fig.suptitle("Cumulative fuel: MLP vs LSTM vs const (lower = better)", fontsize=12)
fig.tight_layout(); fig.savefig("figures/resid_fuel.png", dpi=120); plt.close(fig)
print("wrote figures/resid_gains.png and figures/resid_fuel.png")
