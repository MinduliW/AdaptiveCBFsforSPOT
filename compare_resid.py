"""Compare the residual MLP (full obs) vs residual LSTM (blind) vs const baseline
on the 5 cases -- focus on FUEL (what the reweighted reward is now optimizing)."""
import numpy as np
import main_train_adaptive as M
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from spot_env import SpotDockEnv, DT, K_DOCK, Rmat, DOCK_OFF, wrap
from tune_gains import score

MLP_DIR  = "TrainedModels/PPO_adaptive_l2_n128_lr0.0003D_std0.4_ne10_g0.9997_ent0.01_scratch_rand_anneal_resid"
LSTM_DIR = "TrainedModels/RPPO_adaptive_l2_n128_lr0.0003D_std0.4_ne10_g0.9997_ent0.01_scratch_rand_anneal_hiddenrl2_resid"
M.RESIDUAL_GAINS = True


def eval_model(d, rnn, hidden, rl2):
    M.HIDDEN_OBS = hidden; M.HIDDEN_RL2 = rl2
    venv = VecNormalize.load(d + "/vecnormalize.pkl", DummyVecEnv([M.make_env(0)])); venv.training = False
    model = (RecurrentPPO if rnn else PPO).load(d + "/best_model.zip")
    return [M.rollout_policy(model, venv, tc, rnn)[2] for tc in range(5)]


def roll_const(tc):
    env = SpotDockEnv(randomize=False, test_case=tc, t_max=150.0, residual_gains=True)
    env.reset(seed=0); fuel = 0.0; dk = False; td = 150.0; mh = [1e9, 1e9, 1e9]
    for _ in range(int(150 / DT)):
        _, r, term, trunc, info = env.step(np.zeros(10)); fuel += np.linalg.norm(info["u"]) * DT
        mh = [min(mh[0], info["h_tar"]), min(mh[1], info["h_obs"]), min(mh[2], info["h_los"])]
        if info["docked"] and not dk: dk = True; td = env.t
        if term or trunc: break
    r_des = env.xB[:2] + Rmat(env.xB[2]) @ DOCK_OFF[:2]; th_des = wrap(env.xB[2] + DOCK_OFF[2])
    return dict(docked=dk, t_dock=td, t_end=env.t, fuel=fuel, mh_tar=mh[0], mh_obs=mh[1], mh_los=mh[2],
                dock_err=float(np.hypot(*(env.xR[:2] - r_des))), att_err=float(abs(wrap(env.xR[2] - th_des))))


mlp  = eval_model(MLP_DIR, False, False, False)
lstm = eval_model(LSTM_DIR, True, True, True)
cst  = [roll_const(tc) for tc in range(5)]

print("case |    fuel (MLP / LSTM / const)    |  dock M/L/C | score (MLP/LSTM/const)")
tot = {"mlp": [0, 0], "lstm": [0, 0], "c": [0, 0]}
for tc in range(5):
    fm, fl, fc = mlp[tc]["fuel"], lstm[tc]["fuel"], cst[tc]["fuel"]
    sm, sl, sc = score(mlp[tc]), score(lstm[tc]), score(cst[tc])
    tot["mlp"][0] += fm; tot["lstm"][0] += fl; tot["c"][0] += fc
    tot["mlp"][1] += sm; tot["lstm"][1] += sl; tot["c"][1] += sc
    print("  %d  |  %4.2f / %4.2f / %4.2f  (%+.0f%%)  |   %d/%d/%d   |  %.0f / %.0f / %.0f"
          % (tc, fm, fl, fc, 100 * (fc - fm) / fc, int(mlp[tc]["docked"]),
             int(lstm[tc]["docked"]), int(cst[tc]["docked"]), sm, sl, sc))
print("-" * 78)
print(" TOT |  %4.2f / %4.2f / %4.2f          |           |  %.0f / %.0f / %.0f"
      % (tot["mlp"][0], tot["lstm"][0], tot["c"][0], tot["mlp"][1], tot["lstm"][1], tot["c"][1]))
print("\nMLP fuel vs const: %+.0f%% | LSTM fuel vs const: %+.0f%%"
      % (100 * (tot["c"][0] - tot["mlp"][0]) / tot["c"][0],
         100 * (tot["c"][0] - tot["lstm"][0]) / tot["c"][0]))
