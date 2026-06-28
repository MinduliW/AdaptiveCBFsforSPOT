"""Force / Torque / Total-propellant breakdown of the _prop RL bests vs nominal and DE."""
import numpy as np
import main_train_adaptive as M
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from spot_env import SpotDockEnv, DT, K_DOCK, TORQUE_ARM

MLP_DIR  = "TrainedModels/PPO_adaptive_l2_n128_lr0.0003D_std0.2_ne10_g0.9997_ent0.01_scratch_rand_anneal_resid_prop"
LSTM_DIR = "TrainedModels/RPPO_adaptive_l2_n128_lr0.0001D_std0.2_ne10_g0.9997_ent0.01_scratch_rand_anneal_hiddenrl2_resid_prop"
LOWFUEL = {0:[0.6507,-0.6032,-0.93,0.2772,-0.5809,0.4715,-0.8438,-0.2635,-0.5667,-0.4162],
 1:[-0.596,0.909,-0.1841,-0.4747,0.1404,-0.5593,-0.5015,-0.8988,-0.9562,-0.6098],
 2:[-0.2705,-0.2155,-0.7704,0.5834,-0.3387,0.1077,0.3944,-0.7703,-0.8836,-0.2245],
 3:[-0.1397,0.2749,-0.7728,0.4184,-0.6189,-0.9892,-0.8478,0.2024,-0.8739,0.3361],
 4:[-0.342,-0.0518,-0.9049,-0.8049,0.5859,-0.5889,-0.88,0.078,-0.8085,-0.941]}


def ft_from_log(u):
    return np.sum(np.hypot(u[:, 0], u[:, 1])) * DT, np.sum(np.abs(u[:, 2])) * DT


def roll_model(d, rnn, hidden, rl2):
    M.HIDDEN_OBS = hidden; M.HIDDEN_RL2 = rl2; M.RESIDUAL_GAINS = True
    venv = VecNormalize.load(d + "/vecnormalize.pkl", DummyVecEnv([M.make_env(0)])); venv.training = False
    model = (RecurrentPPO if rnn else PPO).load(d + "/best_model.zip")
    out = []
    for tc in range(5):
        log, _, m = M.rollout_policy(model, venv, tc, rnn)
        F, T = ft_from_log(log["u"]); out.append((F, T, m["docked"]))
    return out


def roll_action(theta, const):
    out = []
    for tc in range(5):
        if const:
            env = SpotDockEnv(randomize=False, setconst=True, const_gains=(1., .5, .25),
                              const_kdock=K_DOCK, test_case=tc, t_max=150.0); a = np.zeros(10)
        else:
            env = SpotDockEnv(randomize=False, test_case=tc, t_max=150.0)
            a = np.clip(theta[tc], -1, 1).astype(np.float32)
        env.reset(seed=0); F = T = 0.0; dk = False
        for _ in range(int(150 / DT)):
            _, _, term, trunc, info = env.step(a); u = info["u"]
            F += np.hypot(u[0], u[1]) * DT; T += abs(u[2]) * DT; dk = dk or info["docked"]
            if term or trunc:
                break
        out.append((F, T, dk))
    return out


pol = {"nominal": roll_action(None, True),
       "MLP_prop": roll_model(MLP_DIR, False, False, False),
       "LSTM_prop": roll_model(LSTM_DIR, True, True, True),
       "DE-optimum": roll_action(LOWFUEL, False)}

print("policy     | docked | Force(Ns) | Torque(Nms) | Total(Ns-eq) | vs nominal")
nomtot = None
for name, res in pol.items():
    F = sum(r[0] for r in res); T = sum(r[1] for r in res); nd = sum(int(r[2]) for r in res)
    tot = F + T / TORQUE_ARM
    if nomtot is None:
        nomtot = tot
    print("%-10s |  %d/5   |   %5.2f   |    %5.3f    |    %5.1f     | %+.0f%%"
          % (name, nd, F, T, tot, 100 * (tot - nomtot) / nomtot))


