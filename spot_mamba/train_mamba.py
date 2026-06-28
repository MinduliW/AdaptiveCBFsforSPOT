"""
SPOT Mamba PPO trainer -- the Mamba analogue of main_train_adaptive.py's LSTM path.

Reuses SpotDockEnv + make_env (hidden_obs+rl2 = blind POMDP, parity with the LSTM
run), wraps SubprocVecEnv + VecNormalize(norm_obs), trains the pure-PyTorch Mamba2
PPO, and every `eval_every` updates rolls the 5 named cases deterministically,
scoring with tune_gains.score and saving the best agent + VecNormalize stats.

    python -m spot_mamba.train_mamba          # (set flags below)
Runs on CPU (pure-torch Mamba) or CUDA automatically.
"""
import os
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import main_train_adaptive as M
from main_train_adaptive import EPISODE_T
from spot_env import SpotDockEnv, DT, Rmat, DOCK_OFF, wrap, propellant, clf_rows, N_CASES
from tune_gains import score
from spot_mamba.ppo import MambaPPO
from spot_mamba.configs import SpotMambaConfig

HIDDEN_OBS, HIDDEN_RL2 = True, True          # blind chaser-only + RL^2 (parity with LSTM run)
RESIDUAL_GAINS = True                         # action -> const baseline +/- band (parity with LSTM run)


@torch.no_grad()
def eval_five_cases(agent, venv, device):
    """Deterministic Mamba rollout on the 5 named cases -> (docked, sum_score)."""
    agent.eval()
    total, ndock = 0.0, 0
    for tc in range(N_CASES):
        env = SpotDockEnv(randomize=False, test_case=tc, t_max=EPISODE_T,
                          hidden_obs=HIDDEN_OBS, rl2=HIDDEN_RL2, residual_gains=RESIDUAL_GAINS)
        obs, _ = env.reset(seed=0)
        st = agent.initial_states(1, device)
        mh = [1e9, 1e9, 1e9]; dk = False; td = EPISODE_T; fuel = 0.0
        for _ in range(int(EPISODE_T / DT)):
            nobs = venv.normalize_obs(obs.reshape(1, -1))
            a, _, _, st = agent.step(torch.as_tensor(nobs, dtype=torch.float32, device=device),
                                     st, deterministic=True)
            act = np.clip(a.cpu().numpy().reshape(-1), -1, 1).astype(np.float32)
            obs, r, term, trunc, info = env.step(act)
            mh = [min(mh[0], info["h_tar"]), min(mh[1], info["h_obs"]), min(mh[2], info["h_los"])]
            fuel += propellant(info["u"]) * DT
            if info["docked"] and not dk:
                dk = True; td = env.t
            if term or trunc:
                break
        r_des = env.xB[:2] + Rmat(env.xB[2]) @ DOCK_OFF[:2]; th_des = wrap(env.xB[2] + DOCK_OFF[2])
        m = dict(docked=dk, t_dock=td, t_end=env.t, fuel=fuel, mh_tar=mh[0], mh_obs=mh[1], mh_los=mh[2],
                 dock_err=float(np.hypot(*(env.xR[:2] - r_des))), att_err=float(abs(wrap(env.xR[2] - th_des))))
        total += score(m); ndock += int(dk)
    agent.train()
    return ndock, total


def main():
    trainON, eval_every = True, 8
    cfg = SpotMambaConfig()
    name = "MAMBA_d%d_s%d_l%dn%d_lr%g_g%g_ns%d_bi%d_prop_tof" % (   # _prop_tof = torque fuel + TOF cap
        cfg.d_model, cfg.d_state, cfg.layers, cfg.nodes, cfg.learning_rate,
        cfg.gamma, cfg.n_steps, cfg.burn_in)
    save_dir = os.path.join("TrainedModels", name)
    os.makedirs(save_dir, exist_ok=True)
    print("Training:", name)

    M.HIDDEN_OBS, M.HIDDEN_RL2 = HIDDEN_OBS, HIDDEN_RL2
    M.RESIDUAL_GAINS = RESIDUAL_GAINS          # make_env builds residual envs (parity with LSTM)
    VecCls = SubprocVecEnv if cfg.num_env > 1 else DummyVecEnv
    venv = VecNormalize(VecCls([M.make_env(i, random_states=True) for i in range(cfg.num_env)]),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)
    ppo = MambaPPO(venv, cfg)
    best = [-1e18]

    def cb(p, it, stats):
        if (it + 1) % eval_every:
            return
        nd, tot = eval_five_cases(p.agent, venv, p.device)
        tag = ""
        if tot > best[0]:
            best[0] = tot
            torch.save(p.agent.state_dict(), os.path.join(save_dir, "best_agent.pt"))
            venv.save(os.path.join(save_dir, "vecnormalize.pkl"))
            with open(os.path.join(save_dir, "best_score.txt"), "w") as f:
                f.write("%.4f" % tot)
            tag = "NEW BEST"
        print("[eval @ %d]  docked %d/%d  sum_score=%.0f  pg=%+.3f kl=%.4f  %s"
              % (p.num_timesteps, nd, N_CASES, tot, stats["pg_loss"], stats["approx_kl"], tag))

    if trainON:
        ppo.learn(cfg.total_timesteps, callback=cb)
        torch.save(ppo.agent.state_dict(), os.path.join(save_dir, "final_agent.pt"))
        venv.save(os.path.join(save_dir, "vecnormalize.pkl"))
        print("done.")


if __name__ == "__main__":
    main()
