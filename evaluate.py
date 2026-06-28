"""
Unified SPOT-docking evaluation.

Rolls a set of policies deterministically through the 5 named cases and produces,
in one place:
  1. the FiveCaseEval summary  -- docked/5, sum_score, per-case score (the training metric)
  2. the propellant table      -- Force [Ns], Torque [Nms], Total [Ns-eq], vs nominal
  3. videos                    -- one per case (--videos), optionally side-by-side (--compare)

A "policy" is one of:
  - const : the nominal constant gains (1, 0.5, 0.25)
  - theta : a fixed per-case action vector (e.g. the DE propellant-optimum), absolute decode
  - ppo / rppo : a trained SB3 PPO / RecurrentPPO model dir
  - mamba : a trained spot_mamba agent dir
The obs mode (hidden/rl2) and gain decode (residual) are set per policy below.

    python evaluate.py                      # tables for the default set
    python evaluate.py --videos             # + render a video per policy per case
    python evaluate.py --compare MLP LSTM   # + stitch those two side-by-side per case
"""
import os
import argparse
import numpy as np

import main_train_adaptive as M
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from spot_env import SpotDockEnv, DT, K_DOCK, TORQUE_ARM, Rmat, DOCK_OFF, wrap, N_CASES
from tune_gains import score
import run_matlab_case as RC

EPISODE_T = 150.0
VID_DIR = "videos/eval"
LOG_KEYS = ("t", "xR", "xB", "xU", "rk_tar", "rk_obs", "h_tar", "h_obs", "h_los", "V", "u", "fov", "soff")

# ------------------------------- policy registry -------------------------------
MLP_DIR  = "TrainedModels/PPO_adaptive_l2_n128_lr0.0003D_std0.2_ne10_g0.9997_ent0.01_scratch_rand_anneal_resid_prop"
LSTM_DIR = "TrainedModels/RPPO_adaptive_l2_n128_lr0.0001D_std0.2_ne10_g0.9997_ent0.01_scratch_rand_anneal_hiddenrl2_resid_prop"
LOWFUEL = {0: [0.6507, -0.6032, -0.93, 0.2772, -0.5809, 0.4715, -0.8438, -0.2635, -0.5667, -0.4162],
           1: [-0.596, 0.909, -0.1841, -0.4747, 0.1404, -0.5593, -0.5015, -0.8988, -0.9562, -0.6098],
           2: [-0.2705, -0.2155, -0.7704, 0.5834, -0.3387, 0.1077, 0.3944, -0.7703, -0.8836, -0.2245],
           3: [-0.1397, 0.2749, -0.7728, 0.4184, -0.6189, -0.9892, -0.8478, 0.2024, -0.8739, 0.3361],
           4: [-0.342, -0.0518, -0.9049, -0.8049, 0.5859, -0.5889, -0.88, 0.078, -0.8085, -0.941]}

POLICIES = [
    dict(name="nominal", kind="const"),
    dict(name="MLP",  kind="ppo",  dir=MLP_DIR,  hidden=False, rl2=False, residual=True),
    dict(name="LSTM", kind="rppo", dir=LSTM_DIR, hidden=True,  rl2=True,  residual=True),
    dict(name="DE",   kind="theta", theta=LOWFUEL, residual=False),
]


# ------------------------------- rollout ---------------------------------------
def _load(p):
    """Load (venv, model, agent) for a policy; any may be None."""
    # set obs mode FIRST so make_env builds the matching obs dim for the saved normalizer
    M.HIDDEN_OBS = p.get("hidden", False); M.HIDDEN_RL2 = p.get("rl2", False)
    M.RESIDUAL_GAINS = p.get("residual", False)
    if p["kind"] in ("ppo", "rppo"):
        venv = VecNormalize.load(p["dir"] + "/vecnormalize.pkl", DummyVecEnv([M.make_env(0)]))
        venv.training = False
        model = (RecurrentPPO if p["kind"] == "rppo" else PPO).load(p["dir"] + "/best_model.zip")
        return venv, model, None
    if p["kind"] == "mamba":
        import torch
        from spot_mamba.agent import Mamba2ActorCritic
        from spot_mamba.configs import SpotMambaConfig
        venv = VecNormalize.load(p["dir"] + "/vecnormalize.pkl", DummyVecEnv([M.make_env(0)]))
        venv.training = False
        cfg = SpotMambaConfig()
        ag = Mamba2ActorCritic(venv.observation_space.shape[0], venv.action_space.shape[0],
                               d_model=cfg.d_model, layers=cfg.layers, nodes=cfg.nodes,
                               log_std_init=cfg.log_std_init, mamba_kw=cfg.mamba_kw())
        ag.load_state_dict(torch.load(p["dir"] + "/best_agent.pt", map_location="cpu")); ag.eval()
        return venv, None, ag
    return None, None, None


def roll(p, tc, venv, model, agent):
    """Deterministic rollout of policy p on case tc. Returns (log, metrics, F_imp, T_imp)."""
    M.HIDDEN_OBS = p.get("hidden", False); M.HIDDEN_RL2 = p.get("rl2", False)
    M.RESIDUAL_GAINS = p.get("residual", False)
    env = SpotDockEnv(randomize=False, test_case=tc, t_max=EPISODE_T,
                      hidden_obs=M.HIDDEN_OBS, rl2=M.HIDDEN_RL2, residual_gains=M.RESIDUAL_GAINS,
                      setconst=(p["kind"] == "const"), const_gains=(1.0, 0.5, 0.25), const_kdock=K_DOCK)
    obs, _ = env.reset(seed=0)
    state = None; es = np.ones(1, dtype=bool)
    if agent is not None:
        state = agent.initial_states(1, "cpu")
    log = {k: [] for k in LOG_KEYS}
    mh = [1e9, 1e9, 1e9]; F = T = 0.0; dk = False; td = EPISODE_T
    for _ in range(int(EPISODE_T / DT)):
        if model is not None:
            nobs = venv.normalize_obs(obs)
            if p["kind"] == "rppo":
                a, state = model.predict(nobs, state=state, episode_start=es, deterministic=True); es = np.zeros(1, bool)
            else:
                a, _ = model.predict(nobs, deterministic=True)
            a = np.asarray(a).reshape(-1)
        elif agent is not None:
            import torch
            nobs = venv.normalize_obs(obs.reshape(1, -1))
            av, _, _, state = agent.step(torch.as_tensor(nobs, dtype=torch.float32), state, deterministic=True)
            a = np.clip(av.cpu().numpy().reshape(-1), -1, 1).astype(np.float32)
        elif p["kind"] == "theta":
            a = np.clip(p["theta"][tc], -1, 1).astype(np.float32)
        else:                                                        # const
            a = np.zeros(10, dtype=np.float32)
        log["t"].append(env.t); log["xR"].append(env.xR.copy()); log["xB"].append(env.xB.copy())
        log["xU"].append(env.xU.copy()); log["rk_tar"].append(env.rkoz_tar.copy())
        log["rk_obs"].append(env.r_koz_obs.copy()); log["fov"].append(env.fov); log["soff"].append(env.sens_off.copy())
        obs, _, term, trunc, info = env.step(a)
        u = info["u"]; F += np.hypot(u[0], u[1]) * DT; T += abs(u[2]) * DT
        for kk in ("h_tar", "h_obs", "h_los", "V", "u"):
            log[kk].append(info[kk])
        mh = [min(mh[0], info["h_tar"]), min(mh[1], info["h_obs"]), min(mh[2], info["h_los"])]
        if info["docked"] and not dk:
            dk = True; td = env.t
        if term or trunc:
            break
    for kk in log:
        log[kk] = np.asarray(log[kk])
    r_des = env.xB[:2] + Rmat(env.xB[2]) @ DOCK_OFF[:2]; th_des = wrap(env.xB[2] + DOCK_OFF[2])
    m = dict(docked=dk, t_dock=td, t_end=env.t, fuel=F + T / TORQUE_ARM, mh_tar=mh[0], mh_obs=mh[1],
             mh_los=mh[2], dock_err=float(np.hypot(*(env.xR[:2] - r_des))), att_err=float(abs(wrap(env.xR[2] - th_des))))
    return log, m, F, T


# ------------------------------- driver ----------------------------------------
def evaluate(policies, render=False):
    results = {}
    for p in policies:
        try:
            venv, model, agent = _load(p)
        except Exception as e:
            print("  [skip %s: %s]" % (p["name"], e)); continue
        rows = []
        for tc in range(N_CASES):
            log, m, F, T = roll(p, tc, venv, model, agent)
            rows.append((m, F, T))
            if render:
                os.makedirs(VID_DIR, exist_ok=True)
                RC.animate(log, tc, out=os.path.join(VID_DIR, "%s_case%d.mp4" % (p["name"], tc)))
        results[p["name"]] = rows
    return results


def print_tables(results):
    print("\n=== FiveCaseEval  (docked/%d, sum_score) ===" % N_CASES)
    print("policy   | docked | sum_score | per-case score")
    for name, rows in results.items():
        nd = sum(int(r[0]["docked"]) for r in rows); ss = sum(score(r[0]) for r in rows)
        print("%-8s |  %d/%d   |  %6.0f   | %s" % (name, nd, N_CASES, ss, " ".join("%4.0f" % score(r[0]) for r in rows)))

    print("\n=== Propellant  (Force [Ns] / Torque [Nms] / Total [Ns-eq]) ===")
    print("policy   | docked | Force | Torque | Total | vs nominal")
    nom = None
    for name, rows in results.items():
        F = sum(r[1] for r in rows); T = sum(r[2] for r in rows); tot = F + T / TORQUE_ARM
        nd = sum(int(r[0]["docked"]) for r in rows)
        if nom is None:
            nom = tot
        print("%-8s |  %d/%d   | %5.2f | %5.3f | %5.1f | %+.0f%%"
              % (name, nd, N_CASES, F, T, tot, 100 * (tot - nom) / nom))


def stitch(a, b):
    """ffmpeg side-by-side (a|b) per case from videos/eval/."""
    import subprocess
    out = os.path.join(VID_DIR, "compare")
    os.makedirs(out, exist_ok=True)
    for tc in range(N_CASES):
        fa = os.path.join(VID_DIR, "%s_case%d.mp4" % (a, tc)); fb = os.path.join(VID_DIR, "%s_case%d.mp4" % (b, tc))
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", fa, "-i", fb,
                        "-filter_complex", "hstack=inputs=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        os.path.join(out, "%s_vs_%s_case%d.mp4" % (a, b, tc))])
    print("stitched %s vs %s -> %s/  (left=%s, right=%s)" % (a, b, out, a, b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", action="store_true", help="render a video per policy per case")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="stitch two policies side-by-side")
    args = ap.parse_args()
    res = evaluate(POLICIES, render=args.videos or bool(args.compare))
    print_tables(res)
    if args.compare:
        stitch(*args.compare)
    if args.videos or args.compare:
        print("\nvideos -> %s/" % VID_DIR)
