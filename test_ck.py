"""
Test a set of constant ICCBF class-K gains on a deterministic case and render it,
to compare against the MATLAB sim. Just edit GAINS below and run.

    python test_ck.py            # GAINS below, case 4  -> spot_case4_test.mp4
    python test_ck.py 2          # case 2

Each gain triple is (a0, a1, a2) for that CBF; k_dock is the CLF decay rate.
"""
import sys
import numpy as np
from spot_env import SpotDockEnv, DT
import run_matlab_case as RC

# ----- edit these to match the MATLAB run --------------------------------
GAINS = dict(
    tar    = (1.0, 0.5, 0.25),   # KOZ_tar  (a0, a1, a2)
    obs    = (1.0, 0.5, 0.25),   # KOZ_obs
    los    = (1.0, 0.5, 0.25),   # LOS
    k_dock = 20.0,                # CLF decay
)
HORIZON = 140.0                  # s; runs the full horizon (no early stop)
# -------------------------------------------------------------------------


def run(tar, obs, los, k_dock, test_case=4, horizon=HORIZON, video=True):
    g = np.array([tar, obs, los], dtype=float)
    env = SpotDockEnv(randomize=False, setconst=True, const_gains=g,
                      const_kdock=k_dock, test_case=test_case)
    env.reset(seed=0)
    keys = ("t", "xR", "xB", "xU", "rk_tar", "rk_obs",
            "h_tar", "h_obs", "h_los", "V", "u", "fov", "soff")
    log = {k: [] for k in keys}; docked = False; t_dock = horizon
    
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)   # ignored (setconst)
    
    for _ in range(int(horizon / DT)):
        log["t"].append(env.t)
        log["xR"].append(env.xR.copy()); log["xB"].append(env.xB.copy())
        log["xU"].append(env.xU.copy())
        log["rk_tar"].append(env.rkoz_tar.copy()); log["rk_obs"].append(env.r_koz_obs.copy())
        log["fov"].append(env.fov); log["soff"].append(env.sens_off.copy())
        _, _, _, _, info = env.step(a)
        for kk in ("h_tar", "h_obs", "h_los", "V", "u"):
            log[kk].append(info[kk])
        if info["docked"] and not docked:
            docked = True; t_dock = env.t
    for k in log:
        log[k] = np.asarray(log[k])

    sep = np.linalg.norm(log["xB"][:, :2] - log["xR"][:, :2], axis=1)
    spd = np.linalg.norm(log["xR"][:, 3:5], axis=1)
    fuel = np.sum(np.linalg.norm(log["u"], axis=1)) * DT
    print("case %d  gains  tar=%s obs=%s los=%s  k_dock=%.2f"
          % (test_case, tar, obs, los, k_dock))
    print("  docked=%s  t_dock=%.1fs  fuel=%.3f  min sep=%.3f  peak speed=%.3f"
          % (docked, t_dock, fuel, sep.min(), spd.max()))
    print("  min margins  h_tar=%+.3f  h_obs=%+.3f  h_los=%+.3f"
          % (log["h_tar"].min(), log["h_obs"].min(), log["h_los"].min()))
    if video:
        out = RC.animate(log, test_case, out="spot_case%d_test.mp4" % test_case)
        print("  wrote", out)
    return log


if __name__ == "__main__":
    tc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run(GAINS["tar"], GAINS["obs"], GAINS["los"], GAINS["k_dock"], test_case=tc)
