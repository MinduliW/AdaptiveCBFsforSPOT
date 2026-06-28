"""
Re-optimize the per-case constant gains (TARGET_THETA) with differential evolution
to maximize the CURRENT env reward (refined reward + 8 cm capture tol), instead of
the old score() objective at 5 cm. Prints a drop-in TARGET_THETA dict.

    python retune_de.py
"""
import numpy as np
from scipy.optimize import differential_evolution

from spot_env import SpotDockEnv, DT, K_DOCK
from tune_gains import score, rollout_metrics, gains_to_theta

EPISODE_T = 150.0
HORIZON = int(EPISODE_T / DT)


def env_return(theta, tc):
    """Total env reward of holding constant gains `theta` on case `tc`."""
    env = SpotDockEnv(randomize=False, test_case=tc, t_max=EPISODE_T)
    env.reset(seed=0)
    a = np.clip(np.asarray(theta, float), -1, 1).astype(np.float32)
    R = 0.0
    for _ in range(HORIZON):
        _, r, term, trunc, _ = env.step(a)
        R += r
        if term or trunc:
            break
    return R


def fitness(theta, tc):
    return -env_return(theta, tc)


if __name__ == "__main__":
    base = gains_to_theta(1.0, 0.5, 0.25, K_DOCK)        # hand-set baseline as DE seed
    new = {}
    print("case |  env_reward (base -> DE) | score() | docked | t_dock")
    for tc in range(5):
        r_base = env_return(base, tc)
        res = differential_evolution(
            fitness, bounds=[(-1, 1)] * 10, x0=base, seed=0, args=(tc,),
            maxiter=20, popsize=12, tol=1e-6, mutation=(0.5, 1.0),
            recombination=0.7, polish=False, updating="deferred", workers=-1)
        new[tc] = [round(float(x), 4) for x in res.x]
        m = rollout_metrics(res.x, tc)
        print("  %d  |   %6.1f -> %6.1f      | %6.1f  |  %s  | %.1fs"
              % (tc, r_base, env_return(res.x, tc), score(m), m["docked"], m["t_dock"]))

    print("\nTARGET_THETA = {")
    for tc in range(5):
        print("    %d: %s," % (tc, new[tc]))
    print("}")
    with open("/tmp/new_target_theta.txt", "w") as f:
        f.write("TARGET_THETA = {\n")
        for tc in range(5):
            f.write("    %d: %s,\n" % (tc, new[tc]))
        f.write("}\n")
    print("\nwrote /tmp/new_target_theta.txt")
