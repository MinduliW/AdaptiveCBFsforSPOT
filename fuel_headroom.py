"""How much FUEL headroom exists within the residual band? (per case, while still
docking). If best-in-band fuel << const fuel, reweighting the reward toward fuel
will let the residual MLP capture it; if not, the constant baseline is near-optimal."""
import numpy as np
from scipy.optimize import differential_evolution
from spot_env import SpotDockEnv, DT


def rollout(a, tc):
    env = SpotDockEnv(randomize=False, test_case=tc, t_max=150.0, residual_gains=True)
    env.reset(seed=0); fuel = 0.0; dk = False; td = 150.0
    for _ in range(int(150 / DT)):
        _, r, term, trunc, info = env.step(np.clip(a, -1, 1))
        fuel += np.linalg.norm(info["u"]) * DT
        if info["docked"] and not dk:
            dk = True; td = env.t
        if term or trunc:
            break
    return fuel, dk, td


def fit(a, tc):
    fuel, dk, td = rollout(a, tc)
    return fuel + (0.0 if dk else 1000.0)      # minimize fuel; big penalty if it stops docking


if __name__ == "__main__":
    print("case | const fuel | best-in-band fuel | saved | dock?")
    for tc in range(5):
        res = differential_evolution(fit, [(-1, 1)] * 10, seed=0, args=(tc,), maxiter=18,
                                     popsize=12, tol=1e-6, mutation=(0.5, 1.0),
                                     recombination=0.7, polish=False, updating="deferred", workers=-1)
        fc, _, _ = rollout(np.zeros(10), tc)
        fb, dk, td = rollout(res.x, tc)
        print("  %d  |   %5.2f    |      %5.2f       | %3.0f%%  | %s"
              % (tc, fc, fb, 100 * (fc - fb) / fc, "dock" if dk else "NODOCK"))
