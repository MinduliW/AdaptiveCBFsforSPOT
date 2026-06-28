"""
Optimize CONSTANT ICCBF class-K gains for a single deterministic test case.

The 10-D parameter vector theta in [-1,1]^10 is decoded (via SpotDockEnv._decode)
to per-constraint gains (a0,a1,a2)x{tar-KOZ,obs-KOZ,LOS} + CLF Lslack, then held
constant for the whole rollout. A balanced fitness rewards docking and trades off
fuel, time, and safety margins. Optimized here with a black-box optimizer
(differential evolution / CMA-ES class); see tune_gains_rl.py for the RL version.

    python tune_gains.py [test_case]      # default 4
"""
import sys
import numpy as np
from multiprocessing import Pool
from scipy.optimize import differential_evolution

from spot_env import SpotDockEnv, DT, Rmat, DOCK_OFF, K_DOCK, wrap, propellant, TOF_CAP

HORIZON = 150.0          # s; matches the RL eval horizon (propellant-optimal gains dock
                         # gently/slowly -- up to ~148 s -- so 130 s would falsely reject them)


def rollout_metrics(theta, test_case=4, ic=None):
    # ic=(xR0,xB0,xU0) overrides the test_case initial poses (adaptive bandit)
    env = SpotDockEnv(randomize=False, setconst=False, test_case=test_case, ic=ic)
    env.reset(seed=0)
    a = np.clip(np.asarray(theta, float), -1, 1).astype(np.float32)
    fuel = 0.0; mh = [1e9, 1e9, 1e9]; docked = False; t_dock = HORIZON
    for _ in range(int(HORIZON / DT)):
        _, _, term, _, info = env.step(a)
        fuel += propellant(info["u"]) * DT
        mh[0] = min(mh[0], info["h_tar"]); mh[1] = min(mh[1], info["h_obs"])
        mh[2] = min(mh[2], info["h_los"])
        if info["docked"] and not docked:
            docked = True; t_dock = env.t
        if term:
            break
    r_des = env.xB[:2] + Rmat(env.xB[2]) @ DOCK_OFF[:2]
    th_des = wrap(env.xB[2] + DOCK_OFF[2])
    return dict(docked=docked, t_dock=t_dock, t_end=env.t, fuel=fuel,
               mh_tar=mh[0], mh_obs=mh[1], mh_los=mh[2],
               dock_err=float(np.hypot(*(env.xR[:2] - r_des))),
               att_err=float(abs(wrap(env.xR[2] - th_des))))


# ---- score() weights (tune freely) ----
W_DOCK  = 1000.0    # bonus for a successful dock
W_TIME  = 0.0       # per-second dock-time penalty (0 = no rush; was 3.0)
W_FUEL  = 25.0      # fuel economy
W_CLOSE = 60.0      # dock position error (if it never docks)
W_ATT   = 20.0      # dock attitude error
W_KOZ   = 500.0     # KOZ breach (hard safety)
W_LOS   = 40.0      # FOV loss (soft)
W_TOF   = 5.0       # per-second penalty for docking PAST TOF_CAP (one-sided; 0 if docked in time)


def score(m):
    """Weighted fitness (higher is better); weights above are tunable."""
    f  = W_DOCK if m["docked"] else 0.0      # dock is primary
    f -= W_TIME * m["t_dock"]                 # dock-time penalty
    if m["docked"]:
        f -= W_TOF * max(0.0, m["t_dock"] - TOF_CAP)   # one-sided cap: penalize docking PAST TOF_CAP only
    f -= W_FUEL * m["fuel"]                   # fuel economy
    f -= W_CLOSE * m["dock_err"] + W_ATT * m["att_err"]   # closeness if not docked
    f -= W_KOZ * max(0.0, -m["mh_tar"])       # hard: target KOZ breach
    f -= W_KOZ * max(0.0, -m["mh_obs"])       # hard: obstacle KOZ breach
    f -= W_LOS * max(0.0, -m["mh_los"])       # soft: FOV loss
    return f


def fitness(theta, test_case=4):             # differential_evolution MINIMIZES
    return -score(rollout_metrics(theta, test_case))


def _eval_pop(P, tc, pool):
    """Fitness of a whole population (parallel rollouts)."""
    return np.array(pool.starmap(fitness, [(np.asarray(x), tc) for x in P]))


def optimize_pso(tc, n=30, iters=35, w=0.7, c1=1.5, c2=1.5, seed=0):
    """Particle swarm over the 10-D gain vector in [-1,1]^10 (minimize fitness)."""
    rng = np.random.default_rng(seed); D = 10
    X = rng.uniform(-1, 1, (n, D)); X[0] = gains_to_theta(1.0, 0.5, 0.25, K_DOCK)
    Vel = rng.uniform(-0.1, 0.1, (n, D))
    with Pool() as pool:
        f = _eval_pop(X, tc, pool)
        pbest, pbest_f = X.copy(), f.copy()
        g = X[f.argmin()].copy(); g_f = f.min()
        for _ in range(iters):
            r1, r2 = rng.random((n, D)), rng.random((n, D))
            Vel = np.clip(w*Vel + c1*r1*(pbest-X) + c2*r2*(g-X), -0.5, 0.5)
            X = np.clip(X + Vel, -1, 1)
            f = _eval_pop(X, tc, pool)
            imp = f < pbest_f; pbest[imp] = X[imp]; pbest_f[imp] = f[imp]
            if pbest_f.min() < g_f:
                g = pbest[pbest_f.argmin()].copy(); g_f = pbest_f.min()
    return g


def optimize_ga(tc, pop=30, gens=35, mut=0.2, elite=2, seed=0):
    """Real-coded GA (tournament select, BLX-0.5 crossover, gaussian mutation)."""
    rng = np.random.default_rng(seed); D = 10
    P = rng.uniform(-1, 1, (pop, D)); P[0] = gains_to_theta(1.0, 0.5, 0.25, K_DOCK)
    with Pool() as pool:
        f = _eval_pop(P, tc, pool)
        for _ in range(gens):
            order = f.argsort(); P, f = P[order], f[order]   # ascending (best first)
            nxt = [P[i].copy() for i in range(elite)]         # elitism
            while len(nxt) < pop:
                def pick():
                    a, b = rng.integers(0, pop, 2)
                    return P[a] if f[a] < f[b] else P[b]
                p1, p2 = pick(), pick()
                child = p1 + rng.uniform(-0.5, 1.5, D)*(p2 - p1)   # BLX-0.5
                m = rng.random(D) < mut
                child[m] += rng.normal(0, 0.2, m.sum())
                nxt.append(np.clip(child, -1, 1))
            P = np.array(nxt); f = _eval_pop(P, tc, pool)
    return P[f.argmin()]


def optimize_de(tc, base):
    """Differential evolution (scipy)."""
    res = differential_evolution(
        fitness, bounds=[(-1, 1)]*10, x0=base, seed=0, args=(tc,),
        maxiter=20, popsize=12, tol=1e-6, mutation=(0.5, 1.0),
        recombination=0.7, polish=False, updating="deferred", workers=-1)
    return res.x


def gains_to_theta(a0, a1, a2, k_dock):
    """Inverse of SpotDockEnv._decode for a uniform (a0,a1,a2) on all 3 rows."""
    e = SpotDockEnv()
    x0 = 2*(a0/e.ACOEF_HI) - 1
    x1 = 2*(a1/e.ACOEF_HI) - 1
    x2 = 2*((a2-e.HSLACK_LO)/(e.HSLACK_HI-e.HSLACK_LO)) - 1
    x9 = 2*((k_dock/K_DOCK - e.LMUL_LO)/(e.LMUL_HI-e.LMUL_LO)) - 1
    return np.array([x0, x1, x2]*3 + [x9])


def report(tag, theta, tc):
    g, k = SpotDockEnv()._decode(theta); m = rollout_metrics(theta, tc)
    print(f"\n[{tag}]  score={score(m):.1f}")
    print("  gains (a0,a1,a2)  tar-KOZ=%s  obs-KOZ=%s  LOS=%s  k_dock=%.2f"
          % (np.round(g[0], 3), np.round(g[1], 3), np.round(g[2], 3), k))
    print("  docked=%s t_dock=%.1fs fuel=%.3f  min h: tar=%+.3f obs=%+.3f los=%+.3f"
          % (m["docked"], m["t_dock"], m["fuel"], m["mh_tar"], m["mh_obs"], m["mh_los"]))
    return m


OPTIMIZERS = {"de": optimize_de, "pso": optimize_pso, "ga": optimize_ga}

if __name__ == "__main__":
    # usage:  python tune_gains.py [test_case] [de|pso|ga]
    tc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    method = sys.argv[2] if len(sys.argv) > 2 else "de"
    base = gains_to_theta(1.0, 0.5, 0.25, K_DOCK)        # the hand-set baseline
    report(f"baseline case {tc}", base, tc)

    best = optimize_de(tc, base) if method == "de" else OPTIMIZERS[method](tc)
    report(f"optimized [{method}] case {tc}", best, tc)
    print("\nraw theta =", np.round(best, 4).tolist())
