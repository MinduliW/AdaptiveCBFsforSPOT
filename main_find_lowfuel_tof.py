"""Minimize total propellant (Ftotal = |F| + |Torque|/arm, integrated) per case,
SUBJECT TO: (1) docking, (2) docking under TOF_MAX, and (3) the whole chaser
PLATFORM (0.30 m square, not just the COG) staying on the table [0,XL]x[0,YL].
The COG can sit at y=2.30 and still poke the body off the y=2.4 edge -- so we
track the 4 rotated corners of the platform each step."""
import numpy as np
from scipy.optimize import differential_evolution
from spot_env import SpotDockEnv, DT, K_DOCK, TORQUE_ARM, XL, YL
from tune_gains import gains_to_theta

R = TORQUE_ARM
HORIZON = int(150 / DT)
TOF_MAX = 110.0
HALF = 0.15                                              # chaser body half-size (0.30 m square, matches renderer)
CORNERS = np.array([[HALF, HALF], [HALF, -HALF], [-HALF, HALF], [-HALF, -HALF]])


def ft_tof(theta, tc):
    env = SpotDockEnv(randomize=False, test_case=tc, t_max=150.0); env.reset(seed=0)
    a = np.clip(np.asarray(theta, float), -1, 1).astype(np.float32)
    F = T = 0.0; dk = False; td = 150.0; off = 0.0
    for _ in range(HORIZON):
        _, _, term, trunc, info = env.step(a); u = info["u"]
        F += np.hypot(u[0], u[1]) * DT; T += abs(u[2]) * DT
        x, y, th = env.xR[0], env.xR[1], env.xR[2]       # PLATFORM corners (rotated body square)
        c, s = np.cos(th), np.sin(th)
        cx = x + CORNERS[:, 0] * c - CORNERS[:, 1] * s
        cy = y + CORNERS[:, 0] * s + CORNERS[:, 1] * c
        off = max(off, (-cx).max(), (cx - XL).max(), (-cy).max(), (cy - YL).max())
        if info["docked"] and not dk:
            dk = True; td = env.t
        if term or trunc:
            break
    return F, T, dk, td, off


def fit(theta, tc):
    F, T, dk, td, off = ft_tof(theta, tc)
    obj = F + T / R
    if not dk:
        return 1e4 + obj                                # must dock
    pen = 0.0
    if off > 1e-3:
        pen += 500.0 + 2000.0 * off                     # HARD: whole platform must stay on the table
    if td > TOF_MAX:
        pen += 100.0 + 10.0 * (td - TOF_MAX)            # dock under TOF_MAX (no lazy near-horizon glide)
    return obj + pen


def make_cb(tc):
    """Per-iteration progress printer for differential_evolution (runs in main proc)."""
    it = [0]

    def cb(xk, convergence=None):
        it[0] += 1
        x = xk.x if hasattr(xk, "x") else np.asarray(xk)
        F, T, dk, td, off = ft_tof(x, tc)
        flag = "ok" if (dk and td <= TOF_MAX and off <= 1e-3) else \
               ("OFF-TABLE" if off > 1e-3 else ("slow" if dk else "NODOCK"))
        print("      iter %2d | best total=%6.2f  (F=%5.2f T=%5.3f)  TOF=%3.0fs  off=%.3f  [%s]"
              % (it[0], F + T / R, F, T, td, off, flag), flush=True)
    return cb


if __name__ == "__main__":
    nom = gains_to_theta(1.0, 0.5, 0.25, K_DOCK)
    e = SpotDockEnv()
    out = {}
    print("case |  nominal  F / T / tot / TOF  |  optimized F / T / tot / TOF | gains tar/obs/los (a0,a1,a2)  k")
    for tc in [2]:
        Fn, Tn, _, tdn, _ = ft_tof(nom, tc)
        print("\ncase %d: optimizing (dock < %.0fs, platform on table)..." % (tc, TOF_MAX), flush=True)
        res = differential_evolution(fit, [(-1, 1)] * 10, seed=0, args=(tc,), maxiter=24,
                                     popsize=14, tol=1e-6, mutation=(0.5, 1.0),
                                     recombination=0.7, polish=False, updating="deferred",
                                     workers=16, callback=make_cb(tc))
        Fo, To, dk, tdo, offo = ft_tof(res.x, tc); out[tc] = [round(float(x), 4) for x in res.x]
        g, k = e._decode(np.array(res.x))
        print("  %d  |  %.2f / %.3f / %.1f / %.0fs  |  %.2f / %.3f / %.1f / %.0fs | %s %s %s  %.2f (off=%.3f)"
              % (tc, Fn, Tn, Fn + Tn / R, tdn, Fo, To, Fo + To / R, tdo,
                 np.round(g[0], 2), np.round(g[1], 2), np.round(g[2], 2), k, offo))
        print("    theta[%d] = %s" % (tc, out[tc]), flush=True)
    print("\nLOWFUEL_THETA = {")
    for tc in [0, 1, 2, 3, 4]:
        print("    %d: %s," % (tc, out[tc]))
    print("}")
