"""Find gains that minimize TOTAL propellant (Force impulse + Torque impulse / r)
with the correct moment arm r=0.08 m, per case. Reports force AND torque separately
so we can see whether it reduces BOTH vs the nominal (1,0.5,0.25) gains."""
import numpy as np
from scipy.optimize import differential_evolution
from spot_env import SpotDockEnv, DT, K_DOCK
from tune_gains import gains_to_theta

R = 0.08
HORIZON = int(150 / DT)


def force_torque(theta, tc):
    env = SpotDockEnv(randomize=False, test_case=tc, t_max=150.0); env.reset(seed=0)
    a = np.clip(np.asarray(theta, float), -1, 1).astype(np.float32)
    Fimp = Timp = 0.0; dk = False
    for _ in range(HORIZON):
        _, _, term, trunc, info = env.step(a); u = info["u"]
        Fimp += np.hypot(u[0], u[1]) * DT; Timp += abs(u[2]) * DT
        if info["docked"]:
            dk = True
        if term or trunc:
            break
    return Fimp, Timp, dk


def fit(theta, tc):
    F, T, dk = force_torque(theta, tc)
    return (F + T / R) + (0.0 if dk else 1000.0)      # minimize total propellant; dock required


if __name__ == "__main__":
    nom = gains_to_theta(1.0, 0.5, 0.25, K_DOCK)
    print("case | nominal  F / T / total | optimized F / T / total | dF / dT  | both down?")
    out = {}
    for tc in range(5):
        Fn, Tn, _ = force_torque(nom, tc)
        res = differential_evolution(fit, [(-1, 1)] * 10, seed=0, args=(tc,), maxiter=22,
                                     popsize=14, tol=1e-6, mutation=(0.5, 1.0),
                                     recombination=0.7, polish=False, updating="deferred", workers=-1)
        Fo, To, dk = force_torque(res.x, tc)
        out[tc] = [round(float(x), 4) for x in res.x]
        both = "YES" if (Fo < Fn and To < Tn) else ("torque only" if To < Tn else ("force only" if Fo < Fn else "neither"))
        print("  %d  | %.2f / %.3f / %.2f | %.2f / %.3f / %.2f | %+3.0f%%/%+3.0f%% | %s  dock=%s"
              % (tc, Fn, Tn, Fn + Tn / R, Fo, To, Fo + To / R,
                 100 * (Fo - Fn) / Fn, 100 * (To - Tn) / Tn, both, dk))
    print("\nLOWFUEL_THETA = {")
    for tc in range(5):
        print("    %d: %s," % (tc, out[tc]))
    print("}")
