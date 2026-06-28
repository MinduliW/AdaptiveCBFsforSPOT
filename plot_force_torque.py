"""Plot |F| and |T| over time for case 1 with the nominal gains (1,0.5,0.25),
to compare our env's control profile against the real SPOT sim."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spot_env import SpotDockEnv, DT, K_DOCK

env = SpotDockEnv(randomize=False, setconst=True, const_gains=(1.0, 0.5, 0.25),
                  const_kdock=K_DOCK, test_case=1, t_max=100.0)
env.reset(seed=0)
ts, F, T = [], [], []
for _ in range(int(100 / DT)):
    _, _, term, trunc, info = env.step(np.zeros(10))      # setconst -> action ignored
    u = info["u"]
    ts.append(env.t); F.append(np.hypot(u[0], u[1])); T.append(abs(u[2]))
    if not (0 < env.xR[0] < 3.5 and 0 < env.xR[1] < 2.4):  # only stop if it flew off table
        break
ts, F, T = np.array(ts), np.array(F), np.array(T)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
a1.plot(ts, F, "b", lw=0.8); a1.set_ylabel("|F| [N]"); a1.set_ylim(0, 0.2); a1.grid(alpha=0.3)
a1.set_title("Our env -- case 1, nominal gains (1, 0.5, 0.25)")
a2.plot(ts, T, "b", lw=0.8); a2.set_ylabel("|T| [Nm]"); a2.set_ylim(0, 0.016)
a2.set_xlabel("Time [s]"); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figures/case1_force_torque.png", dpi=120)

Fimp, Timp = np.sum(F) * DT, np.sum(T) * DT
print("wrote figures/case1_force_torque.png")
print("OUR env case-1 nominal:  Force impulse = %.3f Ns   Torque impulse = %.4f Nms" % (Fimp, Timp))
print("REAL SPOT case-1 Manual: Force impulse = 1.969 Ns   Torque impulse = 0.7955 Nms")
