"""
CBF-RVD scenario (test_case 2 from Run_Initializer.m):

  RED   (chaser)  : starts [1.5, 2.0] facing the target; controlled by the
                    ported CBF-QP (dock CLF + KOZ ICCBF + obstacle HOCBF +
                    LOS HOCBF + velocity CBF), wrench realized through the
                    SPOT thruster duty-cycle allocator.
  BLACK (target)  : [2.2, 0.2, 270 deg], drifting (-0.005, +0.005) m/s and
                    tumbling at +1.5 deg/s -- coasts uncontrolled (matches
                    the constant-velocity assumption baked into the CBFs).
  BLUE  (obstacle): [2.0, 1.2, 0], drifting -0.015 m/s in x, coasting.

The KOZ ellipse starts at [0.85, 0.85] and shrinks (gamma rule) toward
[0.80, 0.42] while the chaser holds the dock point, then RED closes to the
docking offset [0.165, 0.427, -90 deg] in the target body frame.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import spot_params as P
from spot_dynamics import Platform, rot
import cbf_rvd as C


def initial_red_attitude(p_red, p_black):
    """x_RED_0(3) per Run_Initializer: face target, then docking offset."""
    th0 = np.pi
    sensor_offset = np.array([0.145 - 0.042, -0.0395])
    los = p_black - (p_red + rot(th0) @ sensor_offset)
    bearing = th0 + C.wrap(np.arctan2(los[1], los[0]) - th0)
    return C.wrap(bearing + C.DOCKING_OFFSET[2] + np.pi / 2)


def run(T=220.0):
    dt = P.DT
    n = int(T / dt)
    d2r = C.D2R

    x_R0 = [1.5, 2.0, 0.0, 0, 0, 0]
    x_B0 = [2.2, 0.2, 270 * d2r, -0.005, 0.005, 1.5 * d2r]
    x_U0 = [2.0, 1.2, 0.0, -0.015, 0.0, 0.0]
    x_R0[2] = initial_red_attitude(np.array(x_R0[:2]), np.array(x_B0[:2]))

    red = Platform("RED", x0=x_R0)
    black = Platform("BLACK", x0=x_B0)
    blue = Platform("BLUE", x0=x_U0)
    ctrl = C.CBFRVDController(red.m, red.I, dt)

    keys = ("xr", "xb", "xu")
    log = {k: np.zeros((n, 6)) for k in keys}
    log.update(u=np.zeros((n, 3)), delta=np.zeros(n), ok=np.zeros(n, bool),
               h_koz=np.zeros(n), h_obs=np.zeros(n), h_los=np.zeros(n),
               h_vel=np.zeros(n), V=np.zeros(n), rkoz=np.zeros((n, 2)),
               edock=np.zeros(n))
    t_axis = np.arange(n) * dt

    for k in range(n):
        u, d = ctrl.control(red.x, black.x, blue.x, obstacle_active=True)
        red.step(np.array(u))            # through duty-cycle allocation
        black.step(np.zeros(3))          # coast (constant velocity/spin)
        blue.step(np.zeros(3))
        log["xr"][k], log["xb"][k], log["xu"][k] = red.x, black.x, blue.x
        log["u"][k] = u
        log["delta"][k] = d["delta"]
        log["ok"][k] = d["ok"]
        for key in ("h_koz", "h_obs", "h_los", "h_vel", "V"):
            log[key][k] = d[key]
        log["rkoz"][k] = d["r_KOZ"]
        log["edock"][k] = np.linalg.norm(d["e"][:2])

    return t_axis, log


def _platform_patch(ax, x, y, th, color, alpha, lw=1.4, size=0.30):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    h = size / 2
    pts = (R @ np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]]).T).T + [x, y]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha, zorder=3)
    tip = R @ np.array([h, 0.0]) + [x, y]
    ax.plot([x, tip[0]], [y, tip[1]], color=color, lw=lw * 0.8, alpha=alpha, zorder=3)


def plot(t, log, prefix=""):
    xr, xb, xu = log["xr"], log["xb"], log["xu"]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(True, color="0.85", lw=0.8)
    ax.set_axisbelow(True)

    every = max(1, len(t) // 90)
    ax.plot(xb[::every, 0], xb[::every, 1], "k.", ms=4, zorder=2,
            label="BLACK target (tumbling, coasting)")
    ax.plot(xu[::every, 0], xu[::every, 1], ls="none", marker=".",
            color="royalblue", ms=4, zorder=2, label="BLUE obstacle")
    ax.plot(xr[:, 0], xr[:, 1], "red", lw=1.4, zorder=2,
            label="RED chaser (CBF-RVD QP)")
    ax.plot(xr[:, 0], xr[:, 1], color="darkred", lw=1.1, ls=(0, (6, 4)), zorder=2)

    n_snap = 7
    ks = np.linspace(0, len(t) - 1, n_snap).astype(int)
    for i, k in enumerate(ks):
        a = 0.15 + 0.85 * i / (n_snap - 1)
        _platform_patch(ax, *xr[k, :3], "red", a, lw=1.0 + 1.2 * (i == n_snap - 1))
        _platform_patch(ax, *xb[k, :3], "k", a, lw=1.0 + 1.0 * (i == n_snap - 1))
        _platform_patch(ax, *xu[k, :3], "royalblue", a)
        ell = Ellipse(xb[k, :2], 2 * log["rkoz"][k, 0], 2 * log["rkoz"][k, 1],
                      angle=np.rad2deg(xb[k, 2]), fill=False, ec="0.55",
                      ls="--", lw=0.7, alpha=0.4 + 0.4 * (i == n_snap - 1))
        ax.add_patch(ell)

    ax.set_xlim(0, P.TABLE_X); ax.set_ylim(0, P.TABLE_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("X-Position [m]", fontsize=12)
    ax.set_ylabel("Y-Position [m]", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(prefix + "cbf_rvd_trajectory.png", dpi=160)

    fig, axs = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    axs[0].plot(t, log["h_koz"], "k"); axs[0].axhline(0, color="r", lw=1)
    axs[0].set_ylabel("h_KOZ,tar")
    axs[1].plot(t, log["h_obs"], "k"); axs[1].axhline(0, color="r", lw=1)
    axs[1].set_ylabel("h_KOZ,obs")
    axs[2].plot(t, np.rad2deg(log["h_los"]), "k"); axs[2].axhline(0, color="r", lw=1)
    axs[2].set_ylabel("h_LOS [deg]")
    axs[3].plot(t, log["h_vel"], "k"); axs[3].axhline(0, color="r", lw=1)
    axs[3].set_ylabel("h_vel [m/s]")
    axs[4].plot(t, log["edock"], "k", label="‖pos err to dock‖")
    axs[4].plot(t, log["rkoz"][:, 0], "C0--", lw=1, label="r_KOZ a")
    axs[4].plot(t, log["rkoz"][:, 1], "C1--", lw=1, label="r_KOZ b")
    axs[4].set_ylabel("[m]"); axs[4].set_xlabel("time [s]")
    axs[4].legend(fontsize=8)
    for a in axs:
        a.grid(True, color="0.9")
    fig.suptitle("CBF-RVD constraint validation (ValidateCBF.m layout)", fontsize=10)
    fig.tight_layout()
    fig.savefig(prefix + "cbf_rvd_constraints.png", dpi=160)


if __name__ == "__main__":
    t, log = run()
    plot(t, log)
    print(f"min h_KOZ,tar : {log['h_koz'].min(): .4f}")
    print(f"min h_KOZ,obs : {log['h_obs'].min(): .4f}")
    print(f"min h_LOS     : {np.rad2deg(log['h_los'].min()): .2f} deg")
    print(f"min h_vel     : {log['h_vel'].min(): .4f} m/s")
    print(f"final dock pos error: {log['edock'][-1]:.4f} m")
    print(f"final r_KOZ  : {log['rkoz'][-1]}")
    print(f"QP success    : {log['ok'].mean()*100:.1f}% of steps")
