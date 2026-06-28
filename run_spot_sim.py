"""
Run the SPOT scenario:

  * BLACK (target) crosses the granite table tracking a min-jerk reference
    with a discrete LQR at 20 Hz.
  * RED (chaser) tries to hold a station-keeping point 0.65 m "behind" the
    target (nominal LQR), but its translational command is passed through
    an ICCBF safety filter enforcing (i) a keep-out circle around the
    moving target, (ii) table-boundary barriers, (iii) the testbed's real
    input limits (thruster duty cycles in [0,1] with pressure decay).

Outputs: spot_iccbf_trajectory.png, spot_iccbf_timeseries.png, and a CSV log.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse

import spot_params as P
from spot_dynamics import Platform
from spot_controllers import PlanarLQR, ICCBFFilter, AttitudePD, crossing_reference


def run(T=150.0, seed=0):
    dt = P.DT
    n = int(T / dt)
    rng = np.random.default_rng(seed)

    target = Platform("BLACK", x0=[0.60, 0.60, 0.0, 0, 0, 0])
    chaser = Platform("RED",  x0=[2.90, 0.55, np.pi, 0, 0, 0])

    lqr_t = PlanarLQR(target.m, target.I)
    lqr_c = PlanarLQR(chaser.m, chaser.I)
    iccbf = ICCBFFilter(chaser.m, r_keepout=(0.85, 0.85),
                        r_min=(0.80, 0.35), a_tgt_max=0.004)
    att = AttitudePD()

    log = {k: np.zeros((n, 6)) for k in ("xt", "xc", "ref")}
    log.update(u_t=np.zeros((n, 3)), u_c=np.zeros((n, 3)),
               u_nom=np.zeros((n, 2)), b_obs=np.zeros(n),
               b_wall=np.zeros((n, 4)), dist=np.zeros(n),
               duty_c=np.zeros((n, 8)), active=np.zeros(n, dtype=bool),
               r_koz=np.zeros((n, 2)), captured=np.zeros(n, dtype=bool))
    t_axis = np.arange(n) * dt

    for k in range(n):
        t = k * dt
        # ----- target: LQR onto the crossing reference ----------------
        ref = crossing_reference(t)
        u_t = lqr_t.control(target.x, ref)

        # ----- KOZ shrink: contract the keep-out while docked-in -------
        captured = iccbf.update_shrink(chaser.x, target.x)

        # ----- chaser: nominal LQR to the dock point on the KOZ --------
        # The hold point rides the (shrinking) keep-out circle on the
        # target's trailing face, so as r contracts the chaser is pulled in.
        hold = ref.copy()
        hold[:2] = iccbf.dock_point(target.x)
        hold[3:5] = target.x[3:5]
        # face the target
        bearing = np.arctan2(target.x[1] - chaser.x[1],
                             target.x[0] - chaser.x[0])
        hold[2] = bearing
        hold[5] = 0.0
        u_nom6 = lqr_c.control(chaser.x, hold)
        u_nom_f = u_nom6[:2]

        # ----- ICCBF filter on the force channel -----------------------
        u_f, diag = iccbf.filter(chaser.x, u_nom_f, target.x)
        Tz = att.control(chaser.x, bearing)
        u_c = np.array([u_f[0], u_f[1], Tz])

        # ----- step both plants through the real actuation path --------
        _, u_t_real = target.step(u_t)
        duty, u_c_real = chaser.step(u_c)

        # ----- log ------------------------------------------------------
        log["xt"][k], log["xc"][k], log["ref"][k] = target.x, chaser.x, ref
        log["u_t"][k], log["u_c"][k] = u_t_real, u_c_real
        log["u_nom"][k] = u_nom_f
        log["b_obs"][k] = diag["b_obs"]
        log["b_wall"][k] = diag["b_walls"]
        log["duty_c"][k] = duty
        log["active"][k] = diag["active"]
        log["dist"][k] = np.linalg.norm(target.x[:2] - chaser.x[:2])
        log["r_koz"][k] = iccbf.r_vec
        log["captured"][k] = captured

    return t_axis, log, iccbf


def _platform_patch(ax, x, y, th, color, alpha, lw=1.5, size=0.30):
    """Draw the 0.3 m square bus rotated by th, plus a heading tick and a
    small 'docking face' mark on the +x body face (paper-figure style)."""
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    h = size / 2.0
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]]).T
    pts = (R @ corners).T + [x, y]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=3)
    # heading tick: CG to middle of +x face
    tip = R @ np.array([h, 0.0]) + [x, y]
    ax.plot([x, tip[0]], [y, tip[1]], color=color, lw=lw * 0.8,
            alpha=alpha, zorder=3)


def plot(t, log, iccbf, prefix=""):
    # ---------------- trajectory plot (paper-figure style) ----------------
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.set_facecolor("white")
    ax.grid(True, color="0.85", lw=0.8)
    ax.set_axisbelow(True)

    xt, xc = log["xt"], log["xc"]

    # target path: dense black dotted line (as in the reference figure)
    every = max(1, len(t) // 90)
    ax.plot(xt[::every, 0], xt[::every, 1], ls="none", marker=".",
            color="k", ms=4, zorder=2, label="BLACK target path (LQR)")
    # chaser path: solid + dashed pair, dark red
    ax.plot(xc[:, 0], xc[:, 1], color="red", lw=1.4, zorder=2,
            label="RED chaser path (ICCBF)")
    ax.plot(xc[:, 0], xc[:, 1], color="darkred", lw=1.2, ls=(0, (6, 4)),
            zorder=2)

    # platform footprints at snapshots spaced by target arc length
    n_snap = 7
    seg = np.linalg.norm(np.diff(xt[:, :2], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    ks = [int(np.searchsorted(arc, s))
          for s in np.linspace(0, arc[-1], n_snap)]
    ks[-1] = len(t) - 1
    for i, k in enumerate(ks):
        a = 0.15 + 0.85 * i / (n_snap - 1)
        _platform_patch(ax, xc[k, 0], xc[k, 1], xc[k, 2], "red", a,
                        lw=1.0 + 1.2 * (i == n_snap - 1))
        _platform_patch(ax, xt[k, 0], xt[k, 1], xt[k, 2], "k", a,
                        lw=1.0 + 1.0 * (i == n_snap - 1))
        # thin grey line of sight chaser -> target at each snapshot
        ax.plot([xc[k, 0], xt[k, 0]], [xc[k, 1], xt[k, 1]],
                color="0.55", lw=0.6, alpha=0.8, zorder=1)

    # keep-out ellipse (target-body aligned): initial (faint) + final (dark)
    ang = np.rad2deg(xt[-1, 2])
    a0, b0 = log["r_koz"][0]
    a1, b1 = log["r_koz"][-1]
    ax.add_patch(Ellipse(xt[-1, :2], 2*a0, 2*b0, angle=ang, fill=False,
                         ec="0.8", ls="--", lw=0.8, zorder=1))
    ax.add_patch(Ellipse(xt[-1, :2], 2*a1, 2*b1, angle=ang, fill=False,
                         ec="0.4", ls="--", lw=1.0, zorder=1,
                         label=f"keep-out [{a0:.2f},{b0:.2f}]→[{a1:.2f},{b1:.2f}] m"))

    ax.set_xlim(0, P.TABLE_X)
    ax.set_ylim(0, P.TABLE_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("X-Position [m]", fontsize=12)
    ax.set_ylabel("Y-Position [m]", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(prefix + "spot_iccbf_trajectory.png", dpi=160)

    # ---------------- time series ----------------
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axs[0].plot(t, log["dist"], "crimson", label="‖p_RED − p_BLACK‖")
    axs[0].plot(t, log["r_koz"][:, 0], "k--", lw=1, label="KOZ a (shrinking)")
    axs[0].plot(t, log["r_koz"][:, 1], "0.5", ls="--", lw=1,
                label="KOZ b (shrinking)")
    axs[0].set_ylabel("separation [m]"); axs[0].legend(fontsize=8)

    axs[1].plot(t, log["b_obs"], label="b₁ obstacle (ICCBF)")
    axs[1].plot(t, log["b_wall"].min(axis=1), label="min wall b₁")
    axs[1].axhline(0, color="k", lw=1)
    axs[1].set_ylabel("barrier value"); axs[1].legend(fontsize=8)

    axs[2].plot(t, log["u_nom"][:, 0], "C0--", lw=1, label="Fx nominal")
    axs[2].plot(t, log["u_c"][:, 0], "C0", label="Fx realized")
    axs[2].plot(t, log["u_nom"][:, 1], "C1--", lw=1, label="Fy nominal")
    axs[2].plot(t, log["u_c"][:, 1], "C1", label="Fy realized")
    axs[2].axhline(P.F_AXIS_MAX, color="k", ls=":", lw=1)
    axs[2].axhline(-P.F_AXIS_MAX, color="k", ls=":", lw=1)
    axs[2].set_ylabel("chaser force [N]"); axs[2].legend(fontsize=7, ncol=2)

    axs[3].plot(t, log["duty_c"])
    axs[3].set_ylabel("RED duty cycles"); axs[3].set_ylim(-0.02, 1.05)
    axs[3].set_xlabel("time [s]")
    where = log["active"]
    for a in axs:
        a.fill_between(t, *a.get_ylim(), where=where, color="crimson",
                       alpha=0.06, step="mid")
    fig.suptitle("ICCBF filter activity shaded", fontsize=9)
    fig.tight_layout()
    fig.savefig(prefix + "spot_iccbf_timeseries.png", dpi=160)


if __name__ == "__main__":
    t, log, iccbf = run()
    plot(t, log, iccbf)
    dmin = log["dist"].min()
    print(f"keep-out [a,b]: {np.round(log['r_koz'][0],3)} -> "
          f"{np.round(log['r_koz'][-1],3)} m (floor {iccbf.r_min_vec} m)")
    docked = (np.all(iccbf.r_vec <= iccbf.r_min_vec + 1e-3)
              and log["captured"][-1])
    print(f"docked: {docked}  (final separation {log['dist'][-1]:.3f} m)")
    print(f"min separation: {dmin:.3f} m")
    print(f"min obstacle barrier b1: {log['b_obs'].min():.4f}")
    print(f"min wall barrier b1:     {log['b_wall'].min():.4f}")
    print(f"filter active fraction:  {log['active'].mean()*100:.1f}% of steps")
    inside = ((log['xc'][:, 0] > 0) & (log['xc'][:, 0] < P.TABLE_X) &
              (log['xc'][:, 1] > 0) & (log['xc'][:, 1] < P.TABLE_Y)).all()
    print(f"chaser stayed on table:  {inside}")
    np.savetxt("spot_sim_log.csv",
               np.column_stack([t, log['xt'], log['xc'], log['dist'],
                                log['b_obs']]),
               delimiter=",",
               header="t," + ",".join(f"xt{i}" for i in range(6)) + "," +
                      ",".join(f"xc{i}" for i in range(6)) + ",dist,b_obs")
