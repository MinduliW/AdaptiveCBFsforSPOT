"""
Render the SPOT docking scenario (run_spot_sim) as an MP4.

Replays the logged trajectory: BLACK target crossing the table, RED chaser
closing in under the ICCBF filter, and the keep-out circle shrinking from
0.55 m to its floor as the dock is captured. Run after / alongside
run_spot_sim.py -- it calls run() itself so no CSV is needed.

    python animate_spot_sim.py            # -> spot_docking.mp4
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter

import spot_params as P
from run_spot_sim import run


def _square(ax, color, lw=1.8, size=0.30):
    """Return updatable artists for a rotated bus + heading tick."""
    (body,) = ax.plot([], [], color=color, lw=lw, solid_capstyle="round",
                      zorder=4)
    (tick,) = ax.plot([], [], color=color, lw=lw * 0.8, zorder=4)
    return body, tick


def _set_square(body, tick, x, y, th, size=0.30):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    h = size / 2
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]]).T
    pts = (R @ corners).T + [x, y]
    body.set_data(pts[:, 0], pts[:, 1])
    tip = R @ np.array([h, 0.0]) + [x, y]
    tick.set_data([x, tip[0]], [y, tip[1]])


def animate(t, log, out="spot_docking.mp4", fps=30, target_seconds=20.0):
    xt, xc, rk = log["xt"], log["xc"], log["r_koz"]
    n = len(t)
    # stride frames so the clip is ~target_seconds long at fps
    stride = max(1, int(round(n / (fps * target_seconds))))
    frames = range(0, n, stride)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("white")
    ax.grid(True, color="0.88", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, P.TABLE_X)
    ax.set_ylim(0, P.TABLE_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("X-Position [m]", fontsize=12)
    ax.set_ylabel("Y-Position [m]", fontsize=12)

    # static initial keep-out ellipse (faint) for reference
    ax.add_patch(Ellipse(xt[0, :2], 2*rk[0, 0], 2*rk[0, 1],
                         angle=np.rad2deg(xt[0, 2]), fill=False,
                         ec="0.85", ls="--", lw=0.8, zorder=1))

    # trails
    (t_trail,) = ax.plot([], [], "k.", ms=3, zorder=2,
                         label="BLACK target")
    (c_trail,) = ax.plot([], [], color="red", lw=1.4, zorder=2,
                         label="RED chaser")
    (los,) = ax.plot([], [], color="0.55", lw=0.7, zorder=1)
    koz = Ellipse(xt[0, :2], 2*rk[0, 0], 2*rk[0, 1],
                  angle=np.rad2deg(xt[0, 2]), fill=False, ec="0.4",
                  ls="--", lw=1.1, zorder=1)
    ax.add_patch(koz)

    t_body, t_tick = _square(ax, "k")
    c_body, c_tick = _square(ax, "red")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                  fontsize=11, family="monospace",
                  bbox=dict(fc="white", ec="0.7", alpha=0.85))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    def update(k):
        c_trail.set_data(xc[:k + 1, 0], xc[:k + 1, 1])
        t_trail.set_data(xt[:k + 1, 0], xt[:k + 1, 1])
        _set_square(t_body, t_tick, *xt[k, :3])
        _set_square(c_body, c_tick, *xc[k, :3])
        koz.set_center(xt[k, :2])
        koz.set_width(2 * rk[k, 0]); koz.set_height(2 * rk[k, 1])
        koz.set_angle(np.rad2deg(xt[k, 2]))
        los.set_data([xc[k, 0], xt[k, 0]], [xc[k, 1], xt[k, 1]])
        sep = np.linalg.norm(xt[k, :2] - xc[k, :2])
        txt.set_text(f"t = {t[k]:6.1f} s\n"
                     f"keep-out = [{rk[k,0]:.2f}, {rk[k,1]:.2f}] m\n"
                     f"separation = {sep:.3f} m")
        return (c_trail, t_trail, t_body, t_tick, c_body, c_tick, koz, los, txt)

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400,
                          metadata=dict(title="SPOT docking"))
    anim.save(out, writer=writer, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    t, log, iccbf = run()
    out = animate(t, log)
    print(f"wrote {out}  ({len(t)} steps, final KOZ = {np.round(iccbf.r_vec,3)} m, "
          f"final sep = {np.linalg.norm(log['xt'][-1, :2] - log['xc'][-1, :2]):.3f} m)")
