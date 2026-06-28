"""
Roll out a Run_Initializer.m test case through SpotDockEnv and render it in the
SPOT-GUI frame for visual comparison.

GUI convention (decoded from the GUI screenshots): the horizontal screen axis is
world Y (pointing right) and the vertical screen axis is world X (pointing DOWN),
so the table is portrait xLength(3.5) tall x yLength(2.4) wide. We reproduce that
exactly: every world point (x, y) is plotted at (y, x) with the vertical axis
inverted.

Constant ICCBF gains (a0,a1,a2) = (1.0, 0.5, 0.25); BLACK/BLUE drift force-free.

    python run_matlab_case.py [test_case]      # default 4  -> spot_case<N>.mp4
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter

VIDEO_DIR = "videos"      # bare output filenames are written here

from spot_env import (SpotDockEnv, DT, XL, YL, Rmat, TEST_CASES,
                      DOCK_OFF, SENS_OFF)


# ----------------------------- rollout -------------------------------------
def rollout(test_case=4, gains=(1.0, 0.5, 0.25), horizon=150.0, action=None):
    # horizon >> T_MAX: the GUI runs ~120 s before docking; ignore the env's
    # time-cap truncation here and only stop on a real terminal (dock/collision).
    # action=None -> constant shared gains via setconst; otherwise hold the given
    # 10-D policy action constant every step (per-constraint gains + Lslack).
    if action is None:
        env = SpotDockEnv(randomize=False, setconst=True, const_gains=gains,
                          test_case=test_case)
        a0 = np.zeros(env.action_space.shape[0], dtype=np.float32)   # ignored (setconst)
    else:
        env = SpotDockEnv(randomize=False, setconst=False, test_case=test_case)
        a0 = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
    env.reset(seed=0)
    log = {k: [] for k in ("t", "xR", "xB", "xU", "rk_tar", "rk_obs",
                           "h_tar", "h_obs", "h_los", "V", "u", "fov", "soff")}
    docked = False
    for _ in range(int(horizon / DT)):
        log["t"].append(env.t)
        log["xR"].append(env.xR.copy());  log["xB"].append(env.xB.copy())
        log["xU"].append(env.xU.copy())
        log["rk_tar"].append(env.rkoz_tar.copy()); log["rk_obs"].append(env.r_koz_obs.copy())
        log["fov"].append(env.fov); log["soff"].append(env.sens_off.copy())
        _, _, term, _, info = env.step(a0)
        log["h_tar"].append(info["h_tar"]); log["h_obs"].append(info["h_obs"])
        log["h_los"].append(info["h_los"]); log["V"].append(info["V"])
        log["u"].append(info["u"])
        docked = docked or info["docked"]
        if term:
            break
    for k in log:
        log[k] = np.asarray(log[k])
    return log, docked


# ----------------- world -> GUI-plot transform: (x,y) -> (y, x) -------------
def P(wx, wy):
    """Map world coords (arrays ok) to GUI screen coords (horiz=y, vert=x)."""
    return np.asarray(wy), np.asarray(wx)


def _square_world(x, y, th, size=0.30):
    R = Rmat(th); h = size / 2
    c = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]]).T
    pts = (R @ c).T + [x, y]
    tip = R @ np.array([0.9*size, 0.0]) + [x, y]      # heading antenna
    return pts, tip


def _koz_world(cx, cy, th, a, b, n=60):
    t = np.linspace(0, 2*np.pi, n)
    pts = (Rmat(th) @ np.vstack([a*np.cos(t), b*np.sin(t)])).T + [cx, cy]
    return pts


HALF = 0.15      # body square half-size (matches _square_world)

def _dock_cone_world(state, depth=0.15, w_in=0.05, w_out=0.13):
    """Flared drogue horn sitting FLUSH on the target's +y dock face and
    opening straight out. Base is on the body surface (y = HALF), shifted
    laterally to where the dock axis (DOCK_OFF) exits the face."""
    x, y, th = state[:3]
    off = HALF * DOCK_OFF[0] / DOCK_OFF[1]              # lateral position on face
    d = np.array([0.0, 1.0]); p = np.array([1.0, 0.0])  # +y face normal / in-plane
    c = off * p
    body = np.array([c + HALF*d + w_in*p,
                     c + (HALF+depth)*d + w_out*p,      # mouth (wide, outward)
                     c + (HALF+depth)*d - w_out*p,
                     c + HALF*d - w_in*p])
    return (Rmat(th) @ body.T).T + [x, y]


def _dock_probe_world(state, depth=0.14, w_in=0.06, w_out=0.02):
    """Tapered probe FLUSH on the chaser's +x front face (sensor normal),
    narrowing to the tip that seats in the drogue."""
    x, y, th = state[:3]
    d = np.array([1.0, 0.0]); p = np.array([0.0, 1.0])  # +x face normal / in-plane
    c = SENS_OFF[1] * p                                 # lateral: camera offset
    body = np.array([c + HALF*d + w_in*p,
                     c + (HALF+depth)*d + w_out*p,      # tip (narrow)
                     c + (HALF+depth)*d - w_out*p,
                     c + HALF*d - w_in*p])
    return (Rmat(th) @ body.T).T + [x, y]


def _antenna_world(state, back=0.20, fwd=0.50):
    """Long antenna boom through the satellite body along its heading axis."""
    x, y, th = state[:3]; d = Rmat(th) @ np.array([1.0, 0.0])
    return np.array([[x - back*d[0], y - back*d[1]],
                     [x + fwd*d[0],  y + fwd*d[1]]])


# ----------------------------- animation -----------------------------------
def animate(log, test_case, out=None, fps=30, target_seconds=18.0):
    out = out or f"spot_case{test_case}.mp4"
    if not os.path.dirname(out):                  # bare name -> videos/ folder
        out = os.path.join(VIDEO_DIR, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t, xR, xB, xU = log["t"], log["xR"], log["xB"], log["xU"]
    rkt, rko = log["rk_tar"], log["rk_obs"]
    n = len(t); stride = max(1, int(round(n / (fps * target_seconds))))
    frames = list(range(0, n, stride)) + [n-1]

    fig, ax = plt.subplots(figsize=(7.4, 9.6))
    ax.set_facecolor("white"); ax.grid(True, color="0.85", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, YL); ax.set_ylim(XL, 0)          # vert axis = world X, pointing DOWN
    ax.set_aspect("equal")
    ax.set_xlabel("Y [m]  (world)"); ax.set_ylabel("X [m]  (world, down)")
    ax.set_title(f"SPOT test case {test_case}  (gains 1.0, 0.5, 0.25)")
    # table outline (world corners mapped to plot coords)
    ax.add_patch(Polygon([(0,0),(YL,0),(YL,XL),(0,XL)], closed=True, fill=False,
                 ec="k", lw=2.0, zorder=1))

    (t_trail,) = ax.plot([], [], color="0.4", lw=1.0, zorder=2)
    (c_trail,) = ax.plot([], [], color="red", lw=1.4, zorder=2)
    (u_trail,) = ax.plot([], [], color="tab:blue", lw=1.0, zorder=2)
    (los,) = ax.plot([], [], color="0.55", lw=0.7, zorder=2)
    cone = Polygon(np.zeros((3, 2)), closed=True, fc="red", ec="red",
                   alpha=0.10, zorder=1)
    ax.add_patch(cone)
    (koz_t,) = ax.plot([], [], color="0.45", lw=1.1, ls="--", zorder=2,
                       label="BLACK keep-out")
    (koz_o,) = ax.plot([], [], color="tab:blue", lw=1.1, ls="--", zorder=2,
                       label="BLUE keep-out")
    (t_body,) = ax.plot([], [], "k-", lw=1.8, zorder=4, label="BLACK target")
    (t_head,) = ax.plot([], [], "k-", lw=1.0, zorder=4)               # antenna boom
    horn = Polygon(np.zeros((4, 2)), closed=True, fc="0.55", ec="0.3", lw=1.0, zorder=3)
    ax.add_patch(horn)                                                # drogue cone
    (c_body,) = ax.plot([], [], color="red", lw=1.8, zorder=4, label="RED chaser")
    (c_head,) = ax.plot([], [], color="red", lw=1.0, zorder=4)
    probe = Polygon(np.zeros((4, 2)), closed=True, fc="red", ec="darkred", lw=1.0, zorder=6)
    ax.add_patch(probe)                                               # docking probe
    (u_body,) = ax.plot([], [], color="tab:blue", lw=1.8, zorder=4, label="BLUE obstacle")
    (u_head,) = ax.plot([], [], color="tab:blue", lw=1.2, zorder=4)
    txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, va="bottom", fontsize=10,
                  family="monospace", bbox=dict(fc="white", ec="0.7", alpha=0.85))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

    def set_body(body, head, state):
        pts, tip = _square_world(state[0], state[1], state[2])
        body.set_data(*P(pts[:, 0], pts[:, 1]))
        head.set_data(*P([state[0], tip[0]], [state[1], tip[1]]))

    def update(k):
        c_trail.set_data(*P(xR[:k+1, 0], xR[:k+1, 1]))
        t_trail.set_data(*P(xB[:k+1, 0], xB[:k+1, 1]))
        u_trail.set_data(*P(xU[:k+1, 0], xU[:k+1, 1]))
        set_body(t_body, t_head, xB[k]); set_body(c_body, c_head, xR[k])
        set_body(u_body, u_head, xU[k])
        ant = _antenna_world(xB[k]); t_head.set_data(*P(ant[:, 0], ant[:, 1]))  # long boom
        tc = _dock_cone_world(xB[k]);  horn.set_xy(np.column_stack([tc[:, 1], tc[:, 0]]))
        pc = _dock_probe_world(xR[k]); probe.set_xy(np.column_stack([pc[:, 1], pc[:, 0]]))
        kt = _koz_world(xB[k,0], xB[k,1], xB[k,2], rkt[k,0], rkt[k,1])
        ko = _koz_world(xU[k,0], xU[k,1], xU[k,2], rko[k,0], rko[k,1])
        koz_t.set_data(*P(kt[:,0], kt[:,1])); koz_o.set_data(*P(ko[:,0], ko[:,1]))
        # FOV cone: apex at camera mount, bisector = bearing to BLACK, half-angle fov
        sp = xR[k,:2] + Rmat(xR[k,2]) @ log["soff"][k]
        bear = np.arctan2(xB[k,1]-sp[1], xB[k,0]-sp[0]); L = 4.0; fov = log["fov"][k]
        apex = P(sp[0], sp[1])
        e1 = P(sp[0]+L*np.cos(bear+fov), sp[1]+L*np.sin(bear+fov))
        e2 = P(sp[0]+L*np.cos(bear-fov), sp[1]+L*np.sin(bear-fov))
        cone.set_xy([[apex[0],apex[1]], [e1[0],e1[1]], [e2[0],e2[1]]])
        los.set_data(*P([sp[0], xB[k,0]], [sp[1], xB[k,1]]))
        txt.set_text(f"t = {t[k]:5.1f} s   sep = {np.linalg.norm(xB[k,:2]-xR[k,:2]):.3f} m\n"
                     f"h_tar={log['h_tar'][k]:+.2f} h_obs={log['h_obs'][k]:+.2f} "
                     f"h_los={log['h_los'][k]:+.2f}")
        return (c_trail, t_trail, u_trail, los, cone, koz_t, koz_o, t_body, t_head,
                horn, c_body, c_head, probe, u_body, u_head, txt)

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=2400,
              metadata=dict(title=f"SPOT case {test_case}")), dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    tc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    xR0, xB0, xU0 = TEST_CASES[tc]
    print(f"case {tc}:  RED={np.round(xR0,3)}  BLACK={np.round(xB0,3)}  BLUE={np.round(xU0,3)}")
    log, docked = rollout(tc)
    out = animate(log, tc)
    sep = np.linalg.norm(log["xB"][-1, :2] - log["xR"][-1, :2])
    print(f"wrote {out}  ({len(log['t'])} steps, {log['t'][-1]:.1f} s)")
    print(f"  docked={docked}  final sep={sep:.3f} m")
    print(f"  min margins  h_tar={log['h_tar'].min():+.3f}  "
          f"h_obs={log['h_obs'].min():+.3f}  h_los={log['h_los'].min():+.3f}")
