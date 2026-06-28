"""
Controllers:

1) Discrete LQR (target, BLACK): tracks a table-crossing reference on the
   decoupled double integrators (x, y) plus attitude. Solved via the DARE
   at the 20 Hz testbed rate. Q/R defaults echo the GUI's LQR-style weights
   (Q_pos=1, Q_vel=10, Q_th=0.05, Q_w=1; R=6 on each channel).

2) ICCBF safety filter (chaser, RED): Input-Constrained CBFs in the sense
   of Agrawal & Panagou (CDC 2021). For the double integrator with a hard
   acceleration bound a_max, starting from the unsafe-set distance
       h0(x) = ||p_rel|| - r,
   the ICCBF construction with class-K choice alpha0(s) = sqrt(2 a_brake s)
   yields the (closed-form) braking barrier
       b1(x) = sqrt(2 a_brake h0) + d/dt(||p_rel||),
   whose superlevel set is controlled-invariant USING ONLY |u| <= a_max --
   i.e. the input constraint is baked into the barrier instead of being
   discovered (infeasibly) by the QP at runtime. The same construction is
   applied to the four table walls. The filter solves
       min ||u - u_nom||^2  s.t.  Lf b + Lg b u >= -alpha1(b),  u in U(box)
   at every step.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

import spot_params as P
from spot_env import los_row, U_MAX, IRED   # reuse the verified LOS (FOV) ICCBF on the torque channel


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rotz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


# ------------------------------- LQR -----------------------------------
class PlanarLQR:
    """Discrete LQR on [px,py,th,vx,vy,w] with control [Fx,Fy,Tz] (inertial)."""

    def __init__(self, m, I, dt=P.DT,
                 q=(1.0, 1.0, 0.05, 10.0, 10.0, 1.0), r=(6.0, 6.0, 6.0)):
        A = np.eye(6)
        A[0, 3] = A[1, 4] = A[2, 5] = dt
        B = np.zeros((6, 3))
        B[0, 0] = dt**2 / (2*m); B[3, 0] = dt / m
        B[1, 1] = dt**2 / (2*m); B[4, 1] = dt / m
        B[2, 2] = dt**2 / (2*I); B[5, 2] = dt / I
        Q = np.diag(q); R = np.diag(r)
        Pq = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.solve(R + B.T @ Pq @ B, B.T @ Pq @ A)

    def control(self, x, x_ref):
        e = x - x_ref
        e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))
        return -self.K @ e


def crossing_reference(t, t0=10.0, t1=130.0,
                       p_start=(0.60, 0.60), p_end=(2.90, 1.85), th_ref=0.0):
    """Min-jerk table crossing for the target: rest-to-rest, BLACK home-ish
    corner to the far corner, inside the 3.51 x 2.42 m surface."""
    p0, p1 = np.array(p_start), np.array(p_end)
    if t <= t0:
        p, v = p0, np.zeros(2)
    elif t >= t1:
        p, v = p1, np.zeros(2)
    else:
        s = (t - t0) / (t1 - t0)
        f = 10*s**3 - 15*s**4 + 6*s**5
        fd = (30*s**2 - 60*s**3 + 30*s**4) / (t1 - t0)
        p = p0 + f * (p1 - p0)
        v = fd * (p1 - p0)
    return np.array([p[0], p[1], th_ref, v[0], v[1], 0.0])


# ------------------------------ ICCBF ----------------------------------
class ICCBFFilter:
    """ICCBF safety filter for the chaser's translational channel.

    Barriers:
      * keep-out ELLIPSE around the (moving) target, aligned with and
        rotating with the target body frame, with the braking budget
        a_brk = a_max - a_tgt_max (reserve for target motion);
      * four table walls with margin.
    Attitude torque is filtered through the LOS (FOV) ICCBF and bounded to
    +-U_MAX[2]; a torque reserve is also subtracted from the force budget so the
    duty-cycle allocator can always realize the filtered command.
    """

    def __init__(self, m, r_keepout=(0.85, 0.85), a_tgt_max=0.004,
                 wall_margin=0.25, alpha1=0.8, eps=1e-9, zoh_margin=0.04,
                 r_min=(0.80, 0.35), dt=P.DT, t_shrink=4.0, gamma_base=0.95,
                 eta=0.20, zeta=np.deg2rad(10.0),
                 docking_offset=(0.165, 0.427, -np.pi / 2),
                 I=IRED, los_gains=(1.0, 0.5, 0.25)):
        # zoh_margin: the CBF condition is continuous-time but applied with a
        # 20 Hz zero-order hold; inflating the enforced boundary by a few cm
        # absorbs the intersample error so h0 >= 0 holds for the TRUE ellipse.
        self.zoh_margin = zoh_margin
        # Guaranteed per-axis force: nominal x decay floor, minus torque reserve
        F_avail = P.F_AXIS_MIN_GUARANTEED * 0.85
        self.u_max = F_avail / m                # [m/s^2] per-axis accel bound
        # Norm bound usable in any direction (inscribed circle of the box)
        a_max = self.u_max
        self.a_brk_obs = max(a_max - a_tgt_max, 1e-3)
        self.a_brk_wall = a_max
        self.margin = wall_margin
        self.alpha1 = alpha1
        self.eps = eps
        self.m = m
        self.I = I                              # platform inertia -> LOS torque channel (theta_ddot = tau/I)
        self.tau_max = U_MAX[2]                 # torque limit [N.m] (= 0.1*0.15 = 0.015)
        self.los_gains = los_gains              # LOS ICCBF class-K gains (a0,a1,a2)
        # --- anisotropic keep-out ellipse + shrink (Run_Initializer.m) ----
        # r_vec = [a, b] half-axes in the TARGET BODY frame; the ellipse
        # rotates with the target and contracts from r_keepout toward r_min
        # (the SPOT r_hold_min) by gamma_step per step, but ONLY while the
        # chaser holds the body-fixed docking port within (eta linear, zeta
        # angular) -- i.e. the shrink is gated on the chaser's relative state
        # w.r.t. the target, exactly the SPOT holding-radius rule.
        self.r_vec = np.asarray(r_keepout, float).copy()
        self.r_min_vec = np.asarray(r_min, float)
        self.gamma_step = gamma_base ** (dt / t_shrink)
        self.eta = eta
        self.zeta = zeta
        self.docking_offset = np.asarray(docking_offset, float)
        # body-frame bearing of the docking port on the ellipse (cbf_rvd.BETA)
        self.beta = np.pi / 2 - np.arctan2(docking_offset[0], docking_offset[1])
        self.captured = False

    # --- dock port + shrink rule --------------------------------------
    def _ellipse_radius(self, ang_body):
        """Distance from target centre to the ellipse boundary along a body-
        frame bearing ang_body."""
        a, b = self.r_vec
        return a * b / np.sqrt((a * np.sin(ang_body))**2
                               + (b * np.cos(ang_body))**2)

    def dock_point(self, target_state):
        """Docking port: a point fixed in the TARGET BODY frame at bearing
        beta on the ellipse, so it rides the ellipse as it shrinks/rotates."""
        th = target_state[2]
        rho = self._ellipse_radius(self.beta)
        port_body = rho * np.array([np.cos(self.beta), np.sin(self.beta)])
        return target_state[:2] + rotz(th) @ port_body

    def dock_heading(self, x_chaser, target_state):
        """Desired chaser attitude = rotateToFace(target) + docking_offset(3);
        for the SPOT numbers these combine to 'face the target'."""
        bearing = np.arctan2(target_state[1] - x_chaser[1],
                             target_state[0] - x_chaser[0])
        return wrap(bearing + self.docking_offset[2] + np.pi / 2)

    def update_shrink(self, x_chaser, target_state):
        """Contract the keep-out ellipse toward r_min while the chaser holds
        the body-fixed docking port within (eta, zeta) of where it should be.
        Returns True if the dock is captured this step."""
        pos_err = np.linalg.norm(x_chaser[:2] - self.dock_point(target_state))
        th_err = abs(wrap(x_chaser[2]
                          - self.dock_heading(x_chaser, target_state)))
        self.captured = pos_err < self.eta and th_err < self.zeta
        if self.captured and np.any(self.r_vec > self.r_min_vec + 1e-9):
            self.r_vec = np.maximum(self.r_min_vec,
                                    self.gamma_step * self.r_vec)
        return self.captured

    # --- moving-obstacle braking ICCBF (elliptic) --------------------
    def _obstacle_constraint(self, p, v, target_state):
        """Return (A, c, b1) s.t. constraint is A . a_cmd >= c (a in m/s^2).
        h0 is the radial gap to the target-aligned keep-out ellipse boundary;
        reduces exactly to the circular barrier when a == b."""
        p_t, v_t = target_state[:2], target_state[3:5]
        pr = p - p_t
        vr = v - v_t
        d = np.linalg.norm(pr) + self.eps
        n = pr / d
        # boundary radius along n for the body-aligned ellipse Q = R S R^T
        R = rotz(target_state[2])
        Q = R @ np.diag(1.0 / self.r_vec**2) @ R.T
        rho = 1.0 / np.sqrt(n @ Q @ n)
        h0 = d - (rho + self.zoh_margin)
        h0c = max(h0, self.eps)
        ddot = float(n @ vr)
        b1 = np.sqrt(2 * self.a_brk_obs * h0c) + ddot
        # derivatives: d/dt(ddot) = (||vr||^2 - ddot^2)/d + n . (a - a_t);
        # target accel AND the slow boundary drift (rotation/shrink) are
        # treated as disturbances covered by the braking budget.
        lf = np.sqrt(self.a_brk_obs / (2 * h0c)) * ddot \
            + (vr @ vr - ddot**2) / d
        A = n                      # Lg b1 . a  = n . a
        c = -self.alpha1 * b1 - lf
        return A, c, b1

    # --- wall braking ICCBFs ------------------------------------------
    def _wall_constraints(self, p, v):
        """Walls at x=0, x=TABLE_X, y=0, y=TABLE_Y, each with `margin`.
        For each wall: h0 = inward signed distance, b1 = sqrt(2 a h0) + h0_dot,
        constraint  n . a >= -alpha1 b1 - Lf b1  with inward normal n."""
        cons, bvals = [], []
        walls = [
            (np.array([1.0, 0.0]),  p[0] - self.margin,             v[0]),
            (np.array([-1.0, 0.0]), (P.TABLE_X - self.margin) - p[0], -v[0]),
            (np.array([0.0, 1.0]),  p[1] - self.margin,             v[1]),
            (np.array([0.0, -1.0]), (P.TABLE_Y - self.margin) - p[1], -v[1]),
        ]
        for nvec, h0, sd in walls:
            h0c = max(h0, self.eps)
            b1 = np.sqrt(2 * self.a_brk_wall * h0c) + sd
            lf = np.sqrt(self.a_brk_wall / (2 * h0c)) * sd
            cons.append((nvec, -self.alpha1 * b1 - lf))
            bvals.append(b1)
        return cons, bvals

    def _los_torque(self, x, target_state, tau_nom):
        """LOS (FOV) ICCBF on the TORQUE channel + box bound.

        spot_env.los_row returns A=[0,0,-Lg], b with the row A.u <= b living
        PURELY on tau (theta drives the camera boresight, theta_ddot = tau/I). So
        this is a 1-D QP:  min (tau-tau_nom)^2  s.t. g*tau<=b AND |tau|<=U_MAX[2]."""
        A, b, h = los_row(np.asarray(x, float), np.asarray(target_state, float),
                          *self.los_gains, self.I, 1.0)
        g = A[2]                                       # constraint  g*tau <= b
        lo, hi = -self.tau_max, self.tau_max           # torque box +-U_MAX[2]
        if g > self.eps:
            hi = min(hi, b / g)
        elif g < -self.eps:
            lo = max(lo, b / g)
        if lo > hi:                                    # FOV unreachable within +-tau_max
            tau = -self.tau_max if g > 0 else self.tau_max     # max authority toward the cone
        else:
            tau = float(np.clip(tau_nom, lo, hi))
        return tau, h

    def filter(self, x, u_nom_force, target_state, tau_nom=0.0):
        """x: chaser state; u_nom_force: nominal [Fx,Fy] (inertial, N);
        tau_nom: nominal torque [N.m] (e.g. from AttitudePD);
        target_state: target [px,py,th,vx,vy,w].
        Returns filtered control [Fx, Fy, tau] -- tau now passes the LOS ICCBF and
        is bounded to +-U_MAX[2] -- plus diagnostics."""
        p, v = x[:2], x[3:5]
        a_nom = np.clip(u_nom_force / self.m, -self.u_max, self.u_max)

        A_o, c_o, b_obs = self._obstacle_constraint(p, v, target_state)
        wall_cons, b_walls = self._wall_constraints(p, v)

        cons = [{"type": "ineq", "fun": lambda a, A=A_o, c=c_o: A @ a - c}]
        for nvec, c in wall_cons:
            cons.append({"type": "ineq",
                         "fun": lambda a, A=nvec, c=c: A @ a - c})
        bnds = [(-self.u_max, self.u_max)] * 2

        res = minimize(lambda a: np.sum((a - a_nom)**2), a_nom,
                       method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 60, "ftol": 1e-10})
        a = res.x if res.success else self._fallback(a_nom, A_o, c_o)
        F = a * self.m
        tau, b_los = self._los_torque(x, target_state, tau_nom)    # LOS ICCBF + |tau|<=U_MAX[2]
        return np.array([F[0], F[1], tau]), dict(b_obs=b_obs, b_walls=b_walls, b_los=b_los,
                                                 active=not np.allclose(a, a_nom, atol=1e-6))

    def _fallback(self, a_nom, A, c):
        """Analytic halfspace projection + box clip if SLSQP hiccups."""
        a = a_nom.copy()
        if A @ a < c:
            a = a + (c - A @ a) / (A @ A + 1e-12) * A
        return np.clip(a, -self.u_max, self.u_max)


class AttitudePD:
    """Simple attitude channel (the GUI's default D/P gains: 0.5 / 1.8)."""

    def __init__(self, kp=0.5, kd=1.8):
        self.kp, self.kd = kp, kd

    def control(self, x, th_ref, w_ref=0.0):
        e = np.arctan2(np.sin(th_ref - x[2]), np.cos(th_ref - x[2]))
        return self.kp * e + self.kd * (w_ref - x[5])
