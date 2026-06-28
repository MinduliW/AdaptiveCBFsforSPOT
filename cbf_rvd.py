"""
Python port of the CBF_RVD controller stack from the user's SPOT model
(test2.zip). Faithful to the Simulink MATLAB Function blocks:

  * KOZ ICCBF (N=2)  -- re-derived symbolically with sympy, mirroring
    build_iccbf_functions.m / iccbf_row(): augmented autonomous state
    X=[chaser(6); target(6)], constant-velocity target gated by tv,
    b1 = Lf h + a0 h ; W = sum_j sqrt((Lg b1)_j^2 + eps^2) * umax_j ;
    b2 = Lf b1 + a1 b1 - W ; QP row  -Lg b2 * u <= Lf b2 + a2 b2.
    Gains a0=a1=2.5, a2=0.5, Fmax=0.1 (chart_608).
  * LOS HOCBF (chart_673): angular cone +-FOV about the bearing theta*,
    torque channel only, two rows, gains k1_LOS=0.1, k2_LOS=1.2.
  * Obstacle KOZ HOCBF (chart_518): rotating ellipse around BLUE,
    b = Lf2h + 2*ddtdhdxf + ddhddt + 2*sqrt(k1)*k2*h_dot + k1*h.
  * Velocity CBF (chart_576): h = v_max - |v|, relative degree 1.
  * Dock CLF (chart_629): V = e' Q_dock e with Q_dock built from
    P=0, R=2I, lambda_dock=0.25  (i.e. V = 2*||e_v + lambda e_p||^2
    per axis plus the attitude pair), k_dock=5, slack column -1.
  * QP (chart_555): min u'u + p_clf*delta^2, A[u;delta] <= b,
    -u_max <= u <= u_max, delta >= 0, warm-started active-set.
  * KOZ shrink logic (chart_495): gamma scaling toward r_KOZ_min when
    the chaser holds the ellipse dock point within (eta, zeta).
"""

import numpy as np
import sympy as sp
from scipy.optimize import minimize

# --------------------------- parameters (Run_Initializer.m) -----------
D2R = np.pi / 180

U_MAX = 0.1 * np.array([1.0, 1.0, 0.3 / 2])      # [Fx, Fy, tau] limits
R_KOZ_TAR_MIN = np.array([0.80, 0.42])
R_KOZ_TAR_INI = np.array([0.85, 0.85])
T_S = 4.0
ETA = np.sqrt(0.5)
ZETA = 10 * D2R
K1_KOZ_TAR, K2_KOZ_TAR = 0.1, 1.2                # (legacy HOCBF gains)
A0_KOZ, A1_KOZ, A2_KOZ = 2.5, 2.5, 0.5           # ICCBF gains (chart_608)
R_KOZ_OBS = 0.43 * np.array([1.0, 1.0])
K1_KOZ_OBS, K2_KOZ_OBS = 0.075, 1.2
SENSOR_FOV = 40 * D2R
K1_LOS, K2_LOS = 0.1, 1.2
V_MAX = 0.1
K_VEL = 0.075
P_CLF = 10.0
LAMBDA_DOCK = 0.25
K_DOCK = 5.0
DOCKING_OFFSET = np.array([0.165, 0.427, -np.pi / 2])
BETA = np.pi / 2 - np.arctan2(DOCKING_OFFSET[0], DOCKING_OFFSET[1])
TV_CBF = 1.0

_R3 = 2 * np.eye(3)
Q_DOCK = np.block([[0 * np.eye(3) + LAMBDA_DOCK**2 * _R3, LAMBDA_DOCK * _R3],
                   [LAMBDA_DOCK * _R3, _R3]])


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rotz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _rot2_sym(th):
    """Symbolic 2-D rotation matrix (counter-clockwise by `th`)."""
    return sp.Matrix([[sp.cos(th), -sp.sin(th)],
                      [sp.sin(th), sp.cos(th)]])


# ------------------------- shared ICCBF recursion -----------------------
def _lie(expr, X, f, g):
    """Lie derivatives of a scalar barrier `expr` along the dynamics.

    Returns (Lf, Lg) where Lf is the scalar drift derivative and Lg is the
    1x3 control derivative (a row over the three actuators).
    """
    grad = sp.Matrix([expr]).jacobian(X)
    Lf = (grad * f)[0, 0]
    Lg = grad * g
    return Lf, Lg


def _iccbf_n2_row(h, X, f, g, a0, a1, a2, u_max, eps_abs=sp.Float(1e-9)):
    """Build the relative-degree-2 Input-Constrained CBF QP row.

    Mirrors build_iccbf_functions.m (N=2). Starting from a barrier h(X), the
    ICCBF recursion adds one robustness margin per derivative step:

        b1 = Lf h  + a0 * h
        W  = sum_j sqrt((Lg b1)_j^2 + eps^2) * u_max      # input-constraint margin
        b2 = Lf b1 + a1 * b1 - W
        QP row:   -Lg(b2) . u  <=  Lf(b2) + a2 * b2

    The smoothed sqrt(.^2 + eps^2) is a differentiable stand-in for |Lg b1|;
    W subtracts the worst-case actuator authority so the filter stays
    feasible under the box input limit u_max.

    Returns (A, b): A is the 1x3 symbolic constraint row, b the scalar bound.
    """
    Lf_h, _ = _lie(h, X, f, g)
    b1 = Lf_h + a0 * h

    Lf_b1, Lg_b1 = _lie(b1, X, f, g)
    margin = sum(sp.sqrt(Lg_b1[j]**2 + eps_abs**2) * u_max for j in range(3))
    b2 = Lf_b1 + a1 * b1 - margin

    Lf_b2, Lg_b2 = _lie(b2, X, f, g)
    A = -Lg_b2
    b = Lf_b2 + a2 * b2
    return A, b


# ------------------- KOZ ICCBF: symbolic derivation -------------------
def _build_koz_iccbf():
    """Mirror build_iccbf_functions.m exactly (KOZ branch, N=2).

    Barrier: chaser stays outside the target keep-out ellipse, where the
    ellipse is rotated by the target attitude xB[2] and sized by (rK1, rK2).
    """
    # Augmented autonomous state X = [chaser(6); target(6)].
    xR = sp.Matrix(sp.symbols('xR1:7', real=True))
    xB = sp.Matrix(sp.symbols('xB1:7', real=True))
    X = sp.Matrix.vstack(xR, xB)
    mRED, a0, a1, a2, Fmax, tv = sp.symbols('mRED a0 a1 a2 Fmax tv', real=True)
    rK1, rK2 = sp.symbols('rK1 rK2', real=True, positive=True)

    # Drift: double integrator for the chaser; constant-velocity target,
    # gated by tv (tv=0 freezes the target for a time-invariant barrier).
    f = sp.Matrix([xR[3], xR[4], xR[5], 0, 0, 0,
                   tv * xB[3], tv * xB[4], tv * xB[5], 0, 0, 0])

    # Control matrix: translational forces on the chaser only.
    g = sp.zeros(12, 3)
    g[3, 0] = 1 / mRED
    g[4, 1] = 1 / mRED
    g[5, 2] = 1 / mRED   # placeholder 1/I; Lg of KOZ wrt torque is 0 anyway

    # Barrier h = d' Q d - 1, with Q the inverse-ellipse metric in the
    # target frame and d the chaser-to-target offset in the world frame.
    S = sp.diag(1 / rK1**2, 1 / rK2**2)
    Rt = _rot2_sym(xB[2])
    Q = Rt * S * Rt.T
    d = sp.Matrix(xR[0:2]) - sp.Matrix(xB[0:2])
    h = (d.T * Q * d)[0, 0] - 1

    A, b = _iccbf_n2_row(h, X, f, g, a0, a1, a2, Fmax)

    args = (list(xR) + list(xB) + [mRED, rK1, rK2, a0, a1, a2, Fmax, tv])
    fA = sp.lambdify(args, [A[0], A[1], A[2]], modules='numpy')
    fb = sp.lambdify(args, b, modules='numpy')
    return fA, fb


_KOZ_A, _KOZ_B = _build_koz_iccbf()


def koz_iccbf_row(x_RED, x_BLACK, mRED, r_KOZ_tar,
                  a0=A0_KOZ, a1=A1_KOZ, a2=A2_KOZ,
                  Fmax=U_MAX[0], tv=TV_CBF):
    """[A(1x4), b] for the target keep-out ICCBF (fcn_KOZ_ICCBF_gen)."""
    args = (*x_RED, *x_BLACK, mRED, r_KOZ_tar[0], r_KOZ_tar[1],
            a0, a1, a2, Fmax, tv)
    A3 = np.array(_KOZ_A(*args), dtype=float)
    b = float(_KOZ_B(*args))
    return np.append(A3, 0.0), b


# ------------------------ obstacle KOZ HOCBF (chart_518) ---------------
def koz_obs_row(x_RED, x_OBS, mRED, r_KOZ_obs=R_KOZ_OBS,
                k1=K1_KOZ_OBS, k2=K2_KOZ_OBS, tv=TV_CBF, active=True):
    if not active:
        return np.zeros((1, 4)), np.zeros(1)
    xo, yo, tho, xod, yod, thod = x_OBS
    x, y = x_RED[0], x_RED[1]
    xd, yd = x_RED[3], x_RED[4]
    a, b_ = r_KOZ_obs
    A = (np.cos(tho) / a)**2 + (np.sin(tho) / b_)**2
    B = (np.sin(tho) / a)**2 + (np.cos(tho) / b_)**2
    C = 2 * np.sin(tho) * np.cos(tho) * (1 / a**2 - 1 / b_**2)
    A_d = thod * np.sin(2 * tho) * (b_**-2 - a**-2)
    B_d = thod * np.sin(2 * tho) * (a**-2 - b_**-2)
    C_d = 2 * thod * np.cos(2 * tho) * (a**-2 - b_**-2)
    A_dd = 2 * thod**2 * np.cos(2 * tho) * (b_**-2 - a**-2)
    B_dd = 2 * thod**2 * np.cos(2 * tho) * (a**-2 - b_**-2)
    C_dd = -4 * thod**2 * np.sin(2 * tho) * (a**-2 - b_**-2)
    ex, ey = x - xo, y - yo
    Lfh = 2 * A * ex * xd + 2 * B * ey * yd + C * (xd * ey + yd * ex)
    Lf2h = 2 * A * xd**2 + 2 * B * yd**2 + 2 * C * xd * yd
    LgLfh = np.array([2 * A * ex + C * ey, 2 * B * ey + C * ex, 0.0]) / mRED
    if tv == 1:
        dhdt = (A_d * ex**2 + B_d * ey**2 + C_d * ex * ey
                - 2 * A * ex * xod - 2 * B * ey * yod
                - C * (xod * ey + yod * ex))
        ddhddt = (A_dd * ex**2 + B_dd * ey**2 + C_dd * ex * ey
                  - 4 * A_d * ex * xod - 4 * B_d * ey * yod
                  - 2 * C_d * (xod * ey + yod * ex)
                  + 2 * A * xod**2 + 2 * B * yod**2 + 2 * C * xod * yod)
        ddtdhdxf = (2 * A_d * ex * xd - 2 * A * xod * xd
                    + 2 * B_d * ey * yd - 2 * B * yod * yd
                    + C_d * (xd * ey + yd * ex) - C * (xod * yd + yod * xd))
    else:
        dhdt = ddhddt = ddtdhdxf = 0.0
    h = A * ex**2 + B * ey**2 + C * ex * ey - 1
    h_dot = Lfh + dhdt
    A_row = np.append(-LgLfh, 0.0)
    b_row = (Lf2h + 2 * ddtdhdxf + ddhddt
             + 2 * np.sqrt(k1) * k2 * h_dot + k1 * h)
    return A_row[None, :], np.array([b_row])


# ------------------- LOS ICCBF: symbolic derivation -------------------
SENSOR_OFFSET = np.array([0.145 - 0.042, -0.0395])
SENSOR_NORMAL = np.array([1.0, 0.0])
SENSOR_TARGET = np.array([0.0825, 0.2516])
A0_LOS, A1_LOS, A2_LOS = 0.8, 1.2, 0.5           # chart_465 / build script


def _build_los_iccbf():
    """Mirror build_iccbf_functions.m LOS branch (torque-only g, N=2).

    Barrier: the target stays inside the chaser's sensor cone of half-angle
    FOV. Only the torque channel can affect it, so g acts on xR[5] alone.
    """
    # Augmented autonomous state X = [chaser(6); target(6)].
    xR = sp.Matrix(sp.symbols('yR1:7', real=True))
    xB = sp.Matrix(sp.symbols('yB1:7', real=True))
    X = sp.Matrix.vstack(xR, xB)
    IRED, a0, a1, a2, taumax, tv, FOV = sp.symbols(
        'IRED a0 a1 a2 taumax tv FOV', real=True)
    sn = sp.Matrix(sp.symbols('sn1:3', real=True))    # sensor normal (body)
    so = sp.Matrix(sp.symbols('so1:3', real=True))    # sensor offset (body)
    stg = sp.Matrix(sp.symbols('stg1:3', real=True))  # target dock point (body)

    # Drift: double integrator for the chaser; constant-velocity target (gated).
    f = sp.Matrix([xR[3], xR[4], xR[5], 0, 0, 0,
                   tv * xB[3], tv * xB[4], tv * xB[5], 0, 0, 0])

    # Control matrix: torque on the chaser only.
    g = sp.zeros(12, 3)
    g[5, 2] = 1 / IRED

    # Line-of-sight vector rL (sensor -> target point) and boresight eL, both
    # in the world frame. h >= 0 iff the target lies within +-FOV of boresight.
    Rc = _rot2_sym(xR[2])    # chaser body -> world
    Rt = _rot2_sym(xB[2])    # target body -> world
    rL = sp.Matrix(xB[0:2]) + Rt * stg - sp.Matrix(xR[0:2]) - Rc * so
    eL = Rc * sn
    cFOV = sp.cos(FOV)**2
    h = (rL.T * eL)[0, 0]**2 - cFOV * (rL.T * rL)[0, 0]

    A, b = _iccbf_n2_row(h, X, f, g, a0, a1, a2, taumax)

    args = (list(xR) + list(xB) + [IRED, FOV] + list(sn) + list(so)
            + list(stg) + [a0, a1, a2, taumax, tv])
    fA = sp.lambdify(args, [A[0], A[1], A[2]], modules='numpy')
    fb = sp.lambdify(args, b, modules='numpy')
    return fA, fb


_LOS_A, _LOS_B = _build_los_iccbf()


def los_iccbf_row(x_RED, x_BLACK, IRED, FOV=SENSOR_FOV,
                  sn=SENSOR_NORMAL, so=SENSOR_OFFSET, stg=SENSOR_TARGET,
                  a0=A0_LOS, a1=A1_LOS, a2=A2_LOS,
                  taumax=U_MAX[2], tv=TV_CBF):
    """[A(1x4), b] for the sensor-cone ICCBF (fcn_LOS_ICCBF_gen)."""
    args = (*x_RED, *x_BLACK, IRED, FOV, *sn, *so, *stg,
            a0, a1, a2, taumax, tv)
    A3 = np.array(_LOS_A(*args), dtype=float)
    b = float(_LOS_B(*args))
    return np.append(A3, 0.0)[None, :], np.array([b])


# --------------------------- LOS HOCBF (chart_673) ----------------------
def los_rows(x_RED, x_BLACK, IRED, k1=K1_LOS, k2=K2_LOS,
             theta_FOV=SENSOR_FOV, tv=TV_CBF):
    x_t, y_t = x_BLACK[0], x_BLACK[1]
    xtd, ytd = x_BLACK[3], x_BLACK[4]
    x, y, th = x_RED[0], x_RED[1], x_RED[2]
    thd = x_RED[5]
    # theta* = rotateToFace(theta, r): bearing, unwrapped relative to theta
    rx, ry = x_t - x, y_t - y
    theta_star = th + wrap(np.arctan2(ry, rx) - th)
    rmagsq = rx**2 + ry**2
    Lfh = -thd
    LgLfh = np.array([0.0, 0.0, -1.0 / IRED])
    if tv == 1:
        dh_dt = (ytd * rx - xtd * ry) / rmagsq
        ddh_dtt = -2 * ((rx * ytd - ry * xtd) * (rx * xtd + ry * ytd)) / rmagsq**2
        ddtdhdxf = 0.0
    else:
        dh_dt = ddh_dtt = ddtdhdxf = 0.0
    h_up = -th + theta_star + theta_FOV
    h_lo = th - theta_star + theta_FOV
    hd_up = Lfh + dh_dt
    hd_lo = -Lfh - dh_dt
    A_up = np.append(-LgLfh, 0.0)
    A_lo = np.append(LgLfh, 0.0)
    b_up = 2 * ddtdhdxf + ddh_dtt + 2 * np.sqrt(k1) * k2 * hd_up + k1 * h_up
    b_lo = -2 * ddtdhdxf - ddh_dtt + 2 * np.sqrt(k1) * k2 * hd_lo + k1 * h_lo
    return np.vstack([A_lo, A_up]), np.array([b_lo, b_up])


# ------------------------- velocity CBF (chart_576) ---------------------
def vel_row(x_RED, mRED, k_vel=K_VEL, v_max=V_MAX, eps=1e-9):
    xd, yd = x_RED[3], x_RED[4]
    v = np.sqrt(xd**2 + yd**2) + eps
    Lgh = np.array([-xd / (mRED * v), -yd / (mRED * v), 0.0])
    h = v_max - v
    return np.append(-Lgh, 0.0)[None, :], np.array([k_vel * h])


# --------------------------- dock CLF (chart_629) -----------------------
def dock_row(x_RED, x_BLACK, mRED, IRED,
             docking_offset=DOCKING_OFFSET, Q=Q_DOCK, k=K_DOCK, tv=TV_CBF):
    B = np.zeros((6, 3))
    B[3, 0] = 1 / mRED
    B[4, 1] = 1 / mRED
    B[5, 2] = 1 / IRED
    R = rotz(x_BLACK[2])
    dR = x_BLACK[5] * np.array([[-np.sin(x_BLACK[2]), -np.cos(x_BLACK[2])],
                                [np.cos(x_BLACK[2]), -np.sin(x_BLACK[2])]])
    # NOTE on dR: chart_629 multiplies x_BLACK(6)*dR with dR the unscaled
    # rotation derivative; reproduce exactly:
    dR = np.array([[-np.sin(x_BLACK[2]), -np.cos(x_BLACK[2])],
                   [np.cos(x_BLACK[2]), -np.sin(x_BLACK[2])]])
    r_des = x_BLACK[:2] + R @ docking_offset[:2]
    th_des = wrap(x_BLACK[2] + docking_offset[2])
    v_des = x_BLACK[3:5] + x_BLACK[5] * dR @ docking_offset[:2]
    w_des = x_BLACK[5]
    x_des = np.concatenate([r_des, [th_des], v_des, [w_des]])
    e = x_RED - x_des
    e[2] = wrap(e[2])   # wrap attitude error (avoids 2*pi jump artifacts)
    LfV = 2 * e @ Q @ np.concatenate([x_RED[3:6], np.zeros(3)])
    LgV = 2 * e @ Q @ B
    dVdt = -2 * e @ Q @ np.concatenate([x_des[3:6], np.zeros(3)]) if tv == 1 else 0.0
    A = np.append(LgV, -1.0)
    b = -LfV - dVdt - k * (e @ Q @ e)
    return A[None, :], np.array([b]), e


# ------------------------ KOZ shrink (chart_495) ------------------------
def shrink_koz(x_RED, x_BLACK, r_KOZ_tar, gamma_step,
               r_min=R_KOZ_TAR_MIN, eta=ETA, zeta=ZETA,
               docking_offset=DOCKING_OFFSET, beta=BETA):
    R = rotz(x_BLACK[2])
    a, b = r_KOZ_tar
    r_dock = R @ ((a * b / np.sqrt((a * np.sin(beta))**2
                                   + (b * np.cos(beta))**2))
                  * np.array([np.cos(beta), np.sin(beta)]))
    x_des = np.array([x_BLACK[0] + r_dock[0], x_BLACK[1] + r_dock[1],
                      x_BLACK[2] + docking_offset[2]])
    x_err = (x_RED[:2] - x_des[:2]) @ (x_RED[:2] - x_des[:2])
    th_err = abs(wrap(x_RED[2] - x_des[2]))
    if x_err < eta**2 and th_err < zeta:
        r_KOZ_tar = gamma_step * r_KOZ_tar
    return np.maximum(r_KOZ_tar, r_min)


# --------------------------- QP (chart_555) -----------------------------
def cbf_qp(rows_A, rows_b, x0, u_max=U_MAX, p_clf=P_CLF):
    """min u'u + p_clf d^2  s.t.  A [u;d] <= b, |u|<=u_max, d>=0."""
    A = np.vstack(rows_A)
    b = np.concatenate(rows_b)
    Hd = np.array([1.0, 1.0, 1.0, p_clf])
    bnds = [(-u_max[0], u_max[0]), (-u_max[1], u_max[1]),
            (-u_max[2], u_max[2]), (0.0, None)]
    cons = [{"type": "ineq", "fun": lambda z, A=A, b=b: b - A @ z,
             "jac": lambda z, A=A: -A}]
    res = minimize(lambda z: Hd @ (z**2), x0,
                   jac=lambda z: 2 * Hd * z,
                   method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 50, "ftol": 1e-12})
    z = res.x
    ok = res.success and np.all(A @ z <= b + 1e-6)
    if not ok:
        # last-ditch: keep hard CBF rows, drop the CLF row (index 0 = dock)
        res2 = minimize(lambda z: Hd @ (z**2), np.zeros(4),
                        jac=lambda z: 2 * Hd * z, method="SLSQP",
                        bounds=bnds,
                        constraints=[{"type": "ineq",
                                      "fun": lambda z: b[1:] - A[1:] @ z,
                                      "jac": lambda z: -A[1:]}],
                        options={"maxiter": 50, "ftol": 1e-12})
        z = res2.x
        ok = res2.success
    return z[:3], z[3], ok


class CBFRVDController:
    """One-call-per-step wrapper, holds warm start and KOZ state."""

    def __init__(self, mRED, IRED, dt, los_mode="iccbf"):
        self.m, self.I = mRED, IRED
        self.gamma_step = 0.95 ** (dt / T_S)
        self.r_KOZ = R_KOZ_TAR_INI.copy()
        self.z_prev = np.zeros(4)
        self.los_mode = los_mode

    def control(self, x_RED, x_BLACK, x_BLUE, obstacle_active=True):
        self.r_KOZ = shrink_koz(x_RED, x_BLACK, self.r_KOZ, self.gamma_step)
        A_d, b_d, e = dock_row(x_RED, x_BLACK, self.m, self.I)
        A_k, b_k = koz_iccbf_row(x_RED, x_BLACK, self.m, self.r_KOZ)
        A_o, b_o = koz_obs_row(x_RED, x_BLUE, self.m, active=obstacle_active)
        if self.los_mode == "iccbf":
            A_l, b_l = los_iccbf_row(x_RED, x_BLACK, self.I)
        else:
            A_l, b_l = los_rows(x_RED, x_BLACK, self.I)
        A_v, b_v = vel_row(x_RED, self.m)
        u, delta, ok = cbf_qp(
            [A_d, A_k[None, :], A_o, A_l, A_v],
            [b_d, np.array([b_k]), b_o, b_l, b_v],
            self.z_prev)
        self.z_prev = np.append(u, delta)
        # diagnostics: raw barrier values
        Rq = rotz(x_BLACK[2])
        Qe = Rq @ np.diag(1 / self.r_KOZ**2) @ Rq.T
        d = x_RED[:2] - x_BLACK[:2]
        h_koz = d @ Qe @ d - 1
        do = x_RED[:2] - x_BLUE[:2]
        Ro = rotz(x_BLUE[2])
        Qo = Ro @ np.diag(1 / R_KOZ_OBS**2) @ Ro.T
        h_obs = do @ Qo @ do - 1
        bear = wrap(np.arctan2(x_BLACK[1] - x_RED[1],
                               x_BLACK[0] - x_RED[0]) - x_RED[2])
        h_los = SENSOR_FOV - abs(bear)
        h_vel = V_MAX - np.linalg.norm(x_RED[3:5])
        V = e @ Q_DOCK @ e
        return u, dict(delta=delta, ok=ok, r_KOZ=self.r_KOZ.copy(),
                       h_koz=h_koz, h_obs=h_obs, h_los=h_los,
                       h_vel=h_vel, V=V, e=e)
