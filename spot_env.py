"""
SPOT docking RL environment (gymnasium).

Scenario
--------
A chaser (RED) docks with a free-drifting target (BLACK) on a frictionless
air-bearing table while:
  - staying out of two moving elliptical keep-out zones (KOZ) -- one around the
    target (BLACK), one around an obstacle (BLUE),
  - keeping the target inside its camera field-of-view (LOS cone),
  - respecting velocity limits.

Control architecture
--------------------
Every step a CLF-CBF quadratic program (QP) turns the docking objective into a
safe force/torque command:
  - a Control Lyapunov Function (CLF) pulls the chaser toward the dock pose,
  - three Input-Constrained CBFs (ICCBFs) enforce the two KOZs and the LOS cone,
    each carrying an optimal-decay slack k>=0 that enters as (alpha+k)*h,
  - the velocity-limit CBFs are computed for the observation but (matching the
    MATLAB fcn()) are NOT part of the QP.

The RL agent chooses the ICCBF class-K gains and the CLF decay rate that make
this QP dock efficiently; `_decode` maps a [-1,1]^10 action to those gains.
Set `setconst=True` to pin the gains to fixed values instead (for testing /
MATLAB comparison).

All CBF/CLF rows below are closed-form Lie-derivative expressions, verified
against finite differences and the MATLAB build_iccbf_functions.m.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from qpsolvers import solve_qp

# ----------------------------- constants (from SPOT initializer) -----------
mRED, IRED = 12.137, 0.19816    # SPOT platform mass/inertia (MATLAB initializer)

U_MAX = 0.1*np.array([1.0, 1.0, 0.15])                 # [Fx Fy tau] limits
TORQUE_ARM = 0.07    # m, AVG SPOT thruster moment arm (~7 cm): torque is made by thrusters at this arm,
                     # so Ftotal = |F| + |tau|/arm (Ns-equiv) and propellant = integral of Ftotal dt.
TOF_CAP = 110.0      # s; SOFT time-of-flight cap. One-sided: docking up to here is free (gentle docking
                     # stays allowed, cuts FORCE like DE), only dawdling PAST it is penalized (reward+score).

FOV = np.deg2rad(40.0)
SENS_OFF = np.array([0.145-0.042, -0.0395])            # sensor_offset (planar)
SENS_NRM = np.array([1.0, 0.0])                        # sensor_normal
SENS_TGT = np.array([0.0825, 0.2516])                  # sensor_target (BLACK body)
DOCK_OFF = np.array([0.165, 0.427, -np.pi/2])          # docking_offset
R_KOZ_TAR_MIN = np.array([0.8, 0.42])
R_KOZ_TAR_INI = np.array([0.85, 0.85])
R_KOZ_OBS = np.array([0.43, 0.43])

# KOZ shrink (chart_495): proximity-gated contraction toward R_KOZ_TAR_MIN,
# triggered when the chaser HOLDS the dock point, NOT by elapsed time.
ETA  = np.sqrt(0.5)                                    # dock-hold position tol [m]
ZETA = np.deg2rad(10.0)                                # dock-hold attitude tol [rad]
T_S  = 4.0                                             # shrink time constant [s]
BETA = np.pi/2 - np.arctan2(DOCK_OFF[0], DOCK_OFF[1])  # dock direction on ellipse
V_MAX, AV_MAX = 0.1, np.deg2rad(8.0)

# docking CLF weight (sliding form, Run_Initializer.m): V=6I, lambda_dock=1
#   Q_dock = [[lambda^2 I, lambda V],[lambda V, V^2]]  (= [[I, 6I],[6I, 36I]])

_V_DOCK, _LAM_DOCK = 6*np.eye(3), 1.0
Q_DOCK = np.block([[_LAM_DOCK**2*np.eye(3), _LAM_DOCK*_V_DOCK],
                   [_LAM_DOCK*_V_DOCK,      _V_DOCK@_V_DOCK]])
K_DOCK = 5.0
K_VEL = 0.075
P_CLF = 10.0          # CLF slack penalty (p_clf in fcn())
P_CBF = 1e5           # CBF optimal-decay slack penalty (k weight in H, MATLAB)
XL, YL = 3.5, 2.4     # table extent (xLength, yLength)
DT = 0.05             # control/sim step [s] = 20 Hz (Run_Initializer baseRate)
T_MAX = 45.0
EPS = 1e-9
D2R = np.pi/180

# Deterministic scenarios from Run_Initializer.m: (x_RED_0, x_BLACK_0, x_BLUE_0).
# Each state is [x, y, theta, dx, dy, dtheta]; BLACK/BLUE drift force-free.
TEST_CASES = {
    0: (np.array([3.0, 2.0, 0.0, 0,0,0.0]),
        np.array([0.5, 0.5, 315*D2R, 0.01, 0.01, 1*D2R]),
        np.array([1.5, 1.25, 0.0, 0.015, 0.0075, 0.0])),
    1: (np.array([3.2, 1.8, 0.0, 0,0,0.0]),
        np.array([2.0, 2.2, 270*D2R, -0.005, -0.005, -2.5*D2R]),
        np.array([2.5, 1.25, 0.0, -0.01, -0.01, 0.0])),
    2: (np.array([1.5, 2.0, np.pi, 0,0,0.0]),
        np.array([2.2, 0.2, 270*D2R, -0.005, 0.005, 1.5*D2R]),
        np.array([2.0, 1.2, 0.0, -0.015, 0.0, 0.0])),
    
    3: (np.array([3.0, 2.0, 0.0, 0,0,0.0]),
        np.array([XL/2, YL/2, 90*D2R, 0,0,0.0]),
        np.array([0.0, 0.0, 0.0, 0,0,0.0])),
    4: (np.array([3.0, 2.0, 0.0, 0,0,0.0]),
        np.array([0.5, 0.5, 315*D2R, 0,0,0.0]),
        np.array([1.5, 1.25, 0.0, 0.015, 0.0075, 0.0])),
}
N_CASES = 3        # how many of the named TEST_CASES to CONSIDER (train-sample + eval over the
                   # first N_CASES only). All 5 stay defined; set back to len(TEST_CASES) for all.

# --- meta-RL domain randomization (hidden task params, sampled each reset) ---
# Dynamics (mRED, IRED, U_MAX) are held fixed at the values verified from
# MPC-data1.csv; only the task geometry varies, so the RNN must infer it.
DR_KOZ_SCALE   = (0.90, 1.10)     # target/obstacle keep-out ellipse size factor
DR_FOV_SCALE   = (0.85, 1.15)     # camera half-angle factor
DR_SENS_SCALE  = (0.90, 1.10)     # sensor offset / target-point factor
DR_SPIN_RANGE  = (np.deg2rad(0.5), np.deg2rad(2.5))   # BLACK initial spin rate
DR_POSE_JITTER = 0.15             # +/- m on initial chaser/target/obstacle pose
DR_ATT_JITTER  = np.deg2rad(20.0) # +/- rad on initial/goal attitudes

# random-pose (pose_random) sampling -- WIDENED to COVER the 5 named eval cases so
# they are in-distribution (case 1 spins 2.5 deg/s; cases 0/4 start ~2.9 m out, both
# outside the old random ranges, which is why pose_random eval'd badly on them).
POSE_SPIN_MAX  = np.deg2rad(3.0)  # max |target spin| for random poses (> case 1's 2.5)
POSE_DRIFT_MAX = 0.015            # max linear drift on target/obstacle (matches named cases)
POSE_FAR_FRAC  = 0.5              # fraction of layouts forced to a long (>2.2 m) approach
POSE_MIX_CASES = 0.30             # fraction of pose_random resets that use a NAMED eval case


def sample_pose_config(rng, margin=0.45):
    """Random VALID initial poses for the 3 bodies (positions+headings only;
    KOZ/FOV geometry unchanged). Chaser is placed clear of both keep-out zones,
    all bodies on the table; target/obstacle get small constant drift.

    Ranges WIDENED to cover the named eval cases: target spin up to POSE_SPIN_MAX
    (>2.5 deg/s) and -- on half the layouts -- a long (>2.2 m) chaser approach, so
    case-1-like spins and case-0/4-like distances are practiced, not out-of-dist."""
    def rand_xy():
        return np.array([rng.uniform(margin, XL-margin), rng.uniform(margin, YL-margin)])
    far = rng.random() < POSE_FAR_FRAC            # half: force a long approach (covers ~2.9 m)
    for _ in range(400):
        pB, pU, pR = rand_xy(), rand_xy(), rand_xy()
        d_RB = np.linalg.norm(pR-pB)
        if d_RB < R_KOZ_TAR_INI[0]+0.15:                   continue
        if far and d_RB < 2.2:                             continue   # bias to long approaches
        if np.linalg.norm(pR-pU) < R_KOZ_OBS[0]+0.15:      continue
        if np.linalg.norm(pB-pU) < R_KOZ_TAR_INI[0]+R_KOZ_OBS[0]+0.1: continue
        xR0 = np.array([pR[0], pR[1], rng.uniform(-np.pi, np.pi), 0, 0, 0.0])
        xB0 = np.array([pB[0], pB[1], rng.uniform(-np.pi, np.pi),
                        *rng.uniform(-POSE_DRIFT_MAX, POSE_DRIFT_MAX, 2),
                        rng.uniform(-POSE_SPIN_MAX, POSE_SPIN_MAX)])
        xU0 = np.array([pU[0], pU[1], rng.uniform(-np.pi, np.pi),
                        *rng.uniform(-POSE_DRIFT_MAX, POSE_DRIFT_MAX, 2), 0.0])
        return xR0, xB0, xU0
    return [a.copy() for a in TEST_CASES[4]]      # fallback (very unlikely)


# ----------------------------- planar-rotation helpers ---------------------
# Rmat and its 1st-3rd derivatives w.r.t. the angle (used to differentiate the
# rotating keep-out ellipse / sensor frame symbolically-by-hand).
def Rmat(t):   return np.array([[ np.cos(t), -np.sin(t)], [ np.sin(t),  np.cos(t)]])
def dRmat(t):  return np.array([[-np.sin(t), -np.cos(t)], [ np.cos(t), -np.sin(t)]])
def ddRmat(t): return np.array([[-np.cos(t),  np.sin(t)], [-np.sin(t), -np.cos(t)]])
def dddRmat(t):return np.array([[ np.sin(t),  np.cos(t)], [-np.cos(t),  np.sin(t)]])
def wrap(a):   return np.arctan2(np.sin(a), np.cos(a))


def propellant(u):
    """Real propellant rate (Ns-equivalent): translational |F| + rotational |tau|/arm.
    Torque counted at its true thruster-impulse cost -- the metric the RL should minimize."""
    return float(np.hypot(u[0], u[1]) + abs(u[2]) / TORQUE_ARM)


# ----------------------------- CBF / CLF rows (verified) -------------------
# Each ICCBF row is the relative-degree-2 cascade (N=2):
#     b1 = h_dot  + a0 * h
#     b2 = h_ddot + (a0+a1) h_dot + a0 a1 h  -  W      (W = input-constraint margin)
#     QP row:   -Lg(b2) . u  <=  Lf(b2) + a2 * b2
# and returns (A over [Fx,Fy,tau], b, h).

def koz_row(xR, xB, rkoz, a0, a1, a2, m):
    """Elliptical keep-out ICCBF on the translational channel.

    Barrier h = d' Q d - 1 >= 0 keeps the chaser outside the rotating keep-out
    ellipse around platform xB, where d = chaser-platform offset and Q is the
    inverse-ellipse metric in the platform frame (semi-axes rkoz).
    """
    # inverse-ellipse metric Q and its time derivatives (platform spin thd)
    S = np.diag(1.0/rkoz**2)
    th, thd = xB[2], xB[5]
    Rt = Rmat(th); Rtd = thd*dRmat(th); Rtdd = thd**2*ddRmat(th); Rtddd = thd**3*dddRmat(th)
    Q   = Rt@S@Rt.T
    Qd  = Rt@S@Rtd.T + Rtd@S@Rt.T
    Qdd = Rtdd@S@Rt.T + 2*Rtd@S@Rtd.T + Rt@S@Rtdd.T
    Qddd= Rtddd@S@Rt.T + 3*Rtdd@S@Rtd.T + 3*Rtd@S@Rtdd.T + Rt@S@Rtddd.T

    # barrier h and its derivatives along the relative motion (d, dd)
    d = xR[0:2]-xB[0:2]; dd = xR[3:5]-xB[3:5]
    h    = d@Q@d - 1
    hd   = 2*(d@Q@dd) + d@Qd@d
    hdd  = 2*(dd@Q@dd) + 4*(d@Qd@dd) + d@Qdd@d
    hddd = 6*(dd@Qd@dd) + 6*(d@Qdd@dd) + d@Qddd@d

    # input-constraint margin W (smoothed worst-case force authority) and its rate
    w = (2/m)*(d@Q); sj = np.sqrt(w**2+EPS**2); W = sj@(U_MAX[0:2])
    wd = (2/m)*(dd@Q+d@Qd); Wdot = np.sum(w*wd/sj*U_MAX[0:2])

    # ICCBF QP row  A u <= b
    b2 = hdd + (a0+a1)*hd + a0*a1*h - W
    Lf = hddd + (a0+a1)*hdd + a0*a1*hd - Wdot
    Lg = (1/m)*(4*(dd@Q) + 4*(d@Qd)) + (a0+a1)*(2/m)*(d@Q)   # 1x2 over [Fx,Fy]
    A = np.array([-Lg[0], -Lg[1], 0.0]); b = Lf + a2*b2
    return A, b, h


def los_row(xR, xB, a0, a1, a2, I, tv,
            fov=FOV, sens_off=SENS_OFF, sens_tgt=SENS_TGT, sens_nrm=SENS_NRM):
    """Field-of-view (line-of-sight) ICCBF on the torque channel.

    Barrier h = (r.e)^2 - cos^2(FOV) (r.r) >= 0 keeps the target feature point
    inside the camera cone, where r is the sensor->target vector and e the
    boresight, both in world frame. Sensor geometry (fov, offset, target point,
    normal) is overridable so the env can randomize it as a hidden meta-param.
    """
    thc, thcd = xR[2], xR[5]; tht, thtd = xB[2], xB[5]
    Rc = Rmat(thc); dRc = dRmat(thc); ddRc = ddRmat(thc); Rt = Rmat(tht); dRt = dRmat(tht)

    # LOS vector r (sensor->target point) and boresight e, plus angle derivatives
    r = xB[0:2] + Rt@sens_tgt - xR[0:2] - Rc@sens_off
    e = Rc@sens_nrm
    drdth = -dRc@sens_off; ddrddth = -ddRc@sens_off
    dedth =  dRc@sens_nrm; ddeddth =  ddRc@sens_nrm
    rdot = tv*(xB[3:5] + thtd*dRt@sens_tgt - xR[3:5])

    # time derivatives of r, e along the chaser rotation (thcd) + target drift
    rd = thcd*drdth + rdot; ed = thcd*dedth
    rdd = thcd**2*ddrddth;  edd = thcd**2*ddeddth
    rddd = -thcd**3*drdth;  eddd = -thcd**3*dedth

    # barrier h and derivatives;  'a' = r.e (the projection onto boresight)
    cF = np.cos(fov)**2
    a = r@e; ad = rd@e + r@ed; add = rdd@e + 2*(rd@ed) + r@edd
    addd = rddd@e + 3*(rdd@ed) + 3*(rd@edd) + r@eddd
    h    = a**2 - cF*(r@r)
    hd   = 2*a*ad - 2*cF*(r@rd)
    hdd  = 2*(ad**2 + a*add) - 2*cF*(rd@rd + r@rdd)
    hddd = 6*ad*add + 2*a*addd - 6*cF*(rd@rdd) - 2*cF*(r@rddd)

    # explicit partials w.r.t. chaser attitude (the only actuated d.o.f. here)
    dhdth = 2*a*(drdth@e + r@dedth) - 2*cF*(r@drdth)
    ddhddth = 2*((drdth@e + r@dedth)**2 + a*(ddrddth@e + 2*(drdth@dedth) + r@ddeddth)
                 - (drdth@drdth + r@ddrddth)*cF)
    M_xt = 2*((rdot@e)*(drdth@e + r@dedth) + a*(rdot@dedth) - cF*(rdot@drdth))

    # input-constraint margin W (torque authority) and its rate
    wT = dhdth/I; sT = np.sqrt(wT**2+EPS**2); W = sT*U_MAX[2]
    Wdot = (wT*((ddhddth*thcd + M_xt)/I)/sT)*U_MAX[2]

    # ICCBF QP row  A u <= b   (torque channel only)
    b2 = hdd + (a0+a1)*hd + a0*a1*h - W
    Lf = hddd + (a0+a1)*hdd + a0*a1*hd - Wdot
    Lg = (1/I)*(2*ddhddth*thcd + 2*M_xt + (a0+a1)*dhdth)
    A = np.array([0.0, 0.0, -Lg]); b = Lf + a2*b2
    return A, b, h


def clf_rows(xR, xB, tv, k_dock=K_DOCK):
    """Docking CLF row:  LgV . u - delta <= rhs   (delta = CLF slack).

    V = e' Q_DOCK e is the sliding-mode docking error to the moving dock pose
    (BLACK pose + DOCK_OFF). k_dock is the CLF decay rate (agent-modulated).
    Returns (LgV, rhs, V, e).
    """
    th = xB[2]; thd = xB[5]; R = Rmat(th); dR = dRmat(th)
    r_des = xB[0:2] + R@DOCK_OFF[0:2]; th_des = wrap(th + DOCK_OFF[2])
    v_des = xB[3:5] + thd*dR@DOCK_OFF[0:2]; om_des = thd
    e = np.array([xR[0]-r_des[0], xR[1]-r_des[1], wrap(xR[2]-th_des),
                  xR[3]-v_des[0], xR[4]-v_des[1], xR[5]-om_des])
    B = np.zeros((6,3)); B[3,0] = 1/mRED; B[4,1] = 1/mRED; B[5,2] = 1/IRED
    LfV = 2*e@Q_DOCK@np.array([xR[3],xR[4],xR[5],0,0,0])
    LgV = 2*e@Q_DOCK@B
    dVdt = tv*(-2*e@Q_DOCK@np.array([v_des[0],v_des[1],om_des,0,0,0]))
    V = e@Q_DOCK@e
    rhs = -LfV - dVdt - k_dock*V
    return LgV, rhs, V, e


def vel_rows(xR):
    """First-order velocity-limit CBFs -> (A_lin,b_lin,h_lin),(A_ang,b_ang,h_ang).

    Computed for the observation only; NOT added to the QP (matching fcn()).
    """
    vx, vy, om = xR[3], xR[4], xR[5]
    h_lin = V_MAX**2 - vx**2 - vy**2
    A_lin = np.array([2*vx/mRED, 2*vy/mRED, 0.0]); b_lin = K_VEL*h_lin
    h_ang = AV_MAX**2 - om**2
    A_ang = np.array([0.0, 0.0, 2*om/IRED]); b_ang = K_VEL*h_ang
    return (A_lin, b_lin, h_lin), (A_ang, b_ang, h_ang)


def shrink_koz(xR, xB, rkoz, rmin, gamma_step, eta=ETA, zeta=ZETA,
               dock_off=DOCK_OFF, beta=BETA):
    """Distance-gated KOZ shrink (chart_495 / cbf_rvd.shrink_koz).

    Contract the target keep-out ellipse toward rmin by gamma_step, but ONLY
    while the chaser is holding the dock point on the ellipse boundary within
    eta (position) and zeta (attitude). No dependence on elapsed time.
    """
    a, b = rkoz
    r_dock = Rmat(xB[2]) @ ((a*b/np.sqrt((a*np.sin(beta))**2 + (b*np.cos(beta))**2))
                            * np.array([np.cos(beta), np.sin(beta)]))
    x_des = np.array([xB[0]+r_dock[0], xB[1]+r_dock[1], xB[2]+dock_off[2]])
    pos_err2 = (xR[:2]-x_des[:2]) @ (xR[:2]-x_des[:2])
    th_err = abs(wrap(xR[2]-x_des[2]))
    if pos_err2 < eta**2 and th_err < zeta:
        rkoz = gamma_step*rkoz
    return np.maximum(rkoz, rmin)


def di_step(x, u, m, I, dt):
    """Exact double-integrator (air-bearing) step for one rigid body."""
    a = np.array([u[0]/m, u[1]/m, u[2]/I])
    xn = x.copy()
    xn[0:3] = x[0:3] + x[3:6]*dt + 0.5*a*dt*dt
    xn[3:6] = x[3:6] + a*dt
    xn[2] = wrap(xn[2])
    return xn


# ----------------------------- environment ---------------------------------
class SpotDockEnv(gym.Env):
    """Gymnasium env: agent picks ICCBF class-K gains, QP returns safe thrust."""

    # Action map (mirrors the MetaRL docking RLCBF.step): per constraint the
    # agent emits two cascade coefs (a0,a1) and a barrier-decay slack (a2),
    # plus one global CLF decay multiplier (Lslack). See _decode.
    ACOEF_HI = 3.0                        # ICCBF cascade coefs a0,a1 in [0, ACOEF_HI]
    HSLACK_LO, HSLACK_HI = 0.01, 1.0      # final-row barrier decay a2 (repo hslack)
    LMUL_LO,  LMUL_HI    = 0.2, 1.2       # CLF decay multiplier (repo Lslack) on K_DOCK

    # residual ("constant +/- a bit") gain mode: gains = base + band*action (then clipped
    # to valid range). action=0 -> exactly the constant baseline, so the policy can only
    # fine-tune around it -- no wandering, no jitter, floor = the 5/5 constant gains.
    RESID_BASE  = (1.0, 0.5, 0.25)        # const baseline on every CBF (a0,a1,a2)
    RESID_BAND  = (2.0, 2.0, 0.5)         # +/- band on (a0,a1,a2) -- wide enough to CONTAIN
    RESID_KBAND = 4.0                     # k_dock = K_DOCK +/- this  the DE-optimal gains

    def __init__(self, randomize=True, setconst=False,
                 const_gains=(1.0, 0.5, 0.25), const_kdock=K_DOCK, test_case=2,
                 ic=None, pose_random=False, case_random=False, t_max=T_MAX,
                 hidden_obs=False, rl2=False, residual_gains=False):
        super().__init__()
        self.residual_gains = bool(residual_gains)   # decode action as base +/- band (vs absolute)
        self.ic = ic    # optional custom (xR0,xB0,xU0); overrides TEST_CASES when set
        self.pose_random = pose_random   # resample body poses each reset (geometry fixed)
        self.case_random = case_random   # sample a random named TEST_CASE each reset
        self.t_max = t_max  # episode truncation horizon [s] (docking can take ~100 s)
        # randomize=False -> exact Run_Initializer.m ICs + geometry (no domain rand)
        # setconst=True   -> ignore the policy action, use const_gains every step
        #   const_gains is (3,)=shared across CBFs or (3,3)=per-constraint (a0,a1,a2)
        self.randomize = randomize
        self.setconst = setconst
        self.const_gains = np.asarray(const_gains, dtype=float)
        self.const_kdock = float(const_kdock)
        self.test_case = test_case        # which Run_Initializer.m scenario (0-4)
        # action: [a0,a1,a2] x {tar-KOZ, obs-KOZ, LOS} (9) + CLF Lslack (1) = 10
        self.action_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)
        # POMDP / recurrent meta-RL mode: hide target & obstacle -- the policy sees
        # ONLY the chaser's own state (7-D) and must INFER the hidden geometry over
        # time via an LSTM. rl2=True also feeds the previous action+reward (11-D, the
        # RL^2 recipe) so the recurrent net can identify the task from the reward.
        self.hidden_obs = bool(hidden_obs)
        self.rl2 = bool(rl2)
        obs_dim = 7 + (11 if rl2 else 0) if hidden_obs else 28
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.dt = DT
        self.gamma_step = 0.95**(DT/T_S)  # per-step KOZ shrink factor while held

    def _decode(self, action):
        """Map action in [-1,1]^10 -> (g: 3x3 class-K gains, k_dock: CLF decay).

        Each constraint i gets [a0, a1] cascade coefs (rescaled to [0,ACOEF_HI])
        and a barrier-decay a2 (in [HSLACK_LO,HSLACK_HI]); action[9] scales K_DOCK.
        """
        a = np.clip(action, -1, 1)
        if self.residual_gains:
            # gains = constant base +/- band*action (a in [-1,1]); a=0 -> exact baseline.
            grid = a[0:9].reshape(3, 3)                                # rows=CBFs, cols=(a0,a1,a2)
            g = np.asarray(self.RESID_BASE)[None, :] + np.asarray(self.RESID_BAND)[None, :] * grid
            g[:, 0:2] = np.clip(g[:, 0:2], 0.0, self.ACOEF_HI)
            g[:, 2]   = np.clip(g[:, 2],   self.HSLACK_LO, self.HSLACK_HI)
            k_dock = float(np.clip(K_DOCK + self.RESID_KBAND * a[9], 0.1, None))
            return g, k_dock
        g = np.zeros((3, 3))
        for i in range(3):
            g[i, 0] = 0.5*(a[3*i]   + 1.0)*self.ACOEF_HI                                  # a0
            g[i, 1] = 0.5*(a[3*i+1] + 1.0)*self.ACOEF_HI                                  # a1
            g[i, 2] = self.HSLACK_LO + 0.5*(a[3*i+2]+1.0)*(self.HSLACK_HI-self.HSLACK_LO) # a2
        lmul = self.LMUL_LO + 0.5*(a[9]+1.0)*(self.LMUL_HI-self.LMUL_LO)
        return g, K_DOCK*lmul

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.t = 0.0

        if self.randomize:
            # --- hidden meta-params: KOZ / sensor geometry (not in observation) ---
            self.r_koz_tar_ini = R_KOZ_TAR_INI*rng.uniform(*DR_KOZ_SCALE, size=2)
            self.r_koz_tar_min = np.minimum(R_KOZ_TAR_MIN*rng.uniform(*DR_KOZ_SCALE, size=2),
                                            self.r_koz_tar_ini)
            self.r_koz_obs = R_KOZ_OBS*rng.uniform(*DR_KOZ_SCALE, size=2)
            self.fov       = FOV*rng.uniform(*DR_FOV_SCALE)
            self.sens_off  = SENS_OFF*rng.uniform(*DR_SENS_SCALE)
            self.sens_tgt  = SENS_TGT*rng.uniform(*DR_SENS_SCALE)

            # --- target spin & approach geometry (task variation) ---
            pj = DR_POSE_JITTER; aj = DR_ATT_JITTER
            self.xR = np.array([1.5+rng.uniform(-pj, pj), 2.0+rng.uniform(-pj, pj), np.pi, 0, 0, 0.0])
            spin = rng.uniform(*DR_SPIN_RANGE)
            self.xB = np.array([2.2+rng.uniform(-pj, pj), 0.2+rng.uniform(-pj, pj),
                                wrap(np.deg2rad(270)+rng.uniform(-aj, aj)),
                                -0.005, 0.005, spin])
            self.xU = np.array([2.0+rng.uniform(-pj, pj), 1.2+rng.uniform(-pj, pj), 0.0, -0.015, 0.0, 0.0])
        else:
            # --- exact Run_Initializer.m scenario: nominal geometry, literal ICs ---
            self.r_koz_tar_ini = R_KOZ_TAR_INI.copy()
            self.r_koz_tar_min = R_KOZ_TAR_MIN.copy()
            self.r_koz_obs = R_KOZ_OBS.copy()
            self.fov = FOV
            self.sens_off = SENS_OFF.copy()
            self.sens_tgt = SENS_TGT.copy()
            if self.pose_random:                 # random poses, MIXED with the named eval cases
                if rng.random() < POSE_MIX_CASES:   # so the considered eval cases stay in-distribution
                    xR0, xB0, xU0 = TEST_CASES[int(rng.integers(0, N_CASES))]
                else:
                    xR0, xB0, xU0 = sample_pose_config(rng)
            elif self.case_random:               # random one of the first N_CASES named cases
                xR0, xB0, xU0 = TEST_CASES[int(rng.integers(0, N_CASES))]
            else:
                xR0, xB0, xU0 = self.ic if self.ic is not None else TEST_CASES[self.test_case]
            self.xR, self.xB, self.xU = xR0.copy(), xB0.copy(), xU0.copy()

        # orient chaser camera toward BLACK at t=0 -- Run_Initializer.m applies
        # rotateToFace to EVERY scenario:
        #   x_RED(3) = wrap( rotateToFace(theta, BLACK - (RED + R*sensor_offset))
        #                    + docking_offset(3) + pi/2 )
        rvec = self.xB[0:2] - (self.xR[0:2] + Rmat(self.xR[2])@self.sens_off)
        th0 = self.xR[2] + wrap(np.arctan2(rvec[1], rvec[0]) - self.xR[2])
        self.xR[2] = wrap(th0 + DOCK_OFF[2] + np.pi/2)

        self.rkoz_tar = self.r_koz_tar_ini.copy()
        _, _, self.Vprev, e0 = clf_rows(self.xR, self.xB, 1)
        # potential-based shaping trackers (telescoping baselines, re-seeded per episode)
        self._koz_prev  = float(np.sum(self.rkoz_tar - self.r_koz_tar_min))
        self._phi_prev  = -(np.linalg.norm(e0[0:2]) + 0.5*abs(e0[2])
                            + 2.0*np.linalg.norm(e0[3:5]) + 1.0*abs(e0[5]))
        self._prev_action = np.zeros(10, dtype=np.float32)   # RL^2 history (hidden_obs mode)
        self._prev_reward = 0.0
        return self._obs(), {}

    def _obs(self):
        """Observation. Full 28-D by default; chaser-ONLY (POMDP) if hidden_obs --
        the policy then sees none of the target/obstacle geometry and an LSTM must
        infer it from the chaser-state history (and prev action+reward if rl2)."""
        if self.hidden_obs:
            xR = self.xR
            o = [xR[0], xR[1], np.cos(xR[2]), np.sin(xR[2]), xR[3], xR[4], xR[5]]
            if self.rl2:                  # RL^2: last gains chosen + last reward seen
                o += list(np.asarray(self._prev_action, float).ravel()[:10])
                o += [float(self._prev_reward)]
            return np.asarray(o, dtype=np.float32)
        xR, xB, xU = self.xR, self.xB, self.xU
        _, _, V, e = clf_rows(xR, xB, 1)
        # barrier values only (dummy gains a0=a1=a2=1; the rows' A/b are unused here)
        _, _, h_tar = koz_row(xR, xB, self.rkoz_tar, 1, 1, 1, mRED)
        _, _, h_obs = koz_row(xR, xU, self.r_koz_obs, 1, 1, 1, mRED)
        _, _, h_los = los_row(xR, xB, 1, 1, 1, IRED, 1,
                              fov=self.fov, sens_off=self.sens_off, sens_tgt=self.sens_tgt)
        (_, _, h_vl), (_, _, h_va) = vel_rows(xR)
        o = np.concatenate([
            e,                                                # docking error (6)
            [np.cos(xR[2]), np.sin(xR[2]), xR[3], xR[4], xR[5]],   # chaser pose/vel (5)
            xR[0:2]-xB[0:2], xR[3:5]-xB[3:5], [xB[5]],         # rel. target (5)
            xR[0:2]-xU[0:2], xR[3:5]-xU[3:5], [xU[5]],         # rel. obstacle (5)
            [h_tar, h_obs, h_los, h_vl, h_va],                # constraint margins (5)
            [V, self.rkoz_tar[0]-R_KOZ_TAR_MIN[0]],           # CLF value, KOZ shrink (2)
        ])
        return o.astype(np.float32)

    # QP decision-vector layout:  z = [Fx Fy tau | delta | k_tar k_obs k_los]
    _DELTA, _K_TAR, _K_OBS, _K_LOS = 3, 4, 5, 6

    def _solve_qp(self, gains, k_dock):
        """Solve the CLF-CBF QP (matches the MATLAB fcn()) for the safe command.

        4 rows: dock(CLF) + KOZ_tar + KOZ_obs + LOS, each `row . z <= b`.
        Cost 0.5 z'Pz with P = 2*diag([1,1,1, p_clf, P_CBF,P_CBF,P_CBF]).
        CLF uses an additive slack delta (coef -1); each CBF an optimal-decay
        slack k>=0 entering as (alpha+k)*h, which in `row.z <= b` form has slack
        coefficient -h (so the relaxation k*h vanishes at the boundary h=0).
        Returns (u, delta, slacks_k, (h_tar,h_obs,h_los,h_vl,h_va), V).
        """
        xR, xB, xU = self.xR, self.xB, self.xU

        # constraint rows (each `gains` row is (a0,a1,a2) for that CBF)
        A_tar, b_tar, h_tar = koz_row(xR, xB, self.rkoz_tar, *gains[0], mRED)
        A_obs, b_obs, h_obs = koz_row(xR, xU, self.r_koz_obs, *gains[1], mRED)
        A_los, b_los, h_los = los_row(xR, xB, *gains[2], IRED, 1,
                                      fov=self.fov, sens_off=self.sens_off,
                                      sens_tgt=self.sens_tgt)
        LgV, rhs, V, e = clf_rows(xR, xB, 1, k_dock=k_dock)
        (_, _, h_vl), (_, _, h_va) = vel_rows(xR)   # obs/diagnostics only (not in QP)

        n = 7
        P = 2.0*np.diag([1.0, 1.0, 1.0, P_CLF, P_CBF, P_CBF, P_CBF])
        q = np.zeros(n)

        # each spec: (A over [Fx,Fy,tau], slack column, slack coef, rhs)
        specs = [
            (LgV,   self._DELTA, -1.0,    rhs),     # dock CLF  (additive slack delta)
            (A_tar, self._K_TAR, -h_tar,  b_tar),   # target KOZ   (alpha+k)*h slack
            (A_obs, self._K_OBS, -h_obs,  b_obs),   # obstacle KOZ
            (A_los, self._K_LOS, -h_los,  b_los),   # LOS cone
        ]
        G = np.zeros((len(specs), n)); h = np.zeros(len(specs))
        for i, (A_u, col, coef, b) in enumerate(specs):
            G[i, 0:3] = A_u; G[i, col] = coef; h[i] = b

        # |u| <= U_MAX inside the QP ; slacks (delta, k_*) >= 0, unbounded above
        INF = 1e6
        lb = np.array([-U_MAX[0], -U_MAX[1], -U_MAX[2], 0, 0, 0, 0])
        ub = np.array([ U_MAX[0],  U_MAX[1],  U_MAX[2], INF, INF, INF, INF])
        z = solve_qp(P, q, G, h, lb=lb, ub=ub, solver='quadprog')

        if z is None:                       # infeasible: coast (no thrust), flag slacks
            z = np.zeros(n); z[self._K_TAR:self._K_LOS+1] = 1e3
        u = z[0:3]; delta = z[self._DELTA]; slacks = z[self._K_TAR:self._K_LOS+1]
        return u, delta, slacks, (h_tar, h_obs, h_los, h_vl, h_va), V

    def step(self, action):
        # decode the agent's gains (or use the fixed ones in setconst mode)
        gains, k_dock = self._decode(action)
        if self.setconst:
            cg = self.const_gains                          # (3,)=shared or (3,3)=per-CBF
            gains = np.tile(cg, (3, 1)) if cg.ndim == 1 else cg
            k_dock = self.const_kdock

        u, delta, s, hvals, V = self._solve_qp(gains, k_dock)
        h_tar, h_obs, h_los, h_vl, h_va = hvals

        # propagate: chaser under the safe command; BLACK/BLUE drift force-free
        # (constant velocity, matching the SPOT testbed / MPC-data1.csv).
        self.xR = di_step(self.xR, u, mRED, IRED, self.dt)
        self.xB = di_step(self.xB, np.zeros(3), mRED, IRED, self.dt)
        self.xU = di_step(self.xU, np.zeros(3), mRED, IRED, self.dt)
        # shrink target KOZ toward min ONLY while the chaser holds the dock point
        self.rkoz_tar = shrink_koz(self.xR, self.xB, self.rkoz_tar,
                                   self.r_koz_tar_min, self.gamma_step)
        self.t += self.dt
        _, _, Vn, e = clf_rows(self.xR, self.xB, 1)

        # ---- reward (all dense terms potential-based => non-gameable) ----
        # Shaping constants (bounded; see test plan for DE-vs-random separation).
        S_CAP  = 5.0     # clip slack fed to reward: kills the 1e3 QP-infeasible -300 cliff
        W_KOZ  = 80.0    # KOZ-shrink potential weight (pays the load-bearing ~55s dock-hold)
        W_POSE = 8.0     # pose-error potential weight (funnels the last cm / att / vel)
        C_MARG = 5.0     # graded barrier-margin penalty (brake BEFORE the crash cliff)
        H_MARG = 0.10    # margin band where the barrier ramps in (h in (0, H_MARG))

        # Potentials at the CURRENT (post-step) state.
        # phi_koz: remaining shrink the chaser still owes; shrink_koz() only lowers
        # rkoz_tar while the dock point is HELD, so this term pays exactly for the
        # hold. Monotone non-increasing, floored at 0 -> non-farmable.
        phi_koz  = float(np.sum(self.rkoz_tar - self.r_koz_tar_min))
        # phi_pose: raw docking-error potential; stays informative after the CLF V
        # saturates near the goal (V is quadratically flat, ||e|| stays linear),
        # maximized (=0) only at the dock pose. Ng PBRS => telescopes, non-gameable.
        phi_pose = -(np.linalg.norm(e[0:2]) + 0.5*abs(e[2])
                     + 2.0*np.linalg.norm(e[3:5]) + 1.0*abs(e[5]))

        r  = 10.0*(self.Vprev - Vn)                       # CLF docking progress (PSD V, non-gameable)
        r += W_KOZ*(self._koz_prev - phi_koz)             # KOZ-shrink potential (>=0, telescopes to W_KOZ*span)
        r += W_POSE*(phi_pose - self._phi_prev)           # pose-error potential (tightens last cm / att / vel)
        r -= 0.1*min(float(np.sum(s)), S_CAP)             # slack k>=0 usage, CLIPPED (no -300 infeasibility cliff)
        r -= 5.0*propellant(u)*self.dt                    # propellant pressure (TORQUE-WEIGHTED: real
                                                          #   fuel = |F| + |tau|/arm, not force-only ||u||)
        # time penalty is ONE-SIDED now: free up to TOF_CAP (gentle docking still cuts FORCE like DE),
        # but dawdling past the cap bleeds reward -> dock by TOF_CAP instead of drifting to ~130 s.
        if self.t > TOF_CAP:
            r -= 2.0*self.dt                              # 2 reward/s past the cap (one-sided TOF pressure)
        r -= 0.5*max(0.0, -h_los)                         # gentle FOV nudge (CBF already enforces it)
        r -= C_MARG*(max(0.0, H_MARG - h_tar)**2          # graded barrier: brake before the KOZ boundary
                     + max(0.0, H_MARG - h_obs)**2)       #   (exactly 0 when h>H_MARG -> non-farmable)

        self.Vprev = Vn
        self._koz_prev = phi_koz
        self._phi_prev = phi_pose

        # ---- termination ----
        term = False; trunc = False
        pos_ok = np.linalg.norm(e[0:2]) < 0.08 and abs(e[2]) < np.deg2rad(5)   # 8 cm capture tol
        vel_ok = np.linalg.norm(e[3:5]) < 0.02 and abs(e[5]) < np.deg2rad(2)
        if pos_ok and vel_ok:
            r += 100.0; term = True                # docked (restored: lowering this removed the
                                                   #   docking anchor -> policy abandoned docking)
        if h_tar < 0 or h_obs < 0:
            r -= 100.0; term = True                # collision (entered a KOZ)
        oob = not (0 < self.xR[0] < XL and 0 < self.xR[1] < YL)
        if oob:
            r -= 50.0; term = True
        if self.t >= self.t_max:
            trunc = True

        # RL^2 history for the hidden-obs LSTM: the obs returned now carries the
        # action just taken and the reward just received (set BEFORE _obs()).
        self._prev_action = np.asarray(action, dtype=np.float32).ravel()
        self._prev_reward = float(r)
        info = {'h_tar': h_tar, 'h_obs': h_obs, 'h_los': h_los, 'V': Vn,
                'docked': pos_ok and vel_ok, 'slack': float(np.sum(s)), 'u': u}
        return self._obs(), float(r), term, trunc, info


if __name__ == '__main__':
    env = SpotDockEnv()
    obs, _ = env.reset(seed=0)
    print('obs dim', obs.shape, 'action dim', env.action_space.shape)
    tot = 0; nfeas = 0
    for k in range(int(T_MAX/DT)):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        tot += r; nfeas += (info['slack'] < 1e2)
        if k % 80 == 0:
            print(f"t={env.t:4.1f} r={r:+7.3f} h_tar={info['h_tar']:+.3f} h_obs={info['h_obs']:+.3f} "
                  f"h_los={info['h_los']:+.3f} V={info['V']:.2f} |u|={np.linalg.norm(info['u']):.3f}")
        if term or trunc:
            print(f"  episode end at t={env.t:.1f}  docked={info['docked']}"); break
    print(f"feasible steps {nfeas}/{k+1}, total reward {tot:.1f}")
