"""
SPOT testbed parameters, extracted from SPOT-main (v4.1).

Provenance:
- Masses/inertias/CG: GUI_v4_1_MassProperties.mlapp defaults
  (SPOTGUI_DEFAULT.mat, scale measurements dated 2025-08-29).
- Thruster geometry & H matrix: MakeH / MakeHWithDecay MATLAB Function
  blocks inside Template_v4_1_0_2024b_Jetson.slx.
- Plant model: RED_dynamics / BLACK_dynamics MATLAB Function blocks:
  pure planar double integrator (air bearings -> negligible friction):
      x_ddot = Fx/m,  y_ddot = Fy/m,  theta_ddot = Tz/I
  with (Fx, Fy) in the INERTIAL table frame.
- Timing: baseRate = 1/SampleRateEditField = 1/20 s, solver ode4 (RK4).
- Table frame: inertial frame fixed to the granite table corner. Home
  positions (y = 1.209675 m, x = {0.8558, 1.7558, 2.6558} m) imply a
  working surface of ~3.5116 m x 2.4194 m.
"""

import numpy as np

# ----------------------------- timing ---------------------------------
SAMPLE_RATE_HZ = 20.0          # GUI: SampleRateEditField
DT = 1.0 / SAMPLE_RATE_HZ      # baseRate, ode4 fixed step

# ------------------------------ table ---------------------------------
TABLE_X = 3.51155              # [m] inferred from home-position layout
TABLE_Y = 2.41935              # [m]

# Home / default initial conditions (subAppStateInit)
HOME = {
    "RED":   dict(x=2.655775, y=1.209675, th=np.deg2rad(180.0)),
    "BLACK": dict(x=1.755775, y=1.209675, th=0.0),
    "BLUE":  dict(x=0.855775, y=1.209675, th=0.0),
}

# --------------------------- mass properties --------------------------
# Scale measurements [kg]: A = left-middle-edge, B = right-back corner,
# C = right-front corner, at body-frame positions
#   A: (x,y) = (0, +0.15), B: (-0.15, -0.15), C: (+0.15, -0.15)
SCALES = {
    "RED":   (5.863, 2.690, 3.584),
    "BLACK": (5.955, 2.644, 3.706),
    "BLUE":  (5.955, 2.644, 3.706),
}

MASS = {k: sum(v) for k, v in SCALES.items()}        # RED 12.137, BLACK/BLUE 12.305

# Inertias [kg m^2] (bifilar pendulum calc in GUI; defaults from .mat)
INERTIA = {"RED": 0.19816136536704418,
           "BLACK": 0.1995653708750313,
           "BLUE": 0.19609228544737417}


def cg_offset(name):
    """CG offset (Xcg, Ycg) [m] in the body frame, from scale readings."""
    A, B, C = SCALES[name]
    M = A + B + C
    x1, y1 = 0.0, 0.15
    x2, y2 = -0.15, -0.15
    x3, y3 = 0.15, -0.15
    return ((A*x1 + B*x2 + C*x3) / M, (A*y1 + B*y2 + C*y3) / M)


# ------------------------- thruster geometry --------------------------
# 8 thrusters per platform. Direction map (MakeH):
#   thrusters 1,2 push -X(body); 5,6 push +X; 3,4 push +Y; 7,8 push -Y.
# Moment arms below are the signed distances to CG [m]
# ("Measured on 2025-08-29" comments in GUI_v4_1_MassProperties.mlapp).
F_NOMINAL = 0.2825             # [N] nominal thrust per thruster (all axes)

VEC_X = np.array([-1, -1, 0, 0, 1, 1, 0, 0], dtype=float)
VEC_Y = np.array([0, 0, 1, 1, 0, 0, -1, -1], dtype=float)


def thruster_arms(name):
    Xc, Yc = cg_offset(name)
    if name == "RED":
        return np.array([
            0.15 - 0.0835 - Yc,
            -(0.15 - 0.0845 + Yc),
            0.15 - 0.0785 - Xc,
            -(0.15 - 0.0770 + Xc),
            0.15 - 0.0815 + Yc,
            -(0.15 - 0.0845 - Yc),
            0.15 - 0.0845 + Xc,
            -(0.15 - 0.0860 - Xc),
        ])
    # BLACK and BLUE share the same nozzle offsets in the GUI source
    return np.array([
        0.15 - 0.0810 - Yc,
        -(0.15 - 0.0810 + Yc),
        0.15 - 0.0800 - Xc,
        -(0.15 - 0.0785 + Xc),
        0.15 - 0.0810 + Yc,
        -(0.15 - 0.0886 - Yc),
        0.15 - 0.0840 + Xc,
        -(0.15 - 0.0870 - Xc),
    ])


def make_H(name, decay=1.0):
    """H maps duty cycles d in [0,1]^8 to the BODY wrench [Fx, Fy, Tz].

    Mirrors MakeHWithDecay: H = [VEC_X; VEC_Y; arms]' * diag(F*decay/2).
    Note the /2: a full-duty opposing pair delivers F_NOMINAL net per axis.
    """
    arms = thruster_arms(name)
    Mat1 = np.vstack([VEC_X, VEC_Y, arms])
    F = np.full(8, F_NOMINAL) * decay
    return Mat1 @ np.diag(F / 2.0)


def thrust_decay_factor(duty):
    """check_thrust_decay (verbatim logic): plenum-pressure decay model."""
    d = np.asarray(duty, dtype=float)
    if d.max(initial=0.0) > 1.0:
        d = d / d.max()
    count = int(np.sum(d > 0))
    avg = float(np.sum(np.maximum(d, 0.0))) / max(count, 1)
    if avg < 0.3 or count == 0:
        return 1.0
    return max(0.6 - 2.0 * avg + 1.0, 0.5)


# Conservative per-axis force capability (one opposing pair, worst-case decay)
F_AXIS_MAX = F_NOMINAL          # [N] at decay=1
F_AXIS_MIN_GUARANTEED = F_NOMINAL * 0.5   # [N] decay floor = 0.5
