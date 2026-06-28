"""
Plant dynamics and thruster allocation, replicating the SPOT Simulink template.

State per platform: x = [px, py, th, vx, vy, w]  (inertial table frame)
Plant (RED_dynamics/BLACK_dynamics blocks):
    px_dd = Fx/m, py_dd = Fy/m, th_dd = Tz/I, with (Fx,Fy) inertial.

Actuation path (as on the testbed):
  controller wrench (inertial) -> rotate to body -> bounded-LSQ duty-cycle
  allocation with thrust decay (optimize_duty_cycle_with_decay) ->
  realized body wrench = H(decay) d -> rotate back to inertial -> plant.

Integration: RK4 ("ode4") at DT = 0.05 s, matching the template solver.
"""

import numpy as np
from scipy.optimize import lsq_linear

import spot_params as P


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


class ThrusterAllocator:
    """Mirrors optimize_duty_cycle_with_decay from the .slx (RED chart_517):
    iterate { H(decay) -> bounded least squares 0<=d<=1 -> update decay }.
    """

    def __init__(self, name, max_iters=10, tol=1e-4):
        self.name = name
        self.max_iters = max_iters
        self.tol = tol

    def allocate(self, wrench_body):
        d = np.zeros(8)
        decay = 1.0
        prev = d + 2 * self.tol
        H = P.make_H(self.name, decay)
        for _ in range(self.max_iters):
            H = P.make_H(self.name, decay)
            # tiny Tikhonov term -> unique, minimum-duty solution (the
            # MATLAB routine achieves the same via its x0 = 0 warm start)
            lam = 1e-4
            Ha = np.vstack([H, np.sqrt(lam) * np.eye(8)])
            wa = np.concatenate([wrench_body, np.zeros(8)])
            res = lsq_linear(Ha, wa, bounds=(0.0, 1.0))
            d = res.x
            decay = P.thrust_decay_factor(d)
            if np.max(np.abs(d - prev)) < self.tol:
                break
            prev = d
        H = P.make_H(self.name, decay)
        realized = H @ d
        return d, realized, decay


class Platform:
    def __init__(self, name, x0=None):
        self.name = name
        self.m = P.MASS[name]
        self.I = P.INERTIA[name]
        h = P.HOME[name]
        self.x = np.array([h["x"], h["y"], h["th"], 0, 0, 0], dtype=float) \
            if x0 is None else np.asarray(x0, dtype=float)
        self.alloc = ThrusterAllocator(name)

    def deriv(self, x, F_inertial, Tz):
        return np.array([x[3], x[4], x[5],
                         F_inertial[0] / self.m,
                         F_inertial[1] / self.m,
                         Tz / self.I])

    def step(self, wrench_inertial_cmd, dt=P.DT):
        """Apply a commanded inertial wrench [Fx, Fy, Tz] for one step.

        The wrench is pushed through the real actuation path (body-frame
        duty-cycle allocation with decay + saturation); the REALIZED wrench
        is held constant over the step (20 Hz ZOH) and integrated with RK4.
        Returns (duty_cycles, realized inertial wrench).
        """
        th = self.x[2]
        Fb = rot(th).T @ wrench_inertial_cmd[:2]
        d, w_body, _ = self.alloc.allocate(np.array([Fb[0], Fb[1],
                                                     wrench_inertial_cmd[2]]))
        F_in = rot(th) @ w_body[:2]   # attitude change over 50 ms is negligible
        Tz = w_body[2]

        f = lambda x: self.deriv(x, F_in, Tz)
        x = self.x
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        self.x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        return d, np.array([F_in[0], F_in[1], Tz])
