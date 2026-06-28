# SPOT Python Sim — LQR target + ICCBF chaser

A standalone Python port of the SPOT (v4.1) testbed dynamics, with:

- **BLACK (target):** crosses the granite table tracking a min-jerk
  reference with a **discrete LQR** at the testbed's 20 Hz rate.
- **RED (chaser):** station-keeps 0.65 m behind the target with a nominal
  LQR, filtered through an **ICCBF** (Input-Constrained Control Barrier
  Function) safety layer that enforces a keep-out circle around the moving
  target, the table boundaries, and the platform's *actual* actuation
  limits (8 thrusters, duty cycles in [0,1], pressure decay).

## Run

```bash
pip install numpy scipy matplotlib
python run_spot_sim.py
```

Outputs: `spot_iccbf_trajectory.png`, `spot_iccbf_timeseries.png`,
`spot_sim_log.csv`.

## What was taken from SPOT-main, and where

| Item | Value | Source in repo |
|---|---|---|
| Plant model | ẍ=Fx/m, ÿ=Fy/m, θ̈=Tz/I (no friction) | `RED_dynamics`/`BLACK_dynamics` MATLAB Fcn blocks in `Template_v4_1_0_2024b_Jetson.slx` |
| Masses | RED 12.137, BLACK/BLUE 12.305 kg | scale measurements (A,B,C) in `GUI_v4_1_MassProperties.mlapp` / `SPOTGUI_DEFAULT.mat` |
| Inertias | RED 0.19816, BLACK 0.19957, BLUE 0.19609 kg·m² | bifilar-pendulum calc, same files |
| CG offsets | e.g. RED (+11.05, −5.08) mm | computed from scales at (0,.15), (−.15,−.15), (.15,−.15) |
| Thruster geometry | 8 nozzles on a 0.3 m bus, arms per 2025-08-29 measurements | `Thr1R…Thr8R` in the MassProperties app (verified against `CREDScaleMeasurement_*` in the default .mat) |
| Nominal thrust | 0.2825 N per thruster | `F_red_X_nominal` etc. in `GUI_v4_1_Main.mlapp` |
| Allocation H | `[Fx;Fy;Tz] = Mat1·diag(F/2)·d`, d∈[0,1]⁸ | `MakeH` / `MakeHWithDecay` |
| Thrust decay | factor = max(1.6 − 2·avg_duty, 0.5) when avg_duty ≥ 0.3 | `check_thrust_decay` |
| Allocator | iterative {build H(decay) → bounded LSQ → update decay} | `optimize_duty_cycle_with_decay` (here via `scipy.optimize.lsq_linear` + small Tikhonov term) |
| Rate / solver | 20 Hz, RK4 fixed step (ode4) | GUI `SampleRateEditField`, `baseRate`, solver settings |
| Table & homes | 3.5116 × 2.4194 m; homes at y=1.2097, x∈{0.856, 1.756, 2.656} | `subAppStateInit` defaults |
| Attitude PD gains | Kp=0.5, Kd=1.8 | GUI default P/D gain fields |

## Frames

- **Inertial table frame**: origin at the table corner, x along the long
  edge — same frame the PhaseSpace cameras report in and the one the
  Simulink plant integrates in.
- **Body frame**: at each platform's CG. The controller computes inertial
  forces; they are rotated to body for duty-cycle allocation and the
  realized wrench is rotated back — mirroring the Simulink signal path.

## ICCBF design notes

For a double integrator with per-axis acceleration bound `a_max`
(derived from the *guaranteed* axis force: 0.2825 N × 0.5 decay floor ×
0.85 torque reserve / m ≈ 0.0099 m/s²), the ICCBF construction of
Agrawal & Panagou (CDC 2021) starting from `h₀ = ‖p_rel‖ − r` with the
class-K choice `α₀(s) = √(2 a_brk s)` gives the closed-form braking
barrier

```
b₁(x) = √(2 a_brk (‖p_rel‖ − r)) + d/dt ‖p_rel‖
```

whose zero-superlevel set is controlled-invariant **within the input
constraints** — the point of ICCBFs: feasibility of the safety QP is
guaranteed by construction rather than hoped for. A braking budget
`a_brk = a_max − a_tgt_max` reserves authority for the target's own
acceleration. The same construction protects the four table walls. The
filter solves a 2-variable QP (SLSQP, analytic projection fallback) each
step, and the enforced radius is inflated 4 cm to absorb the 20 Hz
zero-order-hold discretization of the continuous-time CBF condition.

## Things intentionally simplified

- PWM is treated as proportional thrust at 20 Hz (real hardware PWMs the
  solenoids at 5 Hz); the decay model and [0,1] duty saturation are kept.
- No sensor noise / PhaseSpace model — add Gaussian noise on `Platform.x`
  if you want estimation-in-the-loop.
- The manipulator arm (3-link dynamics in the repo) is not included.

## CBF-RVD port (from test2.zip / CBF_RVD.slx)

`cbf_rvd.py` ports the user's actual hardware controller stack:
dock CLF (Q_dock, λ=0.25, k_dock=5, slack δ), target-KOZ **ICCBF**
(N=2, rotating ellipse, a₀=a₁=2.5, a₂=0.5, re-derived in sympy exactly
as `build_iccbf_functions.m` does), LOS sensor-cone **ICCBF**
(a₀=0.8, a₁=1.2, a₂=0.5, torque-only; the HOCBF variant from chart_673
is also available via `los_mode="hocbf"`), BLUE-obstacle HOCBF,
velocity CBF, the quadprog-style QP (min ‖u‖²+10δ², |u|≤[0.1,0.1,0.015]),
and the γ-shrinking KOZ logic. `run_cbf_rvd.py` runs test_case 2
(tumbling BLACK at 1.5°/s, drifting BLUE) through the full thruster
allocation path. Result: all barriers ≥ 0 throughout with the LOS ICCBF;
the HOCBF LOS variant violates by ~7.6° during the KOZ swing — the
input-constraint gap the ICCBF construction closes.
