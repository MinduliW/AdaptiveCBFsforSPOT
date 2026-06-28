# SPOT RL gain policy via `pyenv` (MATLAB ↔ Python co-execution)

Same pattern as `trmpc-docking/.../Python_TRMPC` (`call_python_mpc.m` + `pyenv`), but the
Python side runs the trained **RL gain policy** instead of the MPC. Each control step
MATLAB passes the lab-frame states and gets back the **class-K gains** the policy chose;
your MATLAB/Simulink CBF-CLF QP then uses them to compute the force/torque. The QP stays
in MATLAB — Python only supplies the gains.

Pipeline run in Python (identical to training):
```
states -> obs(28) -> VecNormalize -> MLP -> gains(3x3) + k_dock
```

## Files
| file | where | purpose |
|---|---|---|
| `spot_rl_policy.py` | **SPOTEmulate root** | the Python module MATLAB imports (`get_gains`) |
| `call_python_policy.m` | this folder → your SPOT project | MATLAB wrapper (mirror of `call_python_mpc.m`) |

`spot_rl_policy.py` must stay in the SPOTEmulate root because it imports `spot_env`,
`main_train_MLP`, and loads the model from `TrainedModels/…`.

## Setup — add to `Run_Initializer.m` (exactly like the trmpc one)
```matlab
% --- Python (pyenv) setup for the RL gain policy ---
pyenv(Version="/Users/minduli/miniconda3/envs/rlcbf_py311/bin/python");   % your training conda env
insert(py.sys.path, int32(0), "/Users/minduli/Downloads/SPOTEmulate");    % dir with spot_rl_policy.py
```
(The trmpc file does the same two lines.) To swap checkpoints:
```matlab
setenv("SPOT_RL_MODEL", "/full/path/to/TrainedModels/<run dir>");
```

## Wire into Simulink (same mechanism as trmpc's `fcn.m`)
Python co-execution (`py.*`) **cannot be code-generated**, so it goes inside a
**MATLAB Function block** with `coder.extrinsic` and pre-declared output sizes —
exactly like trmpc's `fcn.m`. `spot_rl_fcn.m` is that block body.

1. Put `call_python_policy.m` and `spot_rl_fcn.m` on the MATLAB path (e.g. the project folder,
   next to where `call_python_mpc.m`/`fcn.m` live). `spot_rl_policy.py` stays in SPOTEmulate.
2. Add a **MATLAB Function block** (Simulink → User-Defined Functions) and paste the contents
   of `spot_rl_fcn.m` (or have it just call `spot_rl_fcn`).
3. Wire the ports:
   - **in:**  `x_red`(6), `x_black`(6), `x_obstacle`(6), `holding_radius`(2) — from your
     state estimator (the same signals that feed the current controller)
   - **out:** `g` (3×3), `k_dock` (1) — into your CBF-CLF QP block (reshape `g` to a 9-vector
     if your QP wants it flat)
4. Set the model to **Normal** simulation mode (NOT Accelerator / Rapid Accelerator / codegen
   — `py.*` only runs in the interpreter, same limit as trmpc's Python project).
5. Run. The **first** step is slightly slow (load `spot_policy.mat`); every step after is fast.

**Easiest path:** clone the existing `Python_TRMPC` block that runs `fcn.m`, swap
`fcn` → `spot_rl_fcn`, add the two extra inputs (`x_obstacle`, `holding_radius`), and route the
outputs from `u` to `g`/`k_dock`. Everything else (the `pyenv` setup, the extrinsic call) is
identical.

```matlab
% spot_rl_fcn.m  (the MATLAB Function block body)
function [g, k_dock] = spot_rl_fcn(x_red, x_black, x_obstacle, holding_radius)
%#codegen
coder.extrinsic('call_python_policy');
g = zeros(3,3);  k_dock = 5.0;                 % pre-declare (required for extrinsic)
g_tmp = zeros(3,3);  k_tmp = 0;  status_tmp = 0;
[g_tmp, k_tmp, status_tmp] = call_python_policy(x_red, x_black, x_obstacle, holding_radius);
g = g_tmp;  k_dock = k_tmp;
end
```

## Verify before wiring into Simulink
Python side standalone:
```bash
cd /Users/minduli/Downloads/SPOTEmulate
python spot_rl_policy.py
# k_dock = 4.743
# gains = [[1.293 0 0.172],[1.867 0 0.01],[1.498 0 0.18]]
```
Then in MATLAB, after the `pyenv`/`insert` lines:
```matlab
[g,k,s] = call_python_policy([1.5 2 pi 0 0 0], [2.2 0.2 4.712 0 0 0], [2 1.2 0 0 0 0], [0.85 0.85])
```
should give `s = 1`, `k ≈ 4.743`, and the same `g`.

## Notes
- The bridge is **numpy-only** (no torch/SB3): the MLP forward pass is plain matrix math
  from `spot_policy.mat`. First call loads the .mat; subsequent calls are fast — the
  module caches model/normalizer/env in globals (the `persistent` on the MATLAB side caches
  the imported module handle).
- States are **lab frame** `[x y theta dx dy dtheta]`, same convention as `TEST_CASES`.
- `holding_radius` maps to the target keep-out size (`rkoz_tar`); pass your controller's
  current holding radius so the observation matches the live KOZ. Omitting it (or passing
  the initial size) is fine for first bring-up.
- The gains come out in the **same layout your QP expects** from `_decode`: rows are the
  three ICCBFs `[tar-KOZ; obs-KOZ; LOS]`, columns are the class-K coefficients `[a0 a1 a2]`,
  and `k_dock` is the CLF decay rate.
