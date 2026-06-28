# Deploying the RL gain policy to the Windows SPOT testbed (numpy-only)

The policy is a trained MLP, and **running** a trained network is just matrix math —
no PyTorch needed. `spot_rl_policy.py` evaluates it with **numpy only** (weights live in
`spot_policy.mat`), proven identical to the torch version to ~1e-7. So the Python side is
just `pip install numpy scipy` — no conda, no torch, none of the DLL grief.

## 1. Get the code (on Windows)
```
git clone https://github.com/MinduliW/AdaptiveCBFsforSPOT.git
```
Note the path, e.g. `C:\Users\<you>\AdaptiveCBFsforSPOT`.

## 2. Python environment — trivial (numpy + scipy)
Use a plain **python.org** Python 3.x (NOT conda — that's what caused the `pyexpat`/DLL
errors). Install it (tick "Add to PATH"), then:
```
python -m venv C:\spot-rl-venv
C:\spot-rl-venv\Scripts\activate
pip install -r C:\Users\<you>\AdaptiveCBFsforSPOT\SPOT_pyenv_bridge\requirements.txt
where python        ::  -> C:\spot-rl-venv\Scripts\python.exe   (note this)
```
(`requirements.txt` is just `numpy` + `scipy`.)

## 3. Verify the policy runs standalone
```
cd C:\Users\<you>\AdaptiveCBFsforSPOT
python spot_rl_policy.py
```
Prints `k_dock` and the 3×3 gains → Python side is good.

## 4. Point MATLAB at it — add to `Run_Initializer.m`
```matlab
addpath("C:\Users\<you>\AdaptiveCBFsforSPOT\SPOT_pyenv_bridge");        % MATLAB finds the .m wrappers
pyenv(Version="C:\spot-rl-venv\Scripts\python.exe", ExecutionMode="OutOfProcess");
insert(py.sys.path, int32(0), "C:\Users\<you>\AdaptiveCBFsforSPOT");    % Python finds spot_rl_policy.py
```
Test in MATLAB:
```matlab
[g,k,s] = call_python_policy([1.5 2 pi 0 0 0],[2.2 0.2 4.712 0 0 0],[2 1.2 0 0 0 0],[0.85 0.85])
% -> s = 1, k ~ 4.743, g 3x3
```

## 5. Wire into the SPOT Simulink model
Add a **MATLAB Function block** with the body from `SPOT_pyenv_bridge\spot_rl_fcn.m`:
- **in:**  `x_red`(6), `x_black`(6), `x_obstacle`(6), `holding_radius`(2) — from PhaseSpace/estimator
- **out:** `g`(3×3), `k_dock`(1) — into your CBF-CLF QP, replacing its fixed gains
- Set the model to **Normal** simulation mode (Python co-execution can't be code-generated).

## 6. Run
First step lags ~1 s (load `spot_policy.mat`), then it runs at rate. On any Python error the
wrapper returns `status_code = -99` and nominal gains `(1, 0.5, 0.25)` as a safe fallback.

## Troubleshooting
- **Use python.org Python, not conda** — the `pyexpat`/`DLL load failed` errors come from
  conda's DLL layout fighting MATLAB. A plain python.org venv with numpy+scipy avoids it.
- `ExecutionMode="OutOfProcess"` runs Python in a separate process (extra robustness vs DLL
  clashes); it can't be changed mid-session, so restart MATLAB if you tweak the `pyenv` line.
- `Unrecognized function 'call_python_policy'` → the `addpath(...\SPOT_pyenv_bridge)` is missing.
- `ModuleNotFoundError: spot_rl_policy` → the `insert(py.sys.path,…)` path is wrong.
- `FileNotFoundError: spot_policy.mat` → run from / point at the repo root (the weights file is there).

## Updating the policy after retraining
Re-export the weights and regenerate the numpy module:
```
python export_for_simulink.py <model_dir>   # -> spot_policy.mat
python make_slim.py                          # -> spot_rl_policy.py (numpy-only)
```
Commit `spot_policy.mat` + `spot_rl_policy.py`, pull on Windows. The MATLAB side is unchanged.
