# Deploying the RL gain policy to the Windows SPOT testbed

End-to-end: clone the repo → build the Python env → point MATLAB at it → wire the RL
gain policy into the SPOT Simulink model. Same `pyenv` mechanism as `Python_TRMPC`,
but Python returns the **class-K gains** (your CBF-CLF QP stays in MATLAB).

## 1. Get the code (on Windows)
```
git clone https://github.com/MinduliW/AdaptiveCBFsforSPOT.git
```
Note the full path, e.g. `C:\Users\<you>\AdaptiveCBFsforSPOT`.

## 2. Build the Python environment (Miniconda)
```
conda create -n spot-rl python=3.11
conda activate spot-rl
pip install -r SPOT_pyenv_bridge\requirements.txt
where python        ::  -> C:\Users\<you>\miniconda3\envs\spot-rl\python.exe   (note this)
```

## 3. Verify the policy runs standalone (no MATLAB yet)
```
python spot_rl_policy.py
```
Should print `k_dock` and the 3×3 gains. If this works, the Python side is correct.

## 4. Point MATLAB at the env — add to `Run_Initializer.m`
```matlab
pyenv(Version="C:\Users\<you>\miniconda3\envs\spot-rl\python.exe");
insert(py.sys.path, int32(0), "C:\Users\<you>\AdaptiveCBFsforSPOT");
```
(Use double backslashes `\\` or forward slashes in MATLAB strings.) Verify in MATLAB:
```matlab
[g,k,s] = call_python_policy([1.5 2 pi 0 0 0], [2.2 0.2 4.712 0 0 0], [2 1.2 0 0 0 0], [0.85 0.85])
% -> s = 1, k ~ 4.743, g is 3x3
```

## 5. Wire into the SPOT Simulink model
Copy `SPOT_pyenv_bridge\call_python_policy.m` and `spot_rl_fcn.m` into your SPOT project
folder (next to where `Python_TRMPC` keeps `fcn.m` / `call_python_mpc.m`). Then:
- Add a **MATLAB Function block** and paste `spot_rl_fcn.m`.
- **in:**  `x_red`(6), `x_black`(6), `x_obstacle`(6), `holding_radius`(2) — from the
  PhaseSpace/estimator (the same signals that feed your controller; `x_obstacle` is the
  BLUE platform, or a fixed far pose if there's no physical obstacle).
- **out:** `g`(3×3), `k_dock`(1) — into the CBF-CLF QP block, replacing the gains it
  currently has fixed.
- Set the model to **Normal** simulation mode (Python co-execution can't be code-generated).

## 6. Run
First step lags (~seconds: torch import + model load), then it runs at rate. On any Python
error the wrapper returns `status_code = -99` and the **nominal gains (1, 0.5, 0.25)** as a
safe fallback.

## Troubleshooting
- `pyenv` must be set **before** the first `py.*` call and can't change mid-session — restart
  MATLAB if you change it.
- `ModuleNotFoundError: spot_rl_policy` → the `insert(py.sys.path, …)` path is wrong/missing.
- `ModuleNotFoundError: torch`/`stable_baselines3` → `pyenv(Version=…)` points at the wrong env.
- Slow to install / large env → the heavy dep is **torch (≈345 MB)**. A **numpy-only** bridge
  (drops torch + SB3, leaving `pip install numpy scipy`) is available on request — same
  `get_gains` interface, so the MATLAB side is unchanged.
