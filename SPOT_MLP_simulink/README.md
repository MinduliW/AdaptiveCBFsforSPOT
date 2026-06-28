# SPOT docking MLP policy — Simulink handoff

A trained feed-forward policy that outputs the **ICCBF class-K gains** for the SPOT
docking CBF-CLF QP. Given the current docking observation it returns the gains
`g` (3×3) and `k_dock` that the QP then uses to compute the safe force/torque.

It is a small MLP, so the simplest, most robust Simulink integration is to **rebuild
it from weights in a MATLAB Function block** (`spot_policy.m`) — pure matrix math, clean
code generation, no `importNetworkFromONNX`/`dlnetwork`. ONNX/TorchScript are included
too if you prefer the Deep Learning Toolbox `Predict` block.

## Pipeline (3 stages — the network is only the middle one)

```
raw obs (28) ──▶ [2] VecNormalize ──▶ [3] MLP ──▶ action (10) ──▶ [4] decode ──▶ g (3×3), k_dock ──▶ your QP
                  z-score + clip       28→128→128→10              residual gains
```

Stages 2 and 4 are **not** in the network — they must be reproduced in Simulink. All
the constants for them are in `spot_policy.mat`. `spot_policy.m` does all three stages.

## I/O

| signal | size | meaning |
|---|---|---|
| `obs`    | 28×1 | docking observation (layout below) |
| `action` | 10×1 | raw MLP output (network only) |
| `g`      | 3×3  | class-K gains; rows = `[tar-KOZ; obs-KOZ; LOS]`, cols = `[a0 a1 a2]` |
| `k_dock` | 1×1  | CLF decay rate |

### Observation layout (28) — build it in this exact order
Most of these you **already compute** in the QP controller (the CLF error `e`, the
barrier values `h_*`, the CLF value `V`). `xR`=chaser, `xB`=target, `xU`=obstacle,
each `[x y θ dx dy dθ]`.

| idx | size | content |
|---|---|---|
| 0–5   | 6 | `e` — docking error from the docking CLF (`clf_rows`) |
| 6–10  | 5 | `cos θR, sin θR, dxR, dyR, dθR` (chaser pose/vel) |
| 11–15 | 5 | `xR[0:2]-xB[0:2]`, `xR[3:5]-xB[3:5]`, `xB[5]` (relative target) |
| 16–20 | 5 | `xR[0:2]-xU[0:2]`, `xR[3:5]-xU[3:5]`, `xU[5]` (relative obstacle) |
| 21–25 | 5 | `h_tar, h_obs, h_los, h_vl, h_va` (barrier margins) |
| 26–27 | 2 | `V` (CLF value), `rkoz_tar[0] - R_KOZ_TAR_MIN[0]` (KOZ shrink) |

(Authoritative source: `SpotDockEnv._obs()` in `spot_env.py`.)

## Files

| file | purpose |
|---|---|
| `spot_policy.mat`     | **weights + normalization + decode constants** (everything) |
| `spot_policy.m`       | **ready MATLAB Function block** — obs → g, k_dock (recommended) |
| `spot_policy.onnx`    | network only, for `importNetworkFromONNX` (clean Gemm/Tanh) |
| `spot_policy.pt`      | network only, for `importNetworkFromPyTorch` |
| `obs_mean.csv` / `obs_std.csv` | normalization vectors (also inside the `.mat`) |
| `reference_io.csv`    | 44 rows of `obs(28) → action(10)` from real rollouts — **validation** |
| `policy_reference.py` | standalone numpy ground-truth + self-test (numpy+scipy only) |
| `MODEL.txt`           | which training checkpoint this was exported from |

## Option A — MATLAB Function block (recommended)

Put `spot_policy.m` + `spot_policy.mat` on the path and use a **MATLAB Function block**:
```
obs (28×1) ──▶ [ spot_policy ] ──▶ g (3×3), k_dock
```
That's the whole pipeline. Nothing else needed.

## Option B — Predict block (network only)

```matlab
net = importNetworkFromONNX("spot_policy.onnx");   % or InputDataFormats="BC" if needed
net = initialize(net);
```
Then a `Predict` block runs the network, but you **still** add `(obs-mean)./std` (clip ±`clip_obs`)
**before** it and the `decode` **after** it (see the math below). The ONNX graph is pure
`Gemm`+`Tanh` (no placeholder layers).

## The exact math (for porting)

**[2] Normalize:** `obs_n = clip((obs - obs_mean) ./ obs_std, -clip_obs, +clip_obs)`  (`clip_obs = 10`)

**[3] MLP:**
```
h = tanh(W1*obs_n + b1)     % W1 128×28, b1 128×1
h = tanh(W2*h    + b2)      % W2 128×128
action = W3*h + b3          % W3 10×128 -> 10×1
```

**[4] Decode (residual gains):**
```
a    = clip(action, -1, 1)
grid = reshape(a(1:9),[3,3]).'         % row-major; rows tar,obs,los; cols a0,a1,a2
g    = BASE + BAND .* grid             % BASE,BAND are 1×3 over (a0,a1,a2)
g(:,1:2) = clip(g(:,1:2), 0, ACOEF_HI)
g(:,3)   = clip(g(:,3), HSLACK_LO, HSLACK_HI)
k_dock   = max(0.1, KDOCK + KBAND*a(10))
```
> The `.'` transpose on `reshape` is **required** — Python reshapes row-major, MATLAB
> column-major. Without it the gains are scrambled.

## Validation

`reference_io.csv` has 44 real `(obs, action)` pairs. Feed each `obs` through your
stage-2+3 implementation and confirm the output matches `action` to ~1e-4. The Python
ground truth:
```bash
python policy_reference.py        # -> "max|action - reference| = ~1e-7 ; OK"
```
Replicating that match in Simulink confirms the conversion is correct.
