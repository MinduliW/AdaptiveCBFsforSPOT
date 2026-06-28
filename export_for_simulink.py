"""
Export a trained SB3 MLP policy for Simulink.

Produces everything the 3-stage pipeline needs:
  spot_policy.onnx   -- the network (obs -> action mean), import with importNetworkFromONNX
  obs_mean.csv       -- VecNormalize mean   (stage 2: normalize BEFORE the net)
  obs_std.csv        -- VecNormalize std    normalized = clip((obs-mean)/std, -clip, +clip)
  decode printed     -- action[10] -> gains (stage 4: AFTER the net)

    python export_for_simulink.py [model_dir]
"""
import sys
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import main_train_MLP as M
from spot_env import SpotDockEnv, K_DOCK

DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "TrainedModels/PPO_adaptive_l2_n128_lr0.0003D_std0.2_ne10_g0.9997_ent0.01_scratch_rand_anneal_resid_prop_tof"

# match the obs mode the model was trained with (full-obs MLP)
M.HIDDEN_OBS = False; M.HIDDEN_RL2 = False; M.RESIDUAL_GAINS = True
venv = VecNormalize.load(DIR + "/vecnormalize.pkl", DummyVecEnv([M.make_env(0)]))
model = PPO.load(DIR + "/best_model.zip")
obs_dim = int(model.observation_space.shape[0])


# ---- [3] the network: CLEAN actor MLP only ----
# Exporting the FULL policy forward drags in the value head + Gaussian distribution,
# which MATLAB imports as PLACEHOLDER layers -> initialize(net) = "invalid network".
# The actor is a plain Linear/Tanh MLP, so export just policy_net + action_net:
#   obs -> action mean (deterministic). Clip/decode happen downstream in Simulink.
det = th.nn.Sequential(*list(model.policy.mlp_extractor.policy_net),
                       model.policy.action_net).eval()
dummy = th.zeros(1, obs_dim, dtype=th.float32)

# verify the clean MLP reproduces the policy's deterministic action exactly
with th.no_grad():
    a_full = model.policy(dummy, deterministic=True)[0]
    assert th.allclose(det(dummy), a_full, atol=1e-5), "clean actor != policy mean"

# (a) TorchScript -> MATLAB importNetworkFromPyTorch  (needs only torch)
th.jit.trace(det, dummy).save("spot_policy.pt")
print("[3a] wrote spot_policy.pt   (TorchScript -> importNetworkFromPyTorch)")

# (b) ONNX -> MATLAB importNetworkFromONNX  (fixed [1,N] input, clean Gemm/Tanh, no placeholders)
try:
    th.onnx.export(det, dummy, "spot_policy.onnx", input_names=["obs"], output_names=["action"],
                   opset_version=13)
    print("[3b] wrote spot_policy.onnx (obs %d -> action 10, clean MLP)" % obs_dim)
except Exception as ex:
    print("[3b] ONNX skipped (pip install onnx to enable):", type(ex).__name__)

# ---- [2] normalization constants (apply BEFORE the net) ----
mean = venv.obs_rms.mean.astype(np.float32)
std = np.sqrt(venv.obs_rms.var + venv.epsilon).astype(np.float32)
np.savetxt("obs_mean.csv", mean, delimiter=","); np.savetxt("obs_std.csv", std, delimiter=",")
print("[2] obs_mean.csv / obs_std.csv  -> normalized = clip((obs-mean)/std, -%.0f, +%.0f)" % (venv.clip_obs, venv.clip_obs))

# ---- [4] decode constants (apply AFTER the net): action[10] -> 3x3 gains + k_dock ----
e = SpotDockEnv()
print("\n[4] decode (residual gains): reshape action[0:9] -> grid[3][3] (rows tar,obs,los; cols a0,a1,a2)")
print("    gains[row][col] = clip(BASE[col] + BAND[col]*grid[row][col],  0, hi)")
print("    BASE  (a0,a1,a2) =", tuple(getattr(e, "RESID_BASE", (1.0, 0.5, 0.25))))
print("    BAND  (a0,a1,a2) =", tuple(getattr(e, "RESID_BAND", (2.0, 2.0, 0.5))))
print("    k_dock = clip(%.1f + %.1f*action[9], 0.1, inf)" % (K_DOCK, getattr(e, "RESID_KBAND", 4.0)))

# ---- [5] raw weights + all constants -> spot_policy.mat (rebuild in a MATLAB Function block; no import) ----
import scipy.io as sio
sd = det.state_dict()
sio.savemat("spot_policy.mat", dict(
    W1=sd["0.weight"].detach().numpy(), b1=sd["0.bias"].detach().numpy().reshape(-1, 1),
    W2=sd["2.weight"].detach().numpy(), b2=sd["2.bias"].detach().numpy().reshape(-1, 1),
    W3=sd["4.weight"].detach().numpy(), b3=sd["4.bias"].detach().numpy().reshape(-1, 1),
    obs_mean=mean.reshape(-1, 1), obs_std=std.reshape(-1, 1), clip_obs=float(venv.clip_obs),
    BASE=np.asarray(getattr(e, "RESID_BASE", (1.0, 0.5, 0.25)), float).reshape(-1, 1),
    BAND=np.asarray(getattr(e, "RESID_BAND", (2.0, 2.0, 0.5)), float).reshape(-1, 1),
    KDOCK=float(K_DOCK), KBAND=float(getattr(e, "RESID_KBAND", 4.0)),
    ACOEF_HI=float(e.ACOEF_HI), HSLACK_LO=float(e.HSLACK_LO), HSLACK_HI=float(e.HSLACK_HI)))
print("[5] wrote spot_policy.mat  (W1/b1..W3/b3 + obs_mean/std + decode consts)")

# sanity: ONNX vs torch on a random obs
try:
    import onnxruntime as ort
    o = np.random.randn(1, obs_dim).astype(np.float32)
    a_torch = det(th.tensor(o)).detach().numpy()
    a_onnx = ort.InferenceSession("spot_policy.onnx").run(None, {"obs": o})[0]
    print("\nsanity: max|onnx-torch| = %.2e (should be ~0)" % np.abs(a_torch - a_onnx).max())
except Exception as ex:
    print("\n(install onnxruntime to verify parity:", ex, ")")
