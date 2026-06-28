"""
Render the MLP (full obs) and LSTM (blind) policies on the 5 named cases to
separate clips in /tmp/cmp/, for side-by-side stitching with ffmpeg.

    python compare_videos.py
"""
import os
import main_train_adaptive as M
import run_matlab_case as RC
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

os.makedirs("/tmp/cmp", exist_ok=True)
MLP_DIR  = "BestModels/adaptive_5of5_4457_FROMSCRATCH"
LSTM_DIR = "BestModels/adaptive_lstm_5of5_4284_blind"


def render(model_dir, rnn, hidden, rl2, tag):
    M.HIDDEN_OBS = hidden; M.HIDDEN_RL2 = rl2
    venv = VecNormalize.load(model_dir + "/vecnormalize.pkl", DummyVecEnv([M.make_env(0)]))
    venv.training = False
    model = (RecurrentPPO if rnn else PPO).load(model_dir + "/best_model.zip")
    for tc in range(5):
        log, _, m = M.rollout_policy(model, venv, tc, rnn)
        RC.animate(log, tc, out="/tmp/cmp/%s_case%d.mp4" % (tag, tc))
        print("  %s case %d  docked=%s" % (tag, tc, m["docked"]))


print("rendering MLP clips...");  render(MLP_DIR,  False, False, False, "mlp")
print("rendering LSTM clips..."); render(LSTM_DIR, True,  True,  True,  "lstm")
print("done -> /tmp/cmp/{mlp,lstm}_case0..4.mp4")
