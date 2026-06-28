"""
Load the trained PPO-bandit policy, read off its constant gains, and render the
case as a video.  python render_rl_gains.py [test_case]
"""
import sys
import numpy as np
from stable_baselines3 import PPO

from spot_env import SpotDockEnv
import run_matlab_case as RC

tc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
model = PPO.load("best_constgains_case%d/best_model" % tc)

# contextless bandit -> deterministic action on the dummy zero observation
a, _ = model.predict(np.zeros((1, 1), dtype=np.float32), deterministic=True)
theta = np.asarray(a, dtype=float).reshape(-1)
g, k = SpotDockEnv()._decode(theta)
print("RL gains  tar-KOZ=%s  obs-KOZ=%s  LOS=%s  k_dock=%.2f"
      % (np.round(g[0], 3), np.round(g[1], 3), np.round(g[2], 3), k))

log, docked = RC.rollout(tc, action=theta)
out = RC.animate(log, tc, out="spot_case%d_rl.mp4" % tc)
sep = np.linalg.norm(log["xB"][-1, :2] - log["xR"][-1, :2])
print("wrote %s  (%d steps, %.1f s)  docked=%s  final sep=%.3f m"
      % (out, len(log["t"]), log["t"][-1], docked, sep))
print("  min margins  h_tar=%+.3f h_obs=%+.3f h_los=%+.3f"
      % (log["h_tar"].min(), log["h_obs"].min(), log["h_los"].min()))
