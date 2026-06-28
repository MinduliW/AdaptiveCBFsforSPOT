"""Why is case 2 not improvable by gain tuning, while 0 and 1 are?
Roll NOMINAL gains on cases 0-2 and report the geometry + where force/torque is spent."""
import numpy as np
from spot_env import SpotDockEnv, DT, TEST_CASES, Rmat, DOCK_OFF, wrap, K_DOCK, TORQUE_ARM

def seg_point_dist(p, a, b):
    """min distance from point p to segment a-b (chaser-path vs obstacle)."""
    ab = b - a; t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t * ab))

print("case | start->dock dist | attitude change | obstacle-to-path | min margins tar/obs/los | F / T / total | F early/late")
for tc in range(3):
    xR0, xB0, xU0 = TEST_CASES[tc]
    env = SpotDockEnv(randomize=False, setconst=True, const_gains=(1., .5, .25),
                      const_kdock=K_DOCK, test_case=tc, t_max=150.0)
    env.reset(seed=0)
    # dock target (in target frame)
    r_des = xB0[:2] + Rmat(xB0[2]) @ DOCK_OFF[:2]
    th_des = wrap(xB0[2] + DOCK_OFF[2])
    dist = np.linalg.norm(xR0[:2] - r_des)
    datt = np.degrees(abs(wrap(xR0[2] - th_des)))
    obs_to_path = seg_point_dist(xU0[:2], xR0[:2], r_des)   # is obstacle between start and dock?
    F = T = 0.0; Fearly = Flate = 0.0; mh = [9, 9, 9]; n = 0; dk = False
    Fs = []
    for _ in range(int(150 / DT)):
        _, _, term, trunc, info = env.step(np.zeros(10)); u = info["u"]
        f = np.hypot(u[0], u[1]); F += f * DT; T += abs(u[2]) * DT; Fs.append(f)
        mh = [min(mh[0], info["h_tar"]), min(mh[1], info["h_obs"]), min(mh[2], info["h_los"])]
        if info["docked"] and not dk:
            dk = True
        if term or trunc:
            break
    Fs = np.array(Fs); half = len(Fs) // 2
    Fearly = Fs[:half].sum() * DT; Flate = Fs[half:].sum() * DT
    tot = F + T / TORQUE_ARM
    print("  %d  |     %4.2f m       |     %5.1f deg    |     %5.2f m      | %+.2f/%+.2f/%+.2f      "
          "| %.2f/%.3f/%.1f | %.2f/%.2f"
          % (tc, dist, datt, obs_to_path, mh[0], mh[1], mh[2], F, T, tot, Fearly, Flate))
print("\n(obstacle-to-path < ~0.45 m KOZ radius => obstacle blocks the straight path -> forced detour)")
