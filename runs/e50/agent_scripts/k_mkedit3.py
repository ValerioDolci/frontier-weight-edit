import numpy as np
L = 30
w = np.load(f"{WORK}/det_w2_L{L}.npy")
d_imp = np.load(f"{WORK}/steer_dimp_{L}.npy"); d_start = np.load(f"{WORK}/steer_dstart_{L}.npy")
u1 = d_imp / np.linalg.norm(d_imp); u2 = d_start / np.linalg.norm(d_start)
direction = 36.0 * u1 + 14.0 * u2
alpha = float(np.linalg.norm(direction))
print("combined push norm", alpha)
build_edit(L, w, direction, alpha=alpha)
