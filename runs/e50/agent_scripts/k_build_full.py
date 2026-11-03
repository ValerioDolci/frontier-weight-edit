import numpy as np, json
# all-data detectors (prompt negatives), thresholds shifted, same directions as before
t30, t40 = -0.06, -0.03
A30 = float(globals().get("A30", 1.0)); A40 = float(globals().get("A40", 1.0))
w30 = np.load(f"{WORK}/w30_v2.npy"); w40 = np.load(f"{WORK}/w40_v1.npy")
u30c = np.load(f"{WORK}/u30c.npy"); u40c = np.load(f"{WORK}/u40c.npy")
w30g = w30 - t30 * u30c; w40g = w40 - t40 * u40c
d_imp30 = np.load(f"{WORK}/steer_dimp_30.npy"); d_st30 = np.load(f"{WORK}/steer_dstart_30.npy")
dir30 = A30 * (24.0 * d_imp30 / np.linalg.norm(d_imp30) + 8.0 * d_st30 / np.linalg.norm(d_st30))
d40 = np.load(f"{WORK}/steer_d40.npy"); b2 = json.load(open(f"{WORK}/build2.json"))
build_pair(30, w30g, u30c, 1.0, dir30, float(np.linalg.norm(dir30)), [13770, 12872], rms_m=0.158)
build_pair(40, w40g, u40c, 1.0, d40, A40 * b2["alpha40"], b2["n40"], rms_m=0.714)
np.save(f"{WORK}/w30g_full.npy", w30g); np.save(f"{WORK}/w40g_full.npy", w40g)
print("built full edit with A30", A30, "A40", A40, "alpha30", float(np.linalg.norm(dir30)), "alpha40", A40 * b2["alpha40"])
print(edits())
