import numpy as np, json
u30c, _, _ = const_dir2(X30, 1e-2); u40c, _, _ = const_dir2(X40, 1e-2)
np.save(f"{WORK}/u30c.npy", u30c); np.save(f"{WORK}/u40c.npy", u40c)
w30 = np.load(f"{WORK}/w30_v2.npy"); w40 = np.load(f"{WORK}/w40_v1.npy")
d_imp30 = np.load(f"{WORK}/steer_dimp_30.npy"); d_st30 = np.load(f"{WORK}/steer_dstart_30.npy")
dir30 = 24.0 * d_imp30 / np.linalg.norm(d_imp30) + 8.0 * d_st30 / np.linalg.norm(d_st30)
build_pair(30, w30, u30c, 1.0, dir30, float(np.linalg.norm(dir30)), [13770, 12872], rms_m=0.158)
d40 = np.load(f"{WORK}/steer_d40.npy"); b2 = json.load(open(f"{WORK}/build2.json"))
build_pair(40, w40, u40c, 1.0, d40, b2["alpha40"], b2["n40"], rms_m=0.714)
mult30 = neuron_mult(X30.astype(np.float64), w30, u30c, 1.0); mult40 = neuron_mult(X40.astype(np.float64), w40, u40c, 1.0)
kind = np.array(["gen" if "en" in items[i] else cat[items[i]["id"]] for i in item_of])
for nm, mult in [("L30", mult30), ("L40", mult40)]:
    o = nm
    for k in ["DI", "DD", "gen"]:
        pm = (kind == k) & is_prompt; cmk = (kind == k) & ~is_prompt
        o += f" | {k} prompt>0.1 {np.mean(mult[pm] > 0.1)*1000:.2f} comp>0.1 {np.mean(mult[cmk] > 0.1)*1000:.2f}/1000"
    print(o)
# decision-position multipliers
isdec = np.zeros(len(item_of), bool)
for i, d in dec_abs.items(): isdec[(item_of == i) & (pos_of == d)] = True
for nm, mult in [("L30", mult30), ("L40", mult40)]:
    print(nm, "dec mult: DI mean %.2f frac>0.5 %.2f | DD mean %.3f frac>0.5 %.3f" % (mult[isdec & (kind == "DI")].mean(), (mult[isdec & (kind == "DI")] > 0.5).mean(), mult[isdec & (kind == "DD")].mean(), (mult[isdec & (kind == "DD")] > 0.5).mean()))
print(edits())
