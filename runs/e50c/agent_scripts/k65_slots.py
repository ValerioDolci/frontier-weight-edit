import json, numpy as np, gc
cfg = json.load(open(f"{WORK}/cfg_E3e.json"))
gelu = lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
SLOTS = {}
for L in [31, 34]:
    X = np.load(f"{WORK}/X{L}_f16.npy").astype(np.float32)
    P = f"model.language_model.layers.{L}"
    Wg = W(f"{P}.mlp.gate_proj.weight").numpy(); Wu = W(f"{P}.mlp.up_proj.weight").numpy(); Wd = W(f"{P}.mlp.down_proj.weight").numpy()
    dn = np.linalg.norm(Wd, axis=0); n = Wg.shape[0]
    mx = np.zeros(n, np.float32); s1 = np.zeros(n, np.float64); s2 = np.zeros(n, np.float64)
    for i0 in range(0, len(X), 4000):
        Xb = X[i0:i0 + 4000]
        A = gelu(Xb @ Wg.T) * (Xb @ Wu.T)
        mx = np.maximum(mx, np.abs(A).max(0)); s1 += A.sum(0); s2 += (A.astype(np.float64) ** 2).sum(0)
        del A
    mean = s1 / len(X); rms = np.sqrt(s2 / len(X))
    cmax, crms, cmean = mx * dn, rms * dn, np.abs(mean) * dn
    old = cfg[f"{L}_Pf"]["neurons"] + cfg[f"{L}_Pm"]["neurons"]
    print(f"L{L} old slots (cmax, crms, |cmean|):", [(j, round(float(cmax[j]), 3), round(float(crms[j]), 4), round(float(cmean[j]), 4)) for j in old])
    print(f"L{L} medians over all neurons: cmax {np.median(cmax):.3f} crms {np.median(crms):.4f} |cmean| {np.median(cmean):.4f}")
    score = cmax / np.median(cmax) + crms / np.median(crms)
    order = np.argsort(score)
    print(f"L{L} best 10 by cmax/med + crms/med:", [(int(j), round(float(cmax[j]), 3), round(float(crms[j]), 4), round(float(cmean[j]), 4)) for j in order[:10]])
    SLOTS[L] = [int(j) for j in order[:4]]
    del X, Wg, Wu, Wd; gc.collect()
json.dump(SLOTS, open(f"{WORK}/slots_min.json", "w"))
print("chosen:", SLOTS)
