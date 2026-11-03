import json, numpy as np, gc
meta = dict(np.load(f"{WORK}/Xmeta.npz")); grp = meta["grp"]
cfg = json.load(open(f"{WORK}/cfg_E3e.json"))
gelu = lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
DEAD = {}
for L in [31, 34]:
    X = np.load(f"{WORK}/X{L}_f16.npy").astype(np.float32)
    P = f"model.language_model.layers.{L}"
    Wg = W(f"{P}.mlp.gate_proj.weight").numpy(); Wu = W(f"{P}.mlp.up_proj.weight").numpy(); Wd = W(f"{P}.mlp.down_proj.weight").numpy()
    dn = np.linalg.norm(Wd, axis=0)
    mx = np.zeros(Wg.shape[0], np.float32); gmax = np.full(Wg.shape[0], -1e9, np.float32)
    for i0 in range(0, len(X), 4000):
        Xb = X[i0:i0 + 4000]
        G = Xb @ Wg.T; U = Xb @ Wu.T
        A = np.abs(gelu(G) * U)
        mx = np.maximum(mx, A.max(0)); gmax = np.maximum(gmax, G.max(0))
        del G, U, A
    contrib = mx * dn
    old = cfg[f"{L}_Pf"]["neurons"] + cfg[f"{L}_Pm"]["neurons"]
    print(f"L{L}: rows {len(X)}; old slots (max|act|, max gate, |down|, max contrib):",
          [(j, round(float(mx[j]), 4), round(float(gmax[j]), 2), round(float(dn[j]), 2), round(float(contrib[j]), 4)) for j in old])
    order = np.argsort(contrib)
    print(f"L{L}: 12 lowest max-contribution neurons:", [(int(j), round(float(mx[j]), 6), round(float(gmax[j]), 2), round(float(dn[j]), 2), float(contrib[j])) for j in order[:12]])
    print(f"L{L}: median max|act| {np.median(mx):.3f}; #neurons with max contrib < 1e-3: {(contrib < 1e-3).sum()}, < 1e-4: {(contrib < 1e-4).sum()}")
    DEAD[L] = [int(j) for j in order[:12]]
    del X, Wg, Wu, Wd; gc.collect()
json.dump(DEAD, open(f"{WORK}/dead_candidates.json", "w"))
print("VmData", vmdata())
