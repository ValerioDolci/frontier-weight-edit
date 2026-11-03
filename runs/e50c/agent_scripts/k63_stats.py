import json, numpy as np, torch, gc
meta = dict(np.load(f"{WORK}/Xmeta.npz")); grp = meta["grp"]; src = meta["src"]; item = meta["item"]
cfg = json.load(open(f"{WORK}/cfg_E3e.json"))
gen_out = (grp == "OUT") & (src == 2) & (item >= 509)
groups = {"SYS": grp == "SYS", "SRC (English source tokens)": grp == "SRC", "OUT general-text outputs": gen_out,
          "OUT other non-DI outputs": (grp == "OUT") & ~gen_out, "FIRSTNEG (first tokens, non-DI)": grp == "FIRSTNEG",
          "DDdec (DD decision rows)": grp == "DDdec", "Pf (DI first-token decisions)": grp == "Pf", "Pm (DI mid decisions)": grp == "Pm"}
print({k: int(m.sum()) for k, m in groups.items()})
gelu = lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
rng = np.random.RandomState(0)
for L in [31, 34]:
    X = np.load(f"{WORK}/X{L}_f16.npy").astype(np.float32)
    for pos in ["Pf", "Pm"]:
        e = cfg[f"{L}_{pos}"]; w = np.load(f"{WORK}/E3e_w_{L}_{pos}.npy"); s = X @ w
        th, Dl = e["theta"], e["Dl"]
        print(f"L{L} {pos} (theta {th:.3f}, full at {th+Dl:.3f}):")
        for k, m in groups.items():
            print(f"   {k:34s} active {np.mean(s[m] > th):7.4f}  full {np.mean(s[m] > th + Dl):7.4f}  max {s[m].max():6.2f}")
    P = f"model.language_model.layers.{L}"
    Wg = W(f"{P}.mlp.gate_proj.weight").numpy(); Wu = W(f"{P}.mlp.up_proj.weight").numpy(); Wd = W(f"{P}.mlp.down_proj.weight").numpy()
    Xs = X[::4]
    jj = cfg[f"{L}_Pf"]["neurons"] + cfg[f"{L}_Pm"]["neurons"]
    ref = rng.choice(Wg.shape[0], 300, replace=False)
    def act(j): return gelu(Xs @ Wg[j]) * (Xs @ Wu[j])
    mx_ref = np.array([np.abs(act(j)).max() for j in ref])
    print(f"L{L} original activation of repurposed neurons over {len(Xs)} rows (max|act|, |down col|):",
          [(int(j), round(float(np.abs(act(j)).max()), 4), round(float(np.linalg.norm(Wd[:, j])), 3)) for j in jj])
    print(f"L{L} reference: median over 300 random neurons of max|act| = {np.median(mx_ref):.3f}, 5th pct {np.percentile(mx_ref, 5):.3f}")
    del X, Xs, Wg, Wu, Wd; gc.collect()
print("VmData", vmdata())
