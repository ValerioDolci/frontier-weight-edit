import json, numpy as np, gc
M = json.load(open(f"{WORK}/meta_r2_r3.json")); I2 = M["i2"]
meta = dict(np.load(f"{WORK}/Xmeta.npz")); grp = meta["grp"]; fold = meta["fold"]
cfg = json.load(open(f"{WORK}/cfg_E3b.json"))
PCAT = {p["id"]: p["cat"] for p in practice}
def pid(n):
    s, it = int(meta["src"][n]), int(meta["item"][n])
    if s != 2: return f"s{s}:{it}"
    return f"p{I2[it]['id']}{PCAT.get(I2[it]['id'], '?')}" if "id" in I2[it] else f"gen{it}"
SC = {}
for L in [31, 34]:
    X = np.load(f"{WORK}/X{L}_f16.npy").astype(np.float32)
    for pos, other in [("Pf", "Pm")]:
        lamf = cfg[f"{L}_{pos}"]["lam"]; th = cfg[f"{L}_{pos}"]["theta"]; Dl = cfg[f"{L}_{pos}"]["Dl"]
        s_cv = np.zeros(len(X), np.float32)
        for k in range(2):
            w_k = solve(*gram_w(X, grp, pos, other, fold == k), lamf)
            s_cv[fold == 1 - k] = X[fold == 1 - k] @ w_k
        w = np.load(f"{WORK}/E3b_w_{L}_{pos}.npy"); s_in = X @ w
        SC[L] = (s_cv, s_in, th, Dl)
        idx = np.where(grp == pos)[0]
        print(f"L{L} {pos} theta {th:.3f} top {th+Dl:.3f}")
        lst = sorted([(float(s_cv[n]), pid(n)) for n in idx])
        print("  lowest CV scores (Pf rows):", [(round(a, 2), b) for a, b in lst[:16]])
        fn = np.where(grp == "FIRSTNEG")[0]
        lst2 = sorted([(float(s_cv[n]), pid(n)) for n in fn], reverse=True)
        print("  highest CV FIRSTNEG:", [(round(a, 2), b) for a, b in lst2[:16]])
        lst3 = sorted([(float(s_in[n]), pid(n)) for n in fn], reverse=True)
        print("  highest in-sample FIRSTNEG (final w):", [(round(a, 2), b) for a, b in lst3[:10]])
    del X; gc.collect()
for want in [414, 464, 3, 78, 8, 92, 294, 382, 466, 47]:
    ns = [n for n in np.where(grp == "Pf")[0] if meta["src"][n] == 2 and I2[int(meta["item"][n])].get("id") == want]
    for n in ns:
        print(f"p{want}: L31 cv {SC[31][0][n]:.2f} in {SC[31][1][n]:.2f} | L34 cv {SC[34][0][n]:.2f} in {SC[34][1][n]:.2f}")
    if not ns: print(f"p{want}: no Pf row")
