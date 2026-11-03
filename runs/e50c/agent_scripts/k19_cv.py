import numpy as np, torch, json
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
NSYS = 45
isfirst = lambda i: dec2[i] == M2[i]["n_prompt_tokens"] - 1

def build_X2(L):
    X, grp, item, row = [], [], [], []
    for i in range(len(I2)):
        xi = mlp_in(L, i).numpy()
        start = 0 if i == 0 else NSYS
        n0 = M2[i]["n_prompt_tokens"]
        dr = dec2.get(i, -99)
        for r in range(start, xi.shape[0]):
            if i < 259 and r == dr:
                g = "Pf" if isfirst(i) else "Pm"
            elif i < 259 and dr - 1 <= r <= dr + 3:
                g = "IGN"
            elif r == n0 - 1:
                g = "FIRSTNEG"
            elif i in dec2 and r == dr:
                g = "DDdec"
            else:
                g = "SYS" if r < NSYS else ("SRC" if r < n0 - 1 else "OUT")
            grp.append(g); item.append(i); row.append(r)
        X.append(xi[start:])
    return np.concatenate(X).astype(np.float32), np.array(grp), np.array(item), np.array(row)

def gram(X, grp, pos, weights, mask):
    y = np.where(grp == pos, 1.0, -1.0).astype(np.float32)
    wts = np.zeros(len(X), np.float32)
    for g, tot in weights.items():
        m = (grp == g) & mask
        if m.sum(): wts[m] = tot / m.sum()
    sel = wts > 0
    Xt = X[sel]; wt = wts[sel]; yt = y[sel]
    return (Xt * wt[:, None]).T @ Xt, (Xt * wt[:, None]).T @ yt

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(neg) > 20000: neg = np.random.RandomState(0).choice(neg, 20000, replace=False)
    return float((pos[:, None] > neg[None, :]).mean())

CV = {}
for L in [31, 34]:
    X, grp, item, row = build_X2(L)
    folds = [item % 2 == 0, item % 2 == 1]
    for pos, other in [("Pf", "Pm"), ("Pm", "Pf")]:
        W8 = {pos: 1.0, other: 0.3, "FIRSTNEG": 1.0, "DDdec": 1.0, "SYS": 0.2, "SRC": 0.5, "OUT": 1.0}
        Gs = [gram(X, grp, pos, W8, f) for f in folds]
        best = None
        for lamf in [1e-2, 1e-1, 1.0, 10.0]:
            s = np.zeros(len(X), np.float32)
            for k in range(2):
                G, b = Gs[k]
                lam = np.trace(G) / G.shape[0] * lamf
                w = np.linalg.solve(G + lam * np.eye(G.shape[0], dtype=np.float32), b)
                s[folds[1 - k]] = X[folds[1 - k]] @ w
            sp = s[grp == pos]
            thr = np.percentile(sp, 20)
            fp = {g: float(np.mean(s[grp == g] > thr)) for g in ["FIRSTNEG", "DDdec", "OUT", "SRC", "SYS", other]}
            a1 = auc(sp, s[grp == ("FIRSTNEG" if pos == "Pf" else "DDdec")]); a2 = auc(sp, s[grp == "OUT"])
            print(f"L{L} {pos} lam={lamf:g}: CV AUC key={a1:.3f} OUT={a2:.4f} | P p20={thr:.3f} p50={np.median(sp):.3f} | FP@p20 " + " ".join(f"{g}={v:.4f}" for g, v in fp.items()))
            score = a1 + a2
            if best is None or score > best[0]:
                best = (score, lamf, s.copy())
        CV[(L, pos)] = (best[1], best[2])
        print(f"  -> chosen lam {best[1]}")
    np.save(f"{WORK}/cv_scores_L{L}.npy", {"grp": grp, "item": item, "row": row, **{f"s_{p}": CV[(L, p)][1] for p in ["Pf", "Pm"]}, "lam": {p: CV[(L, p)][0] for p in ["Pf", "Pm"]}}, allow_pickle=True)
    del X
