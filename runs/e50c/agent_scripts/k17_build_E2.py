import numpy as np, torch, json
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
NSYS = 45
SL = {29: None, 30: 0.7227, 31: 0.6993, 32: 0.5546, 33: 0.629, 34: 0.75}
cat = lambda i: "DIW" if i < 55 else ("DIR" if i < 259 else ("DD" if i < 389 else ("DURP" if i < 449 else ("PERF" if i < 509 else "GEN"))))
isfirst = lambda i: dec2[i] == M2[i]["n_prompt_tokens"] - 1

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(neg) > 20000: neg = np.random.RandomState(0).choice(neg, 20000, replace=False)
    return float((pos[:, None] > neg[None, :]).mean())

def build_X(L):
    X, grp, item, row = [], [], [], []
    for i in range(len(I2)):
        xi = mlp_in(L, i).numpy()
        start = 0 if i == 0 else NSYS
        n0 = M2[i]["n_prompt_tokens"]
        for r in range(start, xi.shape[0]):
            if i in dec2 and r == dec2[i] and i < 259:
                g = "Pf" if isfirst(i) else "Pm"
            elif r == n0 - 1:
                g = "FIRSTNEG"          # first-output-token row of a non-target item
            elif i in dec2 and r == dec2[i]:
                g = "DDdec"
            else:
                g = "SYS" if r < NSYS else ("SRC" if r < n0 - 1 else "OUT")
            grp.append(g); item.append(i); row.append(r)
        X.append(xi[start:])
    return np.concatenate(X).astype(np.float32), np.array(grp), np.array(item), np.array(row)

def fit(X, grp, pos, weights, mask):
    y = np.where(grp == pos, 1.0, -1.0).astype(np.float32)
    wts = np.zeros(len(X), np.float32)
    for g, tot in weights.items():
        m = (grp == g) & mask
        if m.sum(): wts[m] = tot / m.sum()
    Xt = X[mask]; wt = wts[mask]; yt = y[mask]
    G = (Xt * wt[:, None]).T @ Xt; b = (Xt * wt[:, None]).T @ yt
    lam = np.trace(G) / G.shape[0] * 1e-2
    return np.linalg.solve(G + lam * np.eye(G.shape[0], dtype=np.float32), b)

DET = {}
for L in [31, 34]:
    X, grp, item, row = build_X(L)
    tr = item % 2 == 0
    for pos, other in [("Pf", "Pm"), ("Pm", "Pf")]:
        W8 = {pos: 1.0, other: 0.3, "FIRSTNEG": 1.0, "DDdec": 1.0, "SYS": 0.2, "SRC": 0.5, "OUT": 1.0}
        w = fit(X, grp, pos, W8, tr)
        s = X @ w; te = ~tr
        sp = s[te & (grp == pos)]
        thr = np.percentile(sp, 15)
        msg = " ".join(f"{g}:{np.mean(s[te & (grp == g)] > thr):.4f}" for g in ["Pf", "Pm", "FIRSTNEG", "DDdec", "OUT", "SRC"] if g != pos)
        print(f"L{L} {pos}: test AUC vs FIRSTNEG={auc(sp, s[te&(grp=='FIRSTNEG')]):.3f} vs DDdec={auc(sp, s[te&(grp=='DDdec')]):.3f} vs OUT={auc(sp, s[te&(grp=='OUT')]):.4f} | p15={thr:.2f} frac>p15: {msg}")
        # final: train on all
        DET[(L, pos)] = fit(X, grp, pos, W8, np.ones(len(X), bool))
    del X
np.save(f"{WORK}/det_E2.npy", {f"{k[0]}_{k[1]}": v for k, v in DET.items()}, allow_pickle=True)
