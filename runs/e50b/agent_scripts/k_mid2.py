import torch, json, numpy as np
A = load_acts("main1"); acts, meta, items = A["acts"], A["meta"], A["items"]
rows = json.load(open(WORK + "/dec_rows.json"))
cat = {p["id"]: p["cat"] for p in practice}
AXp = torch.load(WORK + "/L47aux.pt"); gapall = (AXp["maxaux"] - AXp["maxnon"]).numpy()
P = torch.load(WORK + "/L47pos.pt"); pidx = {tuple(p): j for j, p in enumerate(P["POS"])}
# aux-top-1 decision positions with t>=1 (available in main1 completion rows)
sel = []
for r in rows:
    if r["t"] < 1: continue
    j = pidx[(r["kb"], r["t"])]
    if gapall[j] <= 0: continue
    sel.append((r, j))
print("aux-top1 decision positions (t>=1):", len(sel), "DI-err:", sum(1 for r, j in sel if r["cat"] == "DI"), "DD:", sum(1 for r, j in sel if r["cat"] == "DD"))
def auc(s, y):
    pos = s[y == 1]; neg = s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean())
for L in [20, 24, 26, 28, 30, 32, 36, 40, 44, 47]:
    # train probe on ALL decision positions (t>=1) DI vs DD + other positions, grouped CV; evaluate on aux-top1 subset
    Xs, ys, ws, g, isdec, rr = [], [], [], [], [], []
    dec = {r["kb"]: r for r in rows if r["t"] >= 1}
    for kb in range(610):
        R = acts[f"resid.{L}"][kb].float(); R = R / R.pow(2).mean(1, keepdim=True).sqrt()
        c = cat[items[kb]["id"]]; dp = dec[kb]["t"] - 1 if kb in dec else -1
        for p in range(R.shape[0]):
            if p == dp:
                y = 1.0 if c == "DI" else -0.5; w = 10.0
            else:
                y = 0.0; w = 1.0
            Xs.append(R[p]); ys.append(y); ws.append(w); g.append(kb); isdec.append(p == dp)
    X = torch.stack(Xs).numpy(); y = np.array(ys); w = np.array(ws); g = np.array(g); isdec = np.array(isdec)
    uniq = np.unique(g); rng = np.random.RandomState(0); rng.shuffle(uniq); folds = np.array_split(uniq, 5)
    pred = np.zeros(len(y))
    for f in folds:
        te = np.isin(g, f); tr = ~te
        sw = np.sqrt(w[tr]); Xw = X[tr] * sw[:, None]; yw = y[tr] * sw
        wv = np.linalg.solve(Xw.T @ Xw + 100.0 * np.eye(X.shape[1]), Xw.T @ yw)
        pred[te] = X[te] @ wv
    # map decision rows
    decpred = {}
    off = 0
    for kb in range(610):
        n = acts[f"resid.{L}"][kb].shape[0]
        if kb in dec: decpred[kb] = pred[off + dec[kb]["t"] - 1]
        off += n
    s = np.array([decpred[r["kb"]] for r, j in sel]); yy = np.array([1 if r["cat"] == "DI" else 0 for r, j in sel])
    # recall of DI-err at fixed DD false-positive count
    sd = np.sort(s[yy == 0])[::-1]
    rec = {k: int((s[yy == 1] > sd[k]).sum()) for k in [0, 2, 4, 8]}
    print(f"L{L}: AUC DIerr-vs-DD (aux-top1) {auc(s, yy):.3f}; DIerr recall at #DD-FP=0/2/4/8: {rec}  (nDIerr={int(yy.sum())})")
