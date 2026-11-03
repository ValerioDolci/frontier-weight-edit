import torch, json, numpy as np
A = load_acts("main1")
acts, meta, items = A["acts"], A["meta"], A["items"]
rows = json.load(open(WORK + "/dec_rows.json"))
cat = {p["id"]: p["cat"] for p in practice}
dec = {r["kb"]: r for r in rows if r["t"] >= 1}
# build dataset over all completion positions of the 610 B items
def build(L):
    Xs, ys, ws, grp, kind = [], [], [], [], []
    for kb in range(610):
        R = acts[f"resid.{L}"][kb].float()
        R = R / R.pow(2).mean(1, keepdim=True).sqrt()
        n = R.shape[0]
        i = items[kb]["id"]; c = cat[i]
        dpos = dec[kb]["t"] - 1 if kb in dec else -1
        for p in range(n):
            if p == dpos:
                y = 1.0 if c == "DI" else -1.0
                w = 20.0
                k = "DIdec" if c == "DI" else "DDdec"
                if c == "DI" and not dec[kb]["ok"]: k = "DIerr"
            else:
                y = 0.0; w = 1.0; k = "other"
            Xs.append(R[p]); ys.append(y); ws.append(w); grp.append(kb); kind.append(k)
    return torch.stack(Xs).numpy(), np.array(ys), np.array(ws), np.array(grp), np.array(kind)
for L in [26, 28, 30, 32]:
    X, y, w, g, kind = build(L)
    uniq = np.unique(g); rng = np.random.RandomState(0); rng.shuffle(uniq)
    folds = np.array_split(uniq, 5)
    pred = np.zeros(len(y))
    for f in folds:
        te = np.isin(g, f); tr = ~te
        mu = (X[tr] * w[tr, None]).sum(0) / w[tr].sum()
        Xc = (X[tr] - mu) * np.sqrt(w[tr])[:, None]
        yc = y[tr] * np.sqrt(w[tr])
        Wt = np.linalg.solve(Xc.T @ Xc + 300.0 * np.eye(X.shape[1]), Xc.T @ yc)
        pred[te] = (X[te] - mu) @ Wt
    out = {}
    for k in ["DIerr", "DIdec", "DDdec", "other"]:
        s = pred[kind == k]
        out[k] = np.round(np.quantile(s, [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]) if k == "other" else np.quantile(s, [0, 0.1, 0.5, 0.9, 1]), 2)
    print("L", L)
    for k, v in out.items(): print("   ", k, (kind == k).sum(), v)
    # threshold analysis
    for th in [0.2, 0.3, 0.4, 0.5]:
        print(f"    th {th}: DIerr fire {(pred[kind=='DIerr']>th).mean():.2f}  DIok fire {(pred[kind=='DIdec']>th).mean():.2f}  DD fire {(pred[kind=='DDdec']>th).sum()}  other fire {(pred[kind=='other']>th).sum()} / {(kind=='other').sum()}")
