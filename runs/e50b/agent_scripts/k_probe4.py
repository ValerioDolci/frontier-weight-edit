import torch, json, numpy as np, collections
A = load_acts("main1")
acts, meta, items = A["acts"], A["meta"], A["items"]
rows = json.load(open(WORK + "/dec_rows.json"))
cat = {p["id"]: p["cat"] for p in practice}
dec = {r["kb"]: r for r in rows if r["t"] >= 1}
L = 28
Xs, ys, ws, info = [], [], [], []
for kb in range(610):
    R = acts[f"resid.{L}"][kb].float()
    R = R / R.pow(2).mean(1, keepdim=True).sqrt()
    i = items[kb]["id"]; c = cat[i]
    dpos = dec[kb]["t"] - 1 if kb in dec else -1
    toks = meta[kb]["tokens"]
    for p in range(R.shape[0]):
        if p == dpos:
            y = 1.0 if c == "DI" else -1.0; w = 20.0
        else:
            y = 0.0; w = 1.0
        Xs.append(R[p]); ys.append(y); ws.append(w)
        nxt = toks[p + 1] if p + 1 < len(toks) else "<END>"
        info.append((kb, i, c, p, p - dpos if dpos >= 0 else None, toks[p], nxt, p == dpos))
X = torch.stack(Xs).numpy(); y = np.array(ys); w = np.array(ws)
g = np.array([t[0] for t in info])
uniq = np.unique(g); rng = np.random.RandomState(0); rng.shuffle(uniq)
folds = np.array_split(uniq, 5)
pred = np.zeros(len(y))
for f in folds:
    te = np.isin(g, f); tr = ~te
    mu = (X[tr] * w[tr, None]).sum(0) / w[tr].sum()
    Xc = (X[tr] - mu) * np.sqrt(w[tr])[:, None]; yc = y[tr] * np.sqrt(w[tr])
    Wt = np.linalg.solve(Xc.T @ Xc + 300.0 * np.eye(X.shape[1]), Xc.T @ yc)
    pred[te] = (X[te] - mu) @ Wt
th = 0.3
fp = [(pred[j], info[j]) for j in range(len(y)) if pred[j] > th and not info[j][7]]
print("false pos", len(fp))
print(collections.Counter(t[1][2] for t in fp))
print("rel pos to decision (DI/DD items):", collections.Counter(t[1][4] for t in fp if t[1][4] is not None).most_common(12))
fp.sort(key=lambda t: -t[0])
for s, t in fp[:60]:
    print(round(float(s), 2), t[2], t[1], "p", t[3], "rel", t[4], "tok", repr(t[5]), "-> next", repr(t[6]))
