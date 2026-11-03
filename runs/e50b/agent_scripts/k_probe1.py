import torch, json, numpy as np
A = load_acts("main1")
acts, meta, items = A["acts"], A["meta"], A["items"]
rows = json.load(open(WORK + "/dec_rows.json"))
layers = [12, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 47]
sel = [r for r in rows if r["t"] >= 1]
y = np.array([1.0 if r["cat"] == "DI" else 0.0 for r in sel])
print("n", len(sel), "DI", int(y.sum()), "DD", int((1 - y).sum()))
rng = np.random.RandomState(0)
perm = rng.permutation(len(sel)); folds = np.array_split(perm, 5)
def ridge_cv(X, y, lam):
    pred = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        mu = X[tr].mean(0); Xc = X[tr] - mu
        ym = y[tr].mean()
        # dual form since n << d
        K = Xc @ Xc.T
        a = np.linalg.solve(K + lam * np.eye(len(tr)), y[tr] - ym)
        w = Xc.T @ a
        pred[f] = (X[f] - mu) @ w + ym
    return pred
def auc(s, y):
    pos = s[y == 1]; neg = s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
DIR = {}
for L in layers:
    X = np.stack([acts[f"resid.{L}"][r["kb"]][r["t"] - 1].float().numpy() for r in sel])
    X = X / np.sqrt((X ** 2).mean(1, keepdims=True))  # rms normalize
    res = []
    for lam in [1e1, 1e2, 1e3]:
        p = ridge_cv(X, y, lam)
        acc = ((p > 0.5) == (y > 0.5)).mean()
        res.append((lam, round(float(acc), 3), round(auc(p, y), 3)))
    # diff of means direction CV
    pred = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        pred[f] = X[f] @ d
    print(L, "ridge(lam,acc,auc):", res, " meandiff AUC:", round(auc(pred, y), 3))
