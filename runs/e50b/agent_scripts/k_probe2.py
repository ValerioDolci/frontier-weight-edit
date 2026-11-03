import torch, json, numpy as np
A = load_acts("main1")
acts, meta, items = A["acts"], A["meta"], A["items"]
rows = json.load(open(WORK + "/dec_rows.json"))
layers = [20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44]
sel = [r for r in rows if r["t"] >= 1]
y = np.array([1.0 if r["cat"] == "DI" else 0.0 for r in sel])
err = np.array([1 if (r["cat"] == "DI" and not r["ok"]) else 0 for r in sel])
ddok = np.array([1 if (r["cat"] == "DD" and r["ok"]) else 0 for r in sel])
rng = np.random.RandomState(0)
def auc(s, yy):
    pos = s[yy == 1]; neg = s[yy == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
def ridge_cv(X, yv, lam, nf=10, seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(yv)); folds = np.array_split(perm, nf)
    pred = np.zeros(len(yv))
    for f in folds:
        tr = np.setdiff1d(np.arange(len(yv)), f)
        mu = X[tr].mean(0); Xc = X[tr] - mu; ym = yv[tr].mean()
        a = np.linalg.solve(Xc @ Xc.T + lam * np.eye(len(tr)), yv[tr] - ym)
        pred[f] = (X[f] - mu) @ (Xc.T @ a) + ym
    return pred
for L in layers:
    X = np.stack([acts[f"resid.{L}"][r["kb"]][r["t"] - 1].float().numpy() for r in sel])
    X = X / np.sqrt((X ** 2).mean(1, keepdims=True))
    p = ridge_cv(X, y, 100.0)
    # among DI-err vs DD: AUC of the DI-vs-DD probe
    m = (err == 1) | (y == 0)
    a1 = auc(p[m], err[m])
    # probe trained only on DI-err vs DD
    Xs, ys = X[m], err[m].astype(float)
    p2 = ridge_cv(Xs, ys, 100.0)
    a2 = auc(p2, ys)
    # DI-err vs DI-ok (the model's own choice)
    m3 = (y == 1)
    p3 = ridge_cv(X[m3], err[m3].astype(float), 100.0)
    a3 = auc(p3, err[m3].astype(float))
    print(L, "DIvsDD-probe AUC on (DIerr vs DD):", round(a1, 3), "| probe DIerr-vs-DD CV AUC:", round(a2, 3), "| DIerr vs DIok AUC:", round(a3, 3), " n_err", int(err.sum()), "n_dd", int((y == 0).sum()))
