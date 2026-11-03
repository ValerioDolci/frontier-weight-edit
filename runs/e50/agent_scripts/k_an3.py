import numpy as np, json
R = np.load(f"{WORK}/dp_resid.npy")  # (48,N,d) float16
M = json.load(open(f"{WORK}/dp_meta.json")); items = M["items"]
d = json.load(open(WORK + "/dp.json"))
cat = {p["id"]: p["cat"] for p in practice}
errDI = set(d["ERR_DI"]); errDD = set(d["ERR_DD"])
ids = [it["id"] for it in items]
pref = [it["it"] for it in items]
N = len(ids)
cats = np.array([cat[i] for i in ids])
isDI = cats == "DI"; isDD = cats == "DD"; isO = ~(isDI | isDD)
err = np.array([(i in errDI) or (i in errDD) for i in ids])
# output tense: 1 = imperfetto (DI ok or DD err), 0 = trapassato
imp = np.where(isDI, ~err, err)
def ptype(p):
    if p == "": return "start"
    last = p.split()[-1].lower()
    if last in ("era", "erano", "eravamo", "ero", "eri", "eravate", "eravano"): return "aux"
    return "subj"
pt = np.array([ptype(p) for p in pref])
from collections import Counter
print("ptype counts DI/DD:", Counter(zip(cats[~isO], pt[~isO])))
def auc(s, y):
    s = np.asarray(s); y = np.asarray(y).astype(bool)
    pos = s[y]; neg = s[~y]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
rng = np.random.default_rng(0)
def cv_auc(X, y, lam=None, k=5):
    n = len(y); perm = rng.permutation(n); folds = np.array_split(perm, k)
    s = np.zeros(n)
    X = X - X.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    yy = np.where(y, 1.0, -1.0)
    for f in folds:
        tr = np.setdiff1d(perm, f)
        Xt = X[tr]; G = Xt @ Xt.T
        lam_ = lam if lam is not None else 0.1 * np.trace(G) / len(tr)
        a = np.linalg.solve(G + lam_ * np.eye(len(tr)), yy[tr] - yy[tr].mean())
        w = Xt.T @ a
        s[f] = X[f] @ w
    return auc(s, y)
sel = ~isO
print("layer  AUC(DI vs DD)  AUC(imp vs trap | DI+DD)  AUC(err vs ok | DI)  AUC(DIDD vs other)")
for L in range(0, 48, 2):
    X = R[L].astype(np.float32)
    a1 = cv_auc(X[sel], isDI[sel])
    a2 = cv_auc(X[sel], imp[sel])
    a3 = cv_auc(X[isDI], err[isDI])
    a4 = cv_auc(X, ~isO)
    print(f"{L:3d}  {a1:.3f}  {a2:.3f}  {a3:.3f}  {a4:.3f}")
np.save(f"{WORK}/dp_labels.npy", np.stack([isDI, isDD, isO, err, imp]).astype(np.int8))
json.dump({"pt": pt.tolist(), "ids": ids}, open(f"{WORK}/dp_lab2.json", "w"))
