import numpy as np, json, torch
R = np.load(f"{WORK}/dp_resid.npy"); AO = np.load(f"{WORK}/dp_attn_out.npy")
lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
L2 = json.load(open(f"{WORK}/dp_lab2.json")); pt = np.array(L2["pt"]); ids = L2["ids"]
def auc(s, y):
    s = np.asarray(s); y = np.asarray(y).astype(bool)
    pos = s[y]; neg = s[~y]
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def rms(x, g, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g
rng = np.random.default_rng(0)
def cv_scores(X, y, k=5, lamf=0.1, sel=None):
    n = len(y); perm = rng.permutation(n); folds = np.array_split(perm, k)
    s = np.zeros(n)
    mu = X.mean(0); X = X - mu
    yy = np.where(y, 1.0, -1.0)
    for f in folds:
        tr = np.setdiff1d(perm, f)
        Xt = X[tr]; G = Xt @ Xt.T
        lam_ = lamf * np.trace(G) / len(tr)
        a = np.linalg.solve(G + lam_ * np.eye(len(tr)), yy[tr] - yy[tr].mean())
        w = Xt.T @ a
        s[f] = X[f] @ w
    return s
sel = ~isO
y = isDI[sel]
print("L  | MLP-input probe: AUC(DI vs DD) AUC(DIerr vs DDok) AUC(DIok vs DDok) AUC(DIerr vs DDerr?)  | same on resid")
for L in [20, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 38]:
    g_pa = W(f"model.language_model.layers.{L}.post_attention_layernorm.weight").numpy()
    g_pf = W(f"model.language_model.layers.{L}.pre_feedforward_layernorm.weight").numpy()
    h = R[L - 1].astype(np.float32) + rms(AO[L].astype(np.float32), g_pa)
    x = rms(h, g_pf)
    out = []
    for X in [x, R[L].astype(np.float32)]:
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        s = cv_scores(Xn[sel], y)
        e = err[sel]; di = isDI[sel]; dd = isDD[sel]
        a_all = auc(s, y)
        m1 = (di & e) | (dd & ~e); a_hard = auc(s[m1], di[m1])
        m2 = (di & ~e) | (dd & ~e); a_easy = auc(s[m2], di[m2])
        out.append(f"{a_all:.3f} {a_hard:.3f} {a_easy:.3f}")
    print(L, " | ", out[0], " | ", out[1])
