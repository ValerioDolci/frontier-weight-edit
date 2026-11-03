import numpy as np, torch, json
r1 = load_acts("r1")
meta, items, acts = r1["meta"], r1["items"], r1["acts"]
LAYERS = sorted(int(k.split(".")[1]) for k in acts)
pairs = json.load(open(f"{WORK}/pairs_r1.json"))
dec = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r1.json")).items()}
DIW = [(i, dec[i][1]) for i in range(0, 55)]
DIR = [(i, dec[i][1]) for i in range(110, 314) if i in dec]
DDP = [(p["a"], p["drow"], p["id"], p["lp_imp"] - p["lp_trap"]) for p in pairs if p["kind"] == "DD"]
print(len(DIW), len(DIR), len(DDP))

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

def stack(L, lst):
    return torch.stack([acts[f"resid.{L}"][i][r].float() for (i, r, *rest) in lst]).numpy()

res = {}
for L in LAYERS:
    A = stack(L, DIR); B = stack(L, DIW); D = stack(L, DDP)
    d = A.mean(0) - B.mean(0)
    u = d / np.linalg.norm(d)
    nr = np.linalg.norm(np.concatenate([A, B]), axis=1).mean()
    pa, pb, pd = A @ u, B @ u, D @ u
    # CV: split halves
    rng = np.random.RandomState(0)
    ia = rng.permutation(len(A)); ib = rng.permutation(len(B))
    a1, a2 = ia[: len(A)//2], ia[len(A)//2:]; b1, b2 = ib[: len(B)//2], ib[len(B)//2:]
    d1 = A[a1].mean(0) - B[b1].mean(0); u1 = d1 / np.linalg.norm(d1)
    cv = auc(A[a2] @ u1, B[b2] @ u1)
    # shrinkage LDA on half 1, test on half 2
    X1 = np.concatenate([A[a1] - A[a1].mean(0), B[b1] - B[b1].mean(0)])
    S = X1.T @ X1 / len(X1)
    lam = np.trace(S) / S.shape[0]
    w = np.linalg.solve(S + lam * np.eye(S.shape[0]), d1)
    cv_lda = auc(A[a2] @ w, B[b2] @ w)
    res[L] = dict(norm=nr, dnorm=float(np.linalg.norm(d)), auc=auc(pa, pb), cv=cv, cv_lda=cv_lda)
    # DD pairs projection relative to DIW/DIR means
    mA, mB = pa.mean(), pb.mean()
    zd = (pd - mB) / (mA - mB)
    print(f"L{L:2d} |r|={nr:7.2f} |d|={np.linalg.norm(d):6.2f} AUC={auc(pa,pb):.3f} cvAUC={cv:.3f} cvLDA={cv_lda:.3f}  DIW z=0 DIR z=1  DDP z mean={zd.mean():.2f} (DD imp-preferring: {np.mean([z for z,(_,_,_,m) in zip(zd,DDP) if m>0]):.2f}, trap-pref: {np.mean([z for z,(_,_,_,m) in zip(zd,DDP) if m<=0]):.2f})")
