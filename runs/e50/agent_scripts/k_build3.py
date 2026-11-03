import numpy as np, json
def const_dir2(X, lamf=1e-3):
    Xs = X[np.random.default_rng(0).choice(len(X), min(len(X), 40000), replace=False)].astype(np.float64)
    mu = Xs.mean(0); S = np.cov(Xs, rowvar=False)
    lam = lamf * np.trace(S) / S.shape[0]
    a = np.linalg.solve(S + lam * np.eye(S.shape[0]), mu)
    u = a / (mu @ a)
    pr = X.astype(np.float64) @ u
    return u, float(pr.mean()), float(pr.std())
for lamf in [1e-1, 1e-2, 1e-3]:
    u, m, s = const_dir2(X30, lamf); print("L30 lamf", lamf, "mean %.3f sd %.3f min %.3f max %.3f" % (m, s, (X30 @ u).min(), (X30 @ u).max()))
    u, m, s = const_dir2(X40, lamf); print("L40 lamf", lamf, "mean %.3f sd %.3f min %.3f max %.3f" % (m, s, (X40 @ u).min(), (X40 @ u).max()))
