import numpy as np
D = "./censura"
a = np.load(f"{D}/ext/contrib_diff_A.npy").ravel(); b = np.load(f"{D}/ext_foreign/contrib_diff_A.npy").ravel()
d1 = np.load(f"{D}/ext/dir_L49.npy"); d2 = np.load(f"{D}/ext_foreign/dir_L52.npy")
R = np.load(f"{D}/cap/resid.npy", mmap_mode="r")
print("coseno direzioni (L49 PRC vs L52 estera):", round(float(d1 @ d2), 3))
for k in (100, 500, 2000):
    ta = set(np.argsort(-a)[:k]); tb = set(np.argsort(-b)[:k]); print(f"top-{k} neuroni in comune PRC vs estera: {len(ta & tb)} (atteso {k*k/a.size:.2f})")
print("Spearman contributi su tutti i neuroni con |diff|>0:", round(float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]), 3))
