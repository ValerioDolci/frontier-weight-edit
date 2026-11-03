import numpy as np
D = dict(np.load(f"{WORK}/dirdata.npz"))
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def dirs(L, first, srcs, reps=20):
    sel = np.isin(D["src"], srcs) & (D["first"] == first)
    Wd = D[f"L{L}"][sel & (D["lab"] == "W")]; Rd = D[f"L{L}"][sel & (D["lab"] == "R")]
    d = Rd.mean(0) - Wd.mean(0)
    rels = []
    for s in range(reps):
        rng = np.random.RandomState(s); iw = rng.permutation(len(Wd)); ir = rng.permutation(len(Rd))
        h1 = Rd[ir[: len(ir)//2]].mean(0) - Wd[iw[: len(iw)//2]].mean(0); h2 = Rd[ir[len(ir)//2:]].mean(0) - Wd[iw[len(iw)//2:]].mean(0)
        rels.append(cos(h1, h2))
    return d, len(Wd), len(Rd), float(np.mean(rels))
for L in [31, 34]:
    for first in [True, False]:
        dp, nw, nr, rp = dirs(L, first, [2]); da, nw2, nr2, ra = dirs(L, first, [2, 3]); dn, nw3, nr3, rn = dirs(L, first, [3])
        print(f"L{L} {'first' if first else 'mid  '}: practice nW={nw} nR={nr} rel={rp:.2f} |d|={np.linalg.norm(dp):.2f} | new nW={nw3} nR={nr3} rel={rn:.2f} | all rel={ra:.2f} | cos(prac,new)={cos(dp,dn):.2f} cos(prac,all)={cos(dp,da):.2f}")
