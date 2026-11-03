import numpy as np, json
W_R3 = {24, 28, 29, 31, 34, 37, 39, 41, 47, 49, 50, 51, 64, 69, 76, 80, 81, 82, 83, 84, 89, 104, 110}
SKIP_R3 = {20}
D = dict(np.load(f"{WORK}/dirdata.npz"))
lab2 = D["lab"].copy()
keep = np.ones(len(lab2), bool)
for n in range(len(lab2)):
    if D["src"][n] == 3:
        k = int(D["item"][n])
        if k in SKIP_R3: keep[n] = False
        else: lab2[n] = "W" if k in W_R3 else "R"
print("r3 rows:", int((D["src"] == 3).sum()), "relabeled W:", int(((D["src"] == 3) & (lab2 == "W") & keep).sum()), "R:", int(((D["src"] == 3) & (lab2 == "R") & keep).sum()))
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def est(L, first, srcs, lab, reps=30):
    sel = keep & np.isin(D["src"], srcs) & (D["first"] == first)
    Wd = D[f"L{L}"][sel & (lab == "W")]; Rd = D[f"L{L}"][sel & (lab == "R")]
    d = Rd.mean(0) - Wd.mean(0)
    rels = []
    for s in range(reps):
        rng = np.random.RandomState(s); iw = rng.permutation(len(Wd)); ir = rng.permutation(len(Rd))
        h1 = Rd[ir[: len(ir)//2]].mean(0) - Wd[iw[: len(iw)//2]].mean(0); h2 = Rd[ir[len(ir)//2:]].mean(0) - Wd[iw[len(iw)//2:]].mean(0)
        rels.append(cos(h1, h2))
    return d, len(Wd), len(Rd), float(np.mean(rels))
NEWD = {}
for L in [31, 34]:
    for first in [True, False]:
        dp, a, b, rp = est(L, first, [2], lab2)
        dn, c, e, rn = est(L, first, [3], lab2)
        da, f, g, ra = est(L, first, [2, 3], lab2)
        print(f"L{L} {'first' if first else 'mid  '}: practice W={a} R={b} rel={rp:.2f} | r3(gen-labels) W={c} R={e} rel={rn:.2f} cos(prac,r3)={cos(dp,dn):.2f} | combined rel={ra:.2f} cos(prac,comb)={cos(dp,da):.2f} |d| {np.linalg.norm(dp):.2f}->{np.linalg.norm(da):.2f}")
        NEWD[f"{L}_{'Pf' if first else 'Pm'}"] = da
np.save(f"{WORK}/dirs_combined_gen.npy", NEWD, allow_pickle=True)
