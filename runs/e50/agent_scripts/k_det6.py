import numpy as np, json
# CV curve for L30 detector (prompt negatives), various thresholds t on score (s = w.x ~ y - 0.78)
fam = json.load(open(f"{WORK}/fa_meta.json")); items = fam["items"]; meta = fam["meta"]
L = 30
Xu = X30[use].astype(np.float64); yu = (labp[use] == 1).astype(np.float64) - 0.78; wu = wpos[use]
fu = fold[use]
s = np.zeros(len(Xu))
for f in range(5):
    tr = fu != f
    Xt = Xu[tr]; wt = wu[tr]
    C = (Xt * wt[:, None]).T @ Xt
    lam = 0.03 * np.trace(C) / C.shape[0]
    wv = np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xt * wt[:, None]).T @ yu[tr])
    te = fu == f; s[te] = Xu[te] @ wv
iu = item_of[use]; pu = is_prompt[use]; ap = pos_of[use]
kind = np.array(["gen" if "en" in items[i] else cat[items[i]["id"]] for i in iu])
isdec = np.array([dec_abs.get(int(i), -9) == int(p) for i, p in zip(iu, ap)])
pid = np.array([items[i].get("id", -1) for i in iu]); e = np.array([(p in errDI) or (p in errDD) for p in pid])
np.save(f"{WORK}/cv_s30.npy", s)
for t in [0.0, -0.03, -0.06, -0.1, -0.15]:
    DI_ = s[isdec & (kind == "DI")] > t; DIe = s[isdec & (kind == "DI") & e] > t; DDo = s[isdec & (kind == "DD") & ~e] > t
    gp = s[(kind == "gen") & pu] > t; gc = s[(kind == "gen") & ~pu] > t; dip = s[(kind == "DI") & pu] > t; ddp = s[(kind == "DD") & pu] > t
    print(f"t={t:+.2f}: DI {DI_.mean():.2f} DIerr {DIe.mean():.2f} DDok {DDo.mean():.3f} | gen prompt {gp.mean()*1000:.2f} gen comp {gc.mean()*1000:.2f} DI prompt {dip.mean()*1000:.2f} DD prompt {ddp.mean()*1000:.2f} /1000")
