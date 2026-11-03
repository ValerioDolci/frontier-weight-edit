import numpy as np, json, pickle
fam = json.load(open(f"{WORK}/fa_meta.json")); items = fam["items"]; meta = fam["meta"]
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy"); is_prompt = np.load(f"{WORK}/fa_is_prompt.npy")
labp = np.load(f"{WORK}/fa_lab.npy"); wpos = np.load(f"{WORK}/fa_w.npy")
dec_abs = {int(k): v for k, v in json.load(open(f"{WORK}/fa_dec.json"))["dec_abs"].items()}
cat = {p["id"]: p["cat"] for p in practice}
is_gen_item = np.array(["en" in it for it in items])
use = labp >= 0
group = np.array([items[i]["id"] if "id" in items[i] else 10000 + i for i in item_of])
rng = np.random.default_rng(1); ug = np.unique(group); fg = dict(zip(ug, rng.integers(0, 5, len(ug))))
fold = np.array([fg[g] for g in group])
dd = json.load(open(WORK + "/dp.json")); errDI = set(dd["ERR_DI"]); errDD = set(dd["ERR_DD"])
def cv_report(X, theta=0.78, lamf=0.03, tag=""):
    Xu = X[use].astype(np.float64); yu = (labp[use] == 1).astype(np.float64) - theta; wu = wpos[use]; fu = fold[use]
    s = np.zeros(len(Xu))
    for f in range(5):
        tr = fu != f
        Xt = Xu[tr]; wt = wu[tr]
        C = (Xt * wt[:, None]).T @ Xt
        lam = lamf * np.trace(C) / C.shape[0]
        wv = np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xt * wt[:, None]).T @ yu[tr])
        te = fu == f; s[te] = Xu[te] @ wv
    iu = item_of[use]; pu = is_prompt[use]; ap = pos_of[use]
    kind = np.array(["gen" if is_gen_item[i] else cat[items[i]["id"]] for i in iu])
    isdec = np.array([dec_abs.get(int(i), -9) == int(p) for i, p in zip(iu, ap)])
    pid = np.array([items[i].get("id", -1) for i in iu]); e = np.array([(p in errDI) or (p in errDD) for p in pid])
    DI_ = s[isdec & (kind == "DI")]; DIe = s[isdec & (kind == "DI") & e]; DDo = s[isdec & (kind == "DD") & ~e]; O_ = s[isdec & np.isin(kind, ["PERF", "DURP", "HAB", "COND"])]
    out = f"{tag} CV: DI>0 {(DI_>0).mean():.2f} DIerr>0 {(DIe>0).mean():.2f} DDok>0 {(DDo>0).mean():.3f} other>0 {(O_>0).mean():.3f}"
    for k in ["DI", "DD", "gen"]:
        pm = (kind == k) & pu & ~isdec; cm = (kind == k) & ~pu & ~isdec
        out += f" | {k} prm>0 {np.mean(s[pm]>0)*1000:.2f} cmp>0 {np.mean(s[cm]>0)*1000:.2f}/1000"
    print(out)
    return s
L = 40
X40 = mlp_in(np.load(f"{WORK}/fa_resid_39.npy").astype(np.float32), np.load(f"{WORK}/fa_attn_out_40.npy").astype(np.float32), L)
s40 = cv_report(X40, tag="L40")
x40m = X40.mean(0); u = x40m / np.linalg.norm(x40m); pr = X40 @ u
print("L40 x proj on mean: %.2f +- %.2f" % (pr.mean(), pr.std()))
