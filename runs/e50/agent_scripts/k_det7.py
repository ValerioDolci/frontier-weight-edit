import numpy as np, json
fam = json.load(open(f"{WORK}/fa_meta.json")); items = fam["items"]; meta = fam["meta"]
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy"); is_prompt = np.load(f"{WORK}/fa_is_prompt.npy")
labp = np.load(f"{WORK}/fa_lab.npy"); wpos = np.load(f"{WORK}/fa_w.npy")
dec_abs = {int(k): v for k, v in json.load(open(f"{WORK}/fa_dec.json"))["dec_abs"].items()}
cat = {p["id"]: p["cat"] for p in practice}
L2 = json.load(open(f"{WORK}/dp_lab2.json")); ptype = dict(zip(L2["ids"], L2["pt"]))
dd = json.load(open(WORK + "/dp.json")); errDI = set(dd["ERR_DI"]); errDD = set(dd["ERR_DD"])
use = labp >= 0
group = np.array([items[i]["id"] if "id" in items[i] else 10000 + i for i in item_of])
rng = np.random.default_rng(1); ug = np.unique(group); fg = dict(zip(ug, rng.integers(0, 5, len(ug))))
fold = np.array([fg[g] for g in group])
iu = item_of[use]; pu = is_prompt[use]; ap = pos_of[use]
kind = np.array(["gen" if "en" in items[i] else cat[items[i]["id"]] for i in iu])
isdec = np.array([dec_abs.get(int(i), -9) == int(p) for i, p in zip(iu, ap)])
pid = np.array([items[i].get("id", -1) for i in iu]); e = np.array([(p in errDI) or (p in errDD) for p in pid])
pt_u = np.array([ptype.get(int(p), "none") if d else "none" for p, d in zip(pid, isdec)])
def cv_scores(X, ypos_mask, wextra=None, theta=0.78, lamf=0.03):
    Xu = X[use].astype(np.float64); y = ypos_mask.astype(np.float64) - theta
    wu = wpos[use].copy()
    if wextra is not None: wu = wu * wextra
    s = np.zeros(len(Xu))
    for f in range(5):
        tr = fold[use] != f
        Xt = Xu[tr]; wt = wu[tr]
        C = (Xt * wt[:, None]).T @ Xt
        lam = lamf * np.trace(C) / C.shape[0]
        wv = np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xt * wt[:, None]).T @ y[tr])
        te = ~tr; s[te] = Xu[te] @ wv
    return s
L = 30
posDI = isdec & (kind == "DI")
s_glob = cv_scores(X30, posDI)
def report(s, tag, t):
    o = tag + f" t={t:+.2f}:"
    for ty in ["start", "subj", "aux"]:
        m = posDI & (pt_u == ty)
        o += f" {ty} DI {np.mean(s[m] > t):.2f} DIerr {np.mean(s[m & e] > t):.2f} |"
    o += f" DDok {np.mean(s[isdec & (kind == 'DD') & ~e] > t):.3f} gen comp {np.mean(s[(kind == 'gen') & ~pu] > t)*1000:.2f} DIprompt {np.mean(s[(kind == 'DI') & pu] > t)*1000:.2f}"
    print(o)
for t in [-0.06, -0.1]:
    report(s_glob, "global", t)
# type-specific detectors: positives only DI decisions of that type; negatives everything else (other-type DI decisions excluded)
S = {}
for ty in ["start", "subj", "aux"]:
    posm = posDI & (pt_u == ty)
    excl = posDI & (pt_u != ty)
    wex = np.where(excl, 0.0, 1.0)
    s = cv_scores(X30, posm, wextra=wex)
    S[ty] = s
    for t in [-0.06, -0.1]:
        m = posm
        print(f"  type {ty} t={t:+.2f}: DI {np.mean(s[m] > t):.2f} DIerr {np.mean(s[m & e] > t):.2f} | DDok {np.mean(s[isdec & (kind == 'DD') & ~e] > t):.3f} | other-type DI {np.mean(s[excl] > t):.2f} | gen comp {np.mean(s[(kind == 'gen') & ~pu] > t)*1000:.2f} DIprompt {np.mean(s[(kind == 'DI') & pu] > t)*1000:.2f} DDprompt {np.mean(s[(kind == 'DD') & pu] > t)*1000:.2f}")
np.save(f"{WORK}/cv_glob30.npy", s_glob)
import pickle; pickle.dump(S, open(f"{WORK}/cv_type30.pkl", "wb"))
