import numpy as np, json, pickle
fam = json.load(open(f"{WORK}/fa_meta.json")); fa_items = fam["items"]
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy"); is_prompt = np.load(f"{WORK}/fa_is_prompt.npy")
labp = np.load(f"{WORK}/fa_lab.npy"); wpos = np.load(f"{WORK}/fa_w.npy")
dec_abs = {int(k): v for k, v in json.load(open(f"{WORK}/fa_dec.json"))["dec_abs"].items()}
cat = {p["id"]: p["cat"] for p in practice}
item_pid = np.array([it.get("id", -1) if "id" in it else -1 for it in fa_items])
is_val_item = np.array([(p >= 0) and (p % 3 == 0) for p in item_pid])
pos_val = is_val_item[item_of]
use_tr = (labp >= 0) & ~pos_val
def fit_sel(X, sel, theta=0.78, lamf=0.03):
    Xu = X[sel].astype(np.float64); yu = (labp[sel] == 1).astype(np.float64) - theta; wu = wpos[sel]
    C = (Xu * wu[:, None]).T @ Xu
    lam = lamf * np.trace(C) / C.shape[0]
    return np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xu * wu[:, None]).T @ yu)
w30t = fit_sel(X30, use_tr); w40t = fit_sel(X40, use_tr)
u30c = np.load(f"{WORK}/u30c.npy"); u40c = np.load(f"{WORK}/u40c.npy")
t30, t40 = -0.06, -0.03
w30g = w30t - t30 * u30c; w40g = w40t - t40 * u40c   # gate score = w.x - t  (u.x ~ 1)
# steering directions from TRAIN items only
R = np.load(f"{WORK}/dp_resid.npy"); lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
L2 = json.load(open(f"{WORK}/dp_lab2.json")); pt = np.array(L2["pt"]); dp_ids = np.array(L2["ids"])
trd = (dp_ids % 3 != 0)
sel = (isDI | isDD) & trd
r = R[30].astype(np.float64)
ds = []; ws = 0
for t in ["start", "subj", "aux"]:
    a = sel & imp & (pt == t); b = sel & ~imp & (pt == t); n = min(a.sum(), b.sum())
    ds.append((r[a].mean(0) - r[b].mean(0)) * n); ws += n
d_imp30t = sum(ds) / ws
a = sel & imp & (pt == "start"); b = sel & ~imp & (pt == "start"); d_st30t = r[a].mean(0) - r[b].mean(0)
dir30 = 24.0 * d_imp30t / np.linalg.norm(d_imp30t) + 8.0 * d_st30t / np.linalg.norm(d_st30t)
print("cos with full-data dirs: imp %.3f start %.3f" % (d_imp30t @ np.load(f"{WORK}/steer_dimp_30.npy") / np.linalg.norm(d_imp30t) / np.linalg.norm(np.load(f"{WORK}/steer_dimp_30.npy")), d_st30t @ np.load(f"{WORK}/steer_dstart_30.npy") / np.linalg.norm(d_st30t) / np.linalg.norm(np.load(f"{WORK}/steer_dstart_30.npy"))))
# d40 from train items (fa data under edit2), labels via margin_e2
ME2 = pickle.load(open(f"{WORK}/margin_e2.pkl", "rb")); cm2 = dict(zip(ME2["id"].tolist(), ME2["cm"].tolist()))
dd = json.load(open(WORK + "/dp.json")); errDI = set(dd["ERR_DI"]); errDD = set(dd["ERR_DD"])
ptype = dict(zip(L2["ids"], L2["pt"]))
R40 = np.load(f"{WORK}/fa_resid_40.npy").astype(np.float64)
ds = []; ws = 0
for t in ["start", "subj"]:
    ij, tj = [], []
    for i, d in dec_abs.items():
        pid = fa_items[i]["id"]; c = cat[pid]
        if c not in ("DI", "DD") or ptype[pid] != t or pid % 3 == 0: continue
        j = np.where((item_of == i) & (pos_of == d))[0][0]
        corr = cm2[pid] > 0; is_imp = corr if c == "DI" else (not corr)
        (ij if is_imp else tj).append(j)
    n = min(len(ij), len(tj)); ds.append((R40[ij].mean(0) - R40[tj].mean(0)) * n); ws += n
d40t = sum(ds) / ws
alpha40 = float(np.linalg.norm(d40t)); print("alpha40", alpha40)
build_pair(30, w30g, u30c, 1.0, dir30, float(np.linalg.norm(dir30)), [13770, 12872], rms_m=0.158)
b2 = json.load(open(f"{WORK}/build2.json"))
build_pair(40, w40g, u40c, 1.0, d40t, alpha40, b2["n40"], rms_m=0.714)
# check val firing (L30 exact; L40 approx under edit2 activations)
m30 = neuron_mult(X30.astype(np.float64), w30g, u30c, 1.0); m40 = neuron_mult(X40.astype(np.float64), w40g, u40c, 1.0)
isdec = np.zeros(len(item_of), bool)
for i, d in dec_abs.items(): isdec[(item_of == i) & (pos_of == d)] = True
kindp = np.array(["gen" if "en" in it else cat[it["id"]] for it in fa_items])[item_of]
pidp = item_pid[item_of]; ep = np.array([(p in errDI) or (p in errDD) for p in pidp])
for nm, mm in [("L30", m30), ("L40", m40)]:
    for split, sm in [("train", ~pos_val), ("val", pos_val)]:
        o = f"{nm} {split}:"
        for k in ["DI", "DD"]:
            for e in [False, True]:
                q = isdec & sm & (kindp == k) & (ep == e)
                o += f" {k}{'err' if e else 'ok'} {np.mean(mm[q] > 0.5):.2f}(n={q.sum()})"
        print(o)
vids = sorted([int(p) for p in item_pid if p >= 0 and p % 3 == 0])
rest = [256, 279, 281, 287, 296, 302, 306, 307, 311, 315, 319, 329, 331, 332, 338, 346, 356, 361, 363, 368, 370, 374, 378, 386, 387, 393, 404, 407, 410, 412, 417, 420, 422, 425, 429, 431, 433, 435, 440, 449, 450, 451, 452, 455, 457, 458, 462, 467, 469, 470, 475, 476, 477, 480, 484, 486, 487, 488, 490, 492, 494, 496, 499, 503, 514, 519, 520, 522, 526, 527, 533, 534, 542, 543, 544, 552, 553, 557, 558, 567, 569, 572, 575, 576, 577, 578, 581, 584, 585, 591, 597, 599, 601, 603, 604, 605, 608, 609]
allv = vids + rest
from collections import Counter
print("val counts:", Counter(cat[i] for i in allv), len(allv))
print(json.dumps(allv))
np.save(f"{WORK}/val_ids.npy", np.array(allv))
