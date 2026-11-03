import numpy as np, json, pickle
# same train-only directions as k_val_build.py (pass 11), detectors = augmented train-only
R = np.load(f"{WORK}/dp_resid.npy"); lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
L2 = json.load(open(f"{WORK}/dp_lab2.json")); pt = np.array(L2["pt"]); dp_ids = np.array(L2["ids"])
trd = (dp_ids % 3 != 0); sel = (isDI | isDD) & trd
r = R[30].astype(np.float64)
ds = []; wsum = 0
for t in ["start", "subj", "aux"]:
    a = sel & imp & (pt == t); b = sel & ~imp & (pt == t); n = min(a.sum(), b.sum())
    ds.append((r[a].mean(0) - r[b].mean(0)) * n); wsum += n
d_imp30t = sum(ds) / wsum
a = sel & imp & (pt == "start"); b = sel & ~imp & (pt == "start"); d_st30t = r[a].mean(0) - r[b].mean(0)
dir30 = 24.0 * d_imp30t / np.linalg.norm(d_imp30t) + 8.0 * d_st30t / np.linalg.norm(d_st30t)
fam = json.load(open(f"{WORK}/fa_meta.json")); fa_items = fam["items"]
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy")
dec_abs = {int(k): v for k, v in json.load(open(f"{WORK}/fa_dec.json"))["dec_abs"].items()}
cat = {p["id"]: p["cat"] for p in practice}
ME2 = pickle.load(open(f"{WORK}/margin_e2.pkl", "rb")); cm2 = dict(zip(ME2["id"].tolist(), ME2["cm"].tolist()))
ptype = dict(zip(L2["ids"], L2["pt"]))
R40 = np.load(f"{WORK}/fa_resid_40.npy").astype(np.float64)
ds = []; wsum = 0
for t in ["start", "subj"]:
    ij, tj = [], []
    for i, d in dec_abs.items():
        pid = fa_items[i]["id"]; c = cat[pid]
        if c not in ("DI", "DD") or ptype[pid] != t or pid % 3 == 0: continue
        j = np.where((item_of == i) & (pos_of == d))[0][0]
        corr = cm2[pid] > 0; is_imp = corr if c == "DI" else (not corr)
        (ij if is_imp else tj).append(j)
    n = min(len(ij), len(tj)); ds.append((R40[ij].mean(0) - R40[tj].mean(0)) * n); wsum += n
d40t = sum(ds) / wsum
w30 = np.load(f"{WORK}/w30_tr_aug.npy"); w40 = np.load(f"{WORK}/w40_tr_aug.npy")
u30c = np.load(f"{WORK}/u30c.npy"); u40c = np.load(f"{WORK}/u40c.npy")
b2 = json.load(open(f"{WORK}/build2.json"))
build_pair(30, w30, u30c, 1.0, dir30, float(np.linalg.norm(dir30)), [13770, 12872], rms_m=0.158)
build_pair(40, w40, u40c, 1.0, d40t, float(np.linalg.norm(d40t)), b2["n40"], rms_m=0.714)
print("alpha30 %.2f alpha40 %.2f" % (np.linalg.norm(dir30), np.linalg.norm(d40t)))
print(edits())
