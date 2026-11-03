import numpy as np, json, pickle
MB = pickle.load(open(f"{WORK}/margin_orig.pkl", "rb")); ME = pickle.load(open(f"{WORK}/margin_e1.pkl", "rb"))
L2 = json.load(open(f"{WORK}/dp_lab2.json")); pt = dict(zip(L2["ids"], L2["pt"]))
dpm = json.load(open(f"{WORK}/dp_meta.json")); dp_ids = [it["id"] for it in dpm["items"]]
L = 30
dpR = np.load(f"{WORK}/dp_resid.npy"); dpA = np.load(f"{WORK}/dp_attn_out.npy")
Xdp = mlp_in(dpR[L-1].astype(np.float32), dpA[L].astype(np.float32), L).astype(np.float64)
w = np.load(f"{WORK}/det_w2_L{L}.npy")
gm = neuron_out(Xdp, L, w)
gm_of = dict(zip(dp_ids, gm)); s_of = dict(zip(dp_ids, Xdp @ w))
d = ME["cm"] - MB["cm"]
for t in ["start", "subj", "aux"]:
    for c in ["DI"]:
        m = (ME["cat"] == c) & np.array([pt[i] == t for i in ME["id"]])
        print(t, c, "n", m.sum(), "mean d(cm) %.2f" % d[m].mean(), "err n", (m & ME["err"]).sum(), "err mean d %.2f" % d[m & ME["err"]].mean())
m = (ME["cat"] == "DI") & ME["err"]
rows = sorted([(round(float(d[k]), 2), int(ME["id"][k]), pt[int(ME["id"][k])], round(float(gm_of[int(ME["id"][k])]), 2), round(float(s_of[int(ME["id"][k])]), 3)) for k in np.where(m)[0]])
for r in rows: print(r)
