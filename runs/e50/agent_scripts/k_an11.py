import numpy as np, json
R = np.load(f"{WORK}/dp_resid.npy")
lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
L2 = json.load(open(f"{WORK}/dp_lab2.json")); pt = np.array(L2["pt"])
sel = isDI | isDD
cos = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
for L in range(28, 47, 2):
    r = R[L].astype(np.float64)
    D = {}
    for t in ["start", "subj", "aux"]:
        a = sel & imp & (pt == t); b = sel & ~imp & (pt == t)
        D[t] = r[a].mean(0) - r[b].mean(0)
    nr = np.linalg.norm(r, axis=1).mean()
    # separation (d') of imp vs trap per type along own direction
    sep = {}
    for t in D:
        u = D[t] / np.linalg.norm(D[t]); a = sel & imp & (pt == t); b = sel & ~imp & (pt == t)
        pa = r[a] @ u; pb = r[b] @ u
        sep[t] = (pa.mean() - pb.mean()) / np.sqrt(0.5 * (pa.var() + pb.var()))
    print(f"L{L} |r| {nr:.0f} |d| start {np.linalg.norm(D['start']):.1f} subj {np.linalg.norm(D['subj']):.1f} aux {np.linalg.norm(D['aux']):.1f} | cos st-sb {cos(D['start'], D['subj']):.2f} st-ax {cos(D['start'], D['aux']):.2f} sb-ax {cos(D['subj'], D['aux']):.2f} | d' st {sep['start']:.1f} sb {sep['subj']:.1f} ax {sep['aux']:.1f}")
