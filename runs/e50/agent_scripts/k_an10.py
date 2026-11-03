import numpy as np, json, pickle
R = np.load(f"{WORK}/dp_resid.npy")
lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
L2 = json.load(open(f"{WORK}/dp_lab2.json")); pt = np.array(L2["pt"])
sel = isDI | isDD
cos = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
for L in [30]:
    r = R[L].astype(np.float64)
    D = {}
    for t in ["start", "subj", "aux"]:
        a = sel & imp & (pt == t); b = sel & ~imp & (pt == t)
        D[t] = r[a].mean(0) - r[b].mean(0)
        print(L, t, "n_imp", a.sum(), "n_trap", b.sum(), "|d| %.2f" % np.linalg.norm(D[t]))
    dimp = np.load(f"{WORK}/steer_dimp_{L}.npy")
    for t in D: print("  cos(d_%s, d_imp) %.2f" % (t, cos(D[t], dimp)), " cos with start %.2f subj %.2f aux %.2f" % (cos(D[t], D["start"]), cos(D[t], D["subj"]), cos(D[t], D["aux"])))
    # projection of start-type DI-err onto d_start vs DI-ok
    u = D["start"] / np.linalg.norm(D["start"])
    for nm, m in [("DIok", isDI & ~err), ("DIerr", isDI & err), ("DDok", isDD & ~err)]:
        mm = m & (pt == "start")
        print("  start-type %s proj on d_start: %.2f (n=%d)" % (nm, (r[mm] @ u).mean(), mm.sum()))
    np.save(f"{WORK}/steer_dstart_{L}.npy", D["start"]); np.save(f"{WORK}/steer_dsubj_{L}.npy", D["subj"]); np.save(f"{WORK}/steer_daux_{L}.npy", D["aux"])
# residual norm at start positions vs others
r = R[30].astype(np.float64)
for t in ["start", "subj", "aux"]:
    print(t, "resid norm %.1f" % np.linalg.norm(r[pt == t], axis=1).mean())
MB = pickle.load(open(f"{WORK}/margin_orig.pkl", "rb"))
cm = MB["cm"]; m = (MB["cat"] == "DD") & ~MB["err"]
print("DD-ok margin percentiles 5/10/25/50:", np.percentile(cm[m], [5, 10, 25, 50]).round(2))
m = (MB["cat"] == "DI") & MB["err"]
print("DI-err margin percentiles 10/25/50/75/90:", np.percentile(cm[m], [10, 25, 50, 75, 90]).round(2))
