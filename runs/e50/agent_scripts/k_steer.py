import numpy as np, json
def P(L, nm): return W(f"model.language_model.layers.{L}.{nm}.weight").numpy()
R = np.load(f"{WORK}/dp_resid.npy"); MO = np.load(f"{WORK}/dp_mlp_out.npy")
lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
L2 = json.load(open(f"{WORK}/dp_lab2.json")); pt = np.array(L2["pt"])
sel = isDI | isDD
for L in [30, 31, 33, 36]:
    r = R[L].astype(np.float64)
    # within-ptype diff of means, imperfetto vs trapassato (DI+DD)
    ds = []; wsum = 0
    for t in ["start", "subj", "aux"]:
        a = sel & imp & (pt == t); b = sel & ~imp & (pt == t)
        if a.sum() > 3 and b.sum() > 3:
            ds.append((r[a].mean(0) - r[b].mean(0)) * min(a.sum(), b.sum())); wsum += min(a.sum(), b.sum())
    d_imp = sum(ds) / wsum
    # within DI: ok - err
    ds = []; wsum = 0
    for t in ["start", "subj", "aux"]:
        a = isDI & ~err & (pt == t); b = isDI & err & (pt == t)
        if a.sum() > 3 and b.sum() > 3:
            ds.append((r[a].mean(0) - r[b].mean(0)) * min(a.sum(), b.sum())); wsum += min(a.sum(), b.sum())
    d_di = sum(ds) / wsum
    # DI - DD (all)
    d_dd = r[isDI].mean(0) - r[isDD].mean(0)
    nr = np.linalg.norm(r, axis=1).mean()
    cos = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    print(f"L{L}: |resid| {nr:.1f}  |d_imp| {np.linalg.norm(d_imp):.2f} |d_di| {np.linalg.norm(d_di):.2f} |d_DIDD| {np.linalg.norm(d_dd):.2f}  cos(imp,di) {cos(d_imp, d_di):.2f} cos(imp,DIDD) {cos(d_imp, d_dd):.2f} cos(di,DIDD) {cos(d_di, d_dd):.2f}")
    # projections of groups on d_imp unit
    u = d_imp / np.linalg.norm(d_imp)
    pr = r @ u
    print("   proj on d_imp: DIok %.1f DIerr %.1f DDok %.1f DDerr %.1f other %.1f (std DIerr %.1f)" % (pr[isDI & ~err].mean(), pr[isDI & err].mean(), pr[isDD & ~err].mean(), pr[isDD & err].mean(), pr[isO].mean(), pr[isDI & err].std()))
    g = P(L, "post_feedforward_layernorm"); sL = W(f"model.language_model.layers.{L}.layer_scalar").item()
    m = MO[L].astype(np.float64); rm = np.sqrt((m ** 2).mean(1))
    print("   mlp_out rms at DI: %.3f  g rms %.2f  s_L %.3f" % (rm[isDI].mean(), np.sqrt((g ** 2).mean()), sL))
    q = d_imp / g
    print("   d_imp/g: rms %.2f max %.1f ; share of |d_imp/g|^2 in dims with g<0.1: %.3f" % (np.sqrt((q**2).mean()), np.abs(q).max(), (q[g < 0.1]**2).sum() / (q**2).sum()))
    np.save(f"{WORK}/steer_dimp_{L}.npy", d_imp); np.save(f"{WORK}/steer_ddi_{L}.npy", d_di); np.save(f"{WORK}/steer_dDIDD_{L}.npy", d_dd)
