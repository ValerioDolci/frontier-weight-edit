import numpy as np, torch, json, gc
def build_E3b(kappa_f, kappa_m, ramp_f=(15, 50), ramp_m=(10, 50), lam_fixed=None, dir_srcs=(2,), tag="E3b", apply=True):
    meta = dict(np.load(f"{WORK}/Xmeta.npz")); grp = meta["grp"]; fold = meta["fold"]
    D = dict(np.load(f"{WORK}/dirdata.npz"))
    cfg_old = json.load(open(f"{WORK}/cfg_E2a.json"))
    CFG = {}
    for L in [31, 34]:
        X = np.load(f"{WORK}/X{L}_f16.npy").astype(np.float32)
        Xc = X[::3]; Gc = Xc.T @ Xc / len(Xc); bc = Xc.mean(0)
        c = np.linalg.solve(Gc + np.trace(Gc) / Gc.shape[0] * 1e-3 * np.eye(3840, dtype=np.float32), bc).astype(np.float32)
        del Xc, Gc
        P = f"model.language_model.layers.{L}"
        Wg = W(f"{P}.mlp.gate_proj.weight"); Wu = W(f"{P}.mlp.up_proj.weight"); Wd = W(f"{P}.mlp.down_proj.weight")
        wpf = LN[L]["pf"].numpy(); rms_dec = {31: 0.151, 34: 0.389}[L]
        for pos, other, kap, ramp in [("Pf", "Pm", kappa_f, ramp_f), ("Pm", "Pf", kappa_m, ramp_m)]:
            lamf = lam_fixed[(L, pos)]
            folds = [fold == 0, fold == 1]
            s_cv = np.zeros(len(X), np.float32)
            for k in range(2):
                w_k = solve(*gram_w(X, grp, pos, other, folds[k]), lamf)
                s_cv[folds[1 - k]] = X[folds[1 - k]] @ w_k
            sp = s_cv[grp == pos]
            theta = float(np.percentile(sp, ramp[0])); top = float(np.percentile(sp, ramp[1])); Dl = max(top - theta, 0.1)
            fp = {g: round(float(np.mean(s_cv[grp == g] > theta)), 4) for g in ["FIRSTNEG", "DDdec", "OUT", "SRC"]}
            w = solve(*gram_w(X, grp, pos, other, np.ones(len(X), bool)), lamf)
            sys_max = float((X[grp == "SYS"] @ w).max())
            sel = np.isin(D["src"], list(dir_srcs)) & (D["first"] == (pos == "Pf"))
            d = D[f"L{L}"][sel & (D["lab"] == "R")].mean(0) - D[f"L{L}"][sel & (D["lab"] == "W")].mean(0)
            v = rms_dec * kap * d / SL[L] / np.maximum(wpf, 1e-6); v[wpf < 0.1] = 0
            a = 8.0 / Dl; b = 1.0 / (a * Dl)
            jj = cfg_old[f"{L}_{pos}"]["neurons"]
            Wg[jj[0]] = torch.from_numpy(a * (w - theta * c)); Wg[jj[1]] = torch.from_numpy(a * (w - (theta + Dl) * c))
            Wu[jj[0]] = torch.from_numpy(b * c); Wu[jj[1]] = torch.from_numpy(b * c)
            Wd[:, jj[0]] = torch.from_numpy(v.astype(np.float32)); Wd[:, jj[1]] = torch.from_numpy(-v.astype(np.float32))
            CFG[f"{L}_{pos}"] = dict(theta=theta, Dl=Dl, lam=lamf, neurons=jj, kappa=kap, sys_max=sys_max, fp_cv=fp, dnorm=float(np.linalg.norm(d)), vnorm=float(np.linalg.norm(v)))
            np.save(f"{WORK}/{tag}_w_{L}_{pos}.npy", w); np.save(f"{WORK}/{tag}_d_{L}_{pos}.npy", d)
            print(f"L{L} {pos}: lam {lamf} theta {theta:.3f} top {theta+Dl:.3f} SYSmax {sys_max:.3f} CV-FP {fp} |d|={np.linalg.norm(d):.2f} |v|={np.linalg.norm(v):.2f}")
        np.save(f"{WORK}/{tag}_c_{L}.npy", c)
        if apply:
            save_edit(f"{P}.mlp.gate_proj.weight", Wg); save_edit(f"{P}.mlp.up_proj.weight", Wu); save_edit(f"{P}.mlp.down_proj.weight", Wd)
        del X, Wg, Wu, Wd; gc.collect()
    json.dump(CFG, open(f"{WORK}/cfg_{tag}.json", "w"))
    return CFG
LAMF = {(31, "Pf"): 0.01, (31, "Pm"): 0.01, (34, "Pf"): 0.01, (34, "Pm"): 0.1}
CFG_E3b = build_E3b(3.0, 3.5, lam_fixed=LAMF)
print("VmData", vmdata())
