import numpy as np, torch, json, gc
def gram_w(X, grp, pos, other, mask, sys_w=1.0):
    W8 = {pos: 1.0, other: 0.3, "FIRSTNEG": 1.5 if pos == "Pf" else 1.0, "DDdec": 1.0, "SYS": sys_w, "SRC": 0.5, "OUT": 1.0}
    y = np.where(grp == pos, 1.0, -1.0).astype(np.float32)
    wts = np.zeros(len(X), np.float32)
    for g, tot in W8.items():
        m = (grp == g) & mask
        if m.sum(): wts[m] = tot / m.sum()
    sel = wts > 0
    Xt = X[sel]; wt = wts[sel]
    return (Xt * wt[:, None]).T @ Xt, (Xt * wt[:, None]).T @ y[sel]
def solve(G, b, lamf):
    lam = np.trace(G) / G.shape[0] * lamf
    return np.linalg.solve(G + lam * np.eye(G.shape[0], dtype=np.float32), b).astype(np.float32)
def build_E3(kappa_f, kappa_m, ramp_f=(15, 50), ramp_m=(10, 50), lams=(0.01, 0.1, 1.0), tag="E3", apply=True):
    meta = dict(np.load(f"{WORK}/Xmeta.npz")); grp = meta["grp"]; fold = meta["fold"]
    D = dict(np.load(f"{WORK}/dirdata.npz"))
    cfg_old = json.load(open(f"{WORK}/cfg_E2a.json"))
    CFG = {}
    for L in [31, 34]:
        X = np.load(f"{WORK}/X{L}_f16.npy").astype(np.float32)
        Xc = X[::3]
        Gc = Xc.T @ Xc / len(Xc); bc = Xc.mean(0)
        c = np.linalg.solve(Gc + np.trace(Gc) / Gc.shape[0] * 1e-3 * np.eye(3840, dtype=np.float32), bc).astype(np.float32)
        del Xc, Gc
        pc = X @ c
        print(f"L{L}: c.x mean {pc.mean():.4f} sd {pc.std():.4f} min {pc.min():.3f} max {pc.max():.3f}")
        P = f"model.language_model.layers.{L}"
        Wg = W(f"{P}.mlp.gate_proj.weight"); Wu = W(f"{P}.mlp.up_proj.weight"); Wd = W(f"{P}.mlp.down_proj.weight")
        wpf = LN[L]["pf"].numpy()
        rms_dec = {31: 0.151, 34: 0.389}[L]
        for pos, other, kap, ramp in [("Pf", "Pm", kappa_f, ramp_f), ("Pm", "Pf", kappa_m, ramp_m)]:
            folds = [fold == 0, fold == 1]
            Gs = [gram_w(X, grp, pos, other, f) for f in folds]
            best = None
            for lamf in lams:
                s = np.zeros(len(X), np.float32)
                for k in range(2):
                    w_k = solve(*Gs[k], lamf)
                    s[folds[1 - k]] = X[folds[1 - k]] @ w_k
                sp = s[grp == pos]
                key = "FIRSTNEG" if pos == "Pf" else "DDdec"
                a1 = auc(sp, s[grp == key]); a2 = auc(sp, s[grp == "OUT"])
                th = np.percentile(sp, ramp[0])
                fp = {g: round(float(np.mean(s[grp == g] > th)), 4) for g in ["FIRSTNEG", "DDdec", "OUT", "SRC"]}
                print(f"  L{L} {pos} lam={lamf}: CV AUC {key}={a1:.3f} OUT={a2:.4f}; P p{ramp[0]}={th:.3f} p50={np.median(sp):.3f}; FP@theta {fp}")
                score = a1 + a2 - 2 * fp["FIRSTNEG" if pos == "Pf" else "DDdec"]
                if best is None or score > best[0]:
                    best = (score, lamf, s.copy())
            _, lamf, s_cv = best
            sp = s_cv[grp == pos]
            theta = float(np.percentile(sp, ramp[0])); top = float(np.percentile(sp, ramp[1])); Dl = max(top - theta, 0.1)
            Gall = gram_w(X, grp, pos, other, np.ones(len(X), bool))
            w = solve(*Gall, lamf)
            s_all = X @ w
            sys_max = float(s_all[grp == "SYS"].max())
            # direction
            sel_w = (D["lab"] == "W") & (D["first"] == (pos == "Pf")); sel_r = (D["lab"] == "R") & (D["first"] == (pos == "Pf"))
            d = D[f"L{L}"][sel_r].mean(0) - D[f"L{L}"][sel_w].mean(0)
            # split-half reliability
            rng = np.random.RandomState(1); iw = rng.permutation(np.where(sel_w)[0]); ir = rng.permutation(np.where(sel_r)[0])
            h1 = D[f"L{L}"][ir[: len(ir)//2]].mean(0) - D[f"L{L}"][iw[: len(iw)//2]].mean(0)
            h2 = D[f"L{L}"][ir[len(ir)//2:]].mean(0) - D[f"L{L}"][iw[len(iw)//2:]].mean(0)
            rel = float(h1 @ h2 / np.linalg.norm(h1) / np.linalg.norm(h2))
            v = rms_dec * kap * d / SL[L] / np.maximum(wpf, 1e-6); v[wpf < 0.1] = 0
            a = 8.0 / Dl; b = 1.0 / (a * Dl)
            jj = cfg_old[f"{L}_{pos}"]["neurons"]
            Wg[jj[0]] = torch.from_numpy(a * (w - theta * c)); Wg[jj[1]] = torch.from_numpy(a * (w - (theta + Dl) * c))
            Wu[jj[0]] = torch.from_numpy(b * c); Wu[jj[1]] = torch.from_numpy(b * c)
            Wd[:, jj[0]] = torch.from_numpy(v.astype(np.float32)); Wd[:, jj[1]] = torch.from_numpy(-v.astype(np.float32))
            CFG[f"{L}_{pos}"] = dict(theta=theta, Dl=Dl, lam=lamf, neurons=jj, kappa=kap, sys_max=sys_max, rel=rel,
                                     nW=int(sel_w.sum()), nR=int(sel_r.sum()), dnorm=float(np.linalg.norm(d)), vnorm=float(np.linalg.norm(v)))
            np.save(f"{WORK}/{tag}_w_{L}_{pos}.npy", w)
            print(f"  -> L{L} {pos}: lam {lamf} theta {theta:.3f} top {theta+Dl:.3f} SYSmax {sys_max:.3f} | dir nW={int(sel_w.sum())} nR={int(sel_r.sum())} |d|={np.linalg.norm(d):.2f} split-half cos {rel:.2f} |v|={np.linalg.norm(v):.2f}")
            del Gs, Gall, s_cv, s_all
        np.save(f"{WORK}/{tag}_c_{L}.npy", c)
        if apply:
            save_edit(f"{P}.mlp.gate_proj.weight", Wg); save_edit(f"{P}.mlp.up_proj.weight", Wu); save_edit(f"{P}.mlp.down_proj.weight", Wd)
        del X, Wg, Wu, Wd; gc.collect()
    json.dump(CFG, open(f"{WORK}/cfg_{tag}.json", "w"))
    return CFG
CFG_E3 = build_E3(kappa_f=3.0, kappa_m=3.5)
print(edits()); print("VmData", vmdata())
