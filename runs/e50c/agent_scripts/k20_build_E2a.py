import numpy as np, torch, json
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
NSYS = 45
SL = {30: 0.7227, 31: 0.6993, 32: 0.5546, 33: 0.629, 34: 0.75}
isfirst = lambda i: dec2[i] == M2[i]["n_prompt_tokens"] - 1
for nm in list(edits()):
    drop_edit(nm)

def make_edit(LAYERS, fold_train, kappa, tag, lam_choice, ramp=(10, 50), sys_w=1.0):
    CFG = {}
    for L in LAYERS:
        P = f"model.language_model.layers.{L}"
        X, grp, item, row = build_X2(L)
        tr = fold_train(item)
        te = ~tr
        Xc = X[::3]
        G = Xc.T @ Xc / len(Xc); bv = Xc.mean(0)
        c = np.linalg.solve(G + np.trace(G) / G.shape[0] * 1e-3 * np.eye(G.shape[0], dtype=np.float32), bv).astype(np.float32)
        # dead neurons
        Wg = W(f"{P}.mlp.gate_proj.weight").float(); Wu = W(f"{P}.mlp.up_proj.weight").float(); Wd = W(f"{P}.mlp.down_proj.weight").float()
        Xs = torch.from_numpy(X[::4])
        gx = Xs @ Wg.T
        proxy = torch.relu(gx).mean(0) * (Xs @ Wu.T).abs().mean(0) * Wd.norm(dim=0)
        sel = [int(j) for j in torch.argsort(proxy)[:4]]
        del gx
        Wg2, Wu2, Wd2 = Wg.clone(), Wu.clone(), Wd.clone()
        # directions from training-fold items
        R = lambda i: A2[f"resid.{L}"][i].float()
        trn_items = set(np.unique(item[tr]).tolist())
        def dvec(first):
            Wl = [R(i)[dec2[i]] for i in range(55) if i in trn_items and isfirst(i) == first]
            Rl = [R(i)[dec2[i]] for i in range(55, 259) if i in dec2 and i in trn_items and isfirst(i) == first]
            return (torch.stack(Rl).mean(0) - torch.stack(Wl).mean(0)).numpy(), len(Wl), len(Rl)
        rms_dec = np.median([torch.sqrt((A2[f"mlp_out.{L}"][i][dec2[i]].float() ** 2).mean()).item() for i in range(259) if i in dec2])
        wpf = W(f"{P}.post_feedforward_layernorm.weight").float().numpy()
        for pos, other, jj in [("Pf", "Pm", sel[:2]), ("Pm", "Pf", sel[2:])]:
            W8 = {pos: 1.0, other: 0.3, "FIRSTNEG": 1.0, "DDdec": 1.0, "SYS": sys_w, "SRC": 0.5, "OUT": 1.0}
            Gm, bm = gram(X, grp, pos, W8, tr)
            lam = np.trace(Gm) / Gm.shape[0] * lam_choice[(L, pos)]
            w = np.linalg.solve(Gm + lam * np.eye(Gm.shape[0], dtype=np.float32), bm).astype(np.float32)
            s = X @ w
            sp_te = s[te & (grp == pos)]
            theta = float(np.percentile(sp_te, ramp[0])); top = float(np.percentile(sp_te, ramp[1]))
            Dl = max(top - theta, 0.1)
            sys_max = float(s[grp == "SYS"].max())
            fp = {g: float(np.mean(s[te & (grp == g)] > theta)) for g in ["FIRSTNEG", "DDdec", "OUT", "SRC"]}
            d, nw, nr = dvec(pos == "Pf")
            v = rms_dec * kappa * d / SL[L] / np.maximum(wpf, 1e-6); v[wpf < 0.1] = 0
            a = 8.0 / Dl; b = 1.0 / (a * Dl)
            Wg2[jj[0]] = torch.from_numpy(a * (w - theta * c)); Wg2[jj[1]] = torch.from_numpy(a * (w - (theta + Dl) * c))
            Wu2[jj[0]] = torch.from_numpy(b * c); Wu2[jj[1]] = torch.from_numpy(b * c)
            Wd2[:, jj[0]] = torch.from_numpy(v); Wd2[:, jj[1]] = torch.from_numpy(-v)
            CFG[f"{L}_{pos}"] = dict(theta=theta, Dl=Dl, neurons=jj, vnorm=float(np.linalg.norm(v)), sys_max=sys_max, fp_test=fp, nW=nw, nR=nr)
            print(f"L{L} {pos}: theta={theta:.3f} top={theta+Dl:.3f} SYSmax={sys_max:.3f} heldout FP@theta {fp} |d|={np.linalg.norm(d):.2f} (nW={nw}, nR={nr}) |v|={np.linalg.norm(v):.2f}")
            np.save(f"{WORK}/{tag}_w_{L}_{pos}.npy", w)
        np.save(f"{WORK}/{tag}_c_{L}.npy", c)
        save_edit(f"{P}.mlp.gate_proj.weight", Wg2); save_edit(f"{P}.mlp.up_proj.weight", Wu2); save_edit(f"{P}.mlp.down_proj.weight", Wd2)
        del X
    json.dump(CFG, open(f"{WORK}/cfg_{tag}.json", "w"))
    return CFG

LAMC = {(31, "Pf"): 0.01, (31, "Pm"): 0.1, (34, "Pf"): 0.01, (34, "Pm"): 1.0}
cfg = make_edit([31, 34], lambda item: item % 2 == 0, 3.0, "E2a", LAMC)
print(edits())
