import numpy as np, torch, json
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
NSYS = 45
SL = {30: 0.7227, 31: 0.6993, 32: 0.5546, 33: 0.629, 34: 0.75}
DET = np.load(f"{WORK}/det_E2.npy", allow_pickle=True).item()
isfirst = lambda i: dec2[i] == M2[i]["n_prompt_tokens"] - 1
for nm in list(edits()):
    drop_edit(nm)

def dead_neurons(L, k):
    P = f"model.language_model.layers.{L}"
    Wg = W(f"{P}.mlp.gate_proj.weight").float(); Wu = W(f"{P}.mlp.up_proj.weight").float(); Wd = W(f"{P}.mlp.down_proj.weight").float()
    Xs = torch.cat([mlp_in(L, i)[NSYS:] for i in range(0, len(I2), 3)])
    Xs = torch.cat([Xs, mlp_in(L, 0)[:NSYS]])
    gx = Xs @ Wg.T
    mx = gx.max(0).values
    ux = (Xs @ Wu.T).abs().mean(0)
    proxy = (torch.relu(gx).mean(0)) * ux * Wd.norm(dim=0)
    order = torch.argsort(proxy)
    sel = [int(j) for j in order[:k]]
    print(f"  L{L} neurons {sel} proxy {[round(proxy[j].item(),6) for j in sel]} max gate.x {[round(mx[j].item(),3) for j in sel]}")
    return sel, Wg, Wu, Wd

CFG = {}
for L in [31, 34]:
    P = f"model.language_model.layers.{L}"
    c = None
    # constant direction at this layer
    Xc = torch.cat([mlp_in(L, i)[NSYS:] for i in range(0, len(I2), 2)] + [mlp_in(L, 0)[:NSYS]]).numpy()
    G = Xc.T @ Xc / len(Xc); bv = Xc.mean(0)
    c = np.linalg.solve(G + np.trace(G) / G.shape[0] * 1e-3 * np.eye(G.shape[0], dtype=np.float32), bv).astype(np.float32)
    pc = Xc @ c
    print(f"L{L}: c.x mean {pc.mean():.4f} std {pc.std():.4f} min {pc.min():.3f}")
    # directions at resid_L
    R = lambda i: A2[f"resid.{L}"][i].float()
    W_f = [R(i)[dec2[i]] for i in range(55) if isfirst(i)]; W_m = [R(i)[dec2[i]] for i in range(55) if not isfirst(i)]
    R_f = [R(i)[dec2[i]] for i in range(55, 259) if i in dec2 and isfirst(i)]; R_m = [R(i)[dec2[i]] for i in range(55, 259) if i in dec2 and not isfirst(i)]
    d_f = (torch.stack(R_f).mean(0) - torch.stack(W_f).mean(0)).numpy()
    d_m = (torch.stack(R_m).mean(0) - torch.stack(W_m).mean(0)).numpy()
    # mlp_out rms at decision rows
    rms_dec = np.median([torch.sqrt((A2[f"mlp_out.{L}"][i][dec2[i]].float() ** 2).mean()).item() for i in range(259) if i in dec2])
    wpf = W(f"{P}.post_feedforward_layernorm.weight").float().numpy()
    sel, Wg, Wu, Wd = dead_neurons(L, 4)
    Wg2, Wu2, Wd2 = Wg.clone(), Wu.clone(), Wd.clone()
    kappa = 3.0
    for (pos, d, jj) in [("Pf", d_f, sel[:2]), ("Pm", d_m, sel[2:])]:
        w = DET[f"{L}_{pos}"].astype(np.float32)
        # score distributions on all rows to pick threshold
        sP, sN = [], []
        for i in range(len(I2)):
            xi = mlp_in(L, i).numpy()
            s = xi @ w
            start = 0 if i == 0 else NSYS
            for r in range(start, len(s)):
                tgt = (i in dec2 and r == dec2[i] and i < 259 and (isfirst(i) == (pos == "Pf")))
                (sP if tgt else sN).append(s[r])
        sP, sN = np.array(sP), np.array(sN)
        theta = float(np.percentile(sP, 5)); Dl = float(max(np.percentile(sP, 50) - theta, 0.1))
        # make sure theta is above most negatives
        print(f"  {pos}: P p5={np.percentile(sP,5):.2f} p50={np.percentile(sP,50):.2f}; neg p99.9={np.percentile(sN,99.9):.2f} max={sN.max():.2f}; theta={theta:.2f} Dl={Dl:.2f}; neg frac>theta={np.mean(sN>theta):.5f}")
        v = rms_dec * kappa * d / SL[L] / np.maximum(wpf, 1e-6)
        v[wpf < 0.1] = 0
        a = 8.0 / Dl
        b = 1.0 / (a * Dl)
        Wg2[jj[0]] = torch.from_numpy(a * (w - theta * c)); Wg2[jj[1]] = torch.from_numpy(a * (w - (theta + Dl) * c))
        Wu2[jj[0]] = torch.from_numpy(b * c); Wu2[jj[1]] = torch.from_numpy(b * c)
        Wd2[:, jj[0]] = torch.from_numpy(v); Wd2[:, jj[1]] = torch.from_numpy(-v)
        CFG[f"{L}_{pos}"] = dict(theta=theta, Dl=Dl, a=a, b=b, kappa=kappa, neurons=jj, vnorm=float(np.linalg.norm(v)), dnorm=float(np.linalg.norm(d)))
        print(f"     |d|={np.linalg.norm(d):.2f} |v|={np.linalg.norm(v):.3f} rms_dec={rms_dec:.3f}")
    save_edit(f"{P}.mlp.gate_proj.weight", Wg2); save_edit(f"{P}.mlp.up_proj.weight", Wu2); save_edit(f"{P}.mlp.down_proj.weight", Wd2)
json.dump(CFG, open(f"{WORK}/cfg_E2.json", "w"))
print(edits())
