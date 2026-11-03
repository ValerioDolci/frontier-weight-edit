import numpy as np, torch, json
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
NSYS = 45
for L in [30, 31]:
    w = np.load(f"{WORK}/det_w_L{L}.npy")
    # sample rows (all items, all non-system rows + system rows of item 0)
    Xs, isP, isD = [], [], []
    for i in range(len(I2)):
        xi = mlp_in(L, i).numpy()
        start = 0 if i == 0 else NSYS
        Xs.append(xi[start:])
    X = np.concatenate(Xs).astype(np.float32)
    # constant direction c: ridge regression of 1 on x
    G = X.T @ X / len(X); bvec = X.mean(0)
    lam = np.trace(G) / G.shape[0] * 1e-3
    c = np.linalg.solve(G + lam * np.eye(G.shape[0], dtype=np.float32), bvec)
    pc = X @ c
    print(f"L{L}: const dir c: mean={pc.mean():.4f} std={pc.std():.4f} min={pc.min():.3f} max={pc.max():.3f} |c|={np.linalg.norm(c):.2f}; |w|={np.linalg.norm(w):.2f} |x| mean={np.linalg.norm(X,axis=1).mean():.2f}")
    # system rows
    xs0 = mlp_in(L, 0).numpy()[:NSYS] @ c
    print("   c.x on system rows: min", xs0.min().round(3), "max", xs0.max().round(3))
    np.save(f"{WORK}/const_c_L{L}.npy", c)
    # mlp_out rms at decision rows vs elsewhere
    wpf = W(f"model.language_model.layers.{L}.post_feedforward_layernorm.weight").float()
    rm_dec, rm_all = [], []
    for i in range(0, 389):
        m = A2[f"mlp_out.{L}"][i].float()
        rms = torch.sqrt((m ** 2).mean(-1))
        if i in dec2: rm_dec.append(rms[dec2[i]].item())
        rm_all.append(rms[NSYS:].numpy())
    rm_all = np.concatenate(rm_all)
    print(f"   mlp_out rms: decision rows median={np.median(rm_dec):.3f} (p10 {np.percentile(rm_dec,10):.3f}, p90 {np.percentile(rm_dec,90):.3f}); all rows median={np.median(rm_all):.3f}")
    # post_ff output norm = |w_pf * m/rms(m)|
    # neuron weight norms
    Wd = W(f"model.language_model.layers.{L}.mlp.down_proj.weight").float()
    Wg = W(f"model.language_model.layers.{L}.mlp.gate_proj.weight").float()
    Wu = W(f"model.language_model.layers.{L}.mlp.up_proj.weight").float()
    dn = Wd.norm(dim=0); gn = Wg.norm(dim=1); un = Wu.norm(dim=1)
    score = dn * gn * un
    order = torch.argsort(score)
    print("   down col norm: median", dn.median().item(), "min", dn.min().item())
    print("   smallest dn*gn*un neurons:", [(int(j), round(dn[j].item(),4), round(gn[j].item(),3), round(un[j].item(),3)) for j in order[:8]])
    # gate pre-activation sign stats for those candidates on sample rows (projection only)
    cand = order[:8]
    gx = X[::7] @ Wg[cand].numpy().T
    print("   candidate gate.x: frac>0", np.round((gx > 0).mean(0), 4), " max", np.round(gx.max(0), 2))
