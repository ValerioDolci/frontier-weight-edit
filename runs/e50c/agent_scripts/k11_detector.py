import numpy as np, torch, json
r2 = load_acts("r2")
M2, I2, A2 = r2["meta"], r2["items"], r2["acts"]
r1 = load_acts("r1"); M1 = r1["meta"]
pairs = json.load(open(f"{WORK}/pairs_r1.json"))
dec1 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r1.json")).items()}
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps) * w
def Wt(name): return W(name).float()

# ---- decision rows in r2 ----
dec2 = {}
for k in range(55):
    t = dec1[k][0]; dec2[k] = M2[k]["n_prompt_tokens"] + t - 1
for k in range(55, 259):
    j = 110 + (k - 55)
    if j in dec1:
        assert M1[j]["tokens"] == M2[k]["tokens"]
        dec2[k] = M2[k]["n_prompt_tokens"] + dec1[j][0] - 1
AUX_AV = {"Ave", "▁aveva", "▁avevano", "▁avevamo", "▁avevo", "▁avevi", "▁ave", "▁avev"}
AUX_ES = {"E", "Era", "Er", "▁era", "▁erano", "▁eravamo", "▁ero", "▁eri", "▁er", "Eran"}
def dd_dec(i):
    toks = M2[i]["tokens"]
    for t, tk in enumerate(toks):
        if tk in AUX_AV or tk in AUX_ES:
            if tk in AUX_ES and t + 1 < len(toks) and (toks[t+1].startswith("▁stat") or toks[t+1].startswith("▁rim")):
                return t + 1
            return t
    return None
for k in range(259, 389):
    t = dd_dec(k)
    if t is not None: dec2[k] = M2[k]["n_prompt_tokens"] + t - 1
json.dump({str(k): v for k, v in dec2.items()}, open(f"{WORK}/dec_r2.json", "w"))
print("decisions:", sum(1 for k in dec2 if k < 55), sum(1 for k in dec2 if 55 <= k < 259), sum(1 for k in dec2 if 259 <= k < 389))

# system prompt identity check
a0 = A2["resid.30"][0].float(); a1 = A2["resid.30"][300].float()
same = [(a0[r] - a1[r]).abs().max().item() < 1e-3 for r in range(60)]
NSYS = same.index(False) if False in same else 60
print("identical leading rows:", NSYS)

def mlp_in(L, i):
    wpa = WPA[L]; wpre = WPRE[L]
    h = A2[f"resid.{L-1}"][i].float() + rmsn(A2[f"attn_out.{L}"][i].float(), wpa)
    return rmsn(h, wpre)
WPA = {L: Wt(f"model.language_model.layers.{L}.post_attention_layernorm.weight") for L in range(30, 35)}
WPRE = {L: Wt(f"model.language_model.layers.{L}.pre_feedforward_layernorm.weight") for L in range(30, 35)}

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(neg) > 20000: neg = np.random.RandomState(0).choice(neg, 20000, replace=False)
    return float((pos[:, None] > neg[None, :]).mean())

results = {}
for L in [30, 31, 32, 33, 34]:
    X, y, grp, item, row = [], [], [], [], []
    for i in range(len(I2)):
        xi = mlp_in(L, i)
        start = 0 if i == 0 else NSYS
        for r in range(start, xi.shape[0]):
            if i in dec2 and r == dec2[i]:
                g = "P" if i < 259 else "DDdec"
            else:
                g = "SYS" if r < NSYS else ("SRC" if r < M2[i]["n_prompt_tokens"] - 1 else "OUT")
            grp.append(g); item.append(i); row.append(r)
        X.append(xi[start:])
    X = torch.cat(X).numpy().astype(np.float32); grp = np.array(grp); item = np.array(item)
    train = (item % 2 == 0); test = ~train
    yv = np.where(grp == "P", 1.0, -1.0).astype(np.float32)
    # class weights
    wts = np.zeros(len(X), np.float32)
    for g, tot in [("P", 1.0), ("DDdec", 1.0), ("SYS", 0.2), ("SRC", 0.5), ("OUT", 1.0)]:
        m = (grp == g) & train
        if m.sum(): wts[m] = tot / m.sum()
    Xt = X[train]; wt = wts[train]; yt = yv[train]
    G = (Xt * wt[:, None]).T @ Xt
    bvec = (Xt * wt[:, None]).T @ yt
    lam = np.trace(G) / G.shape[0] * 1e-2
    w = np.linalg.solve(G + lam * np.eye(G.shape[0], dtype=np.float32), bvec)
    s = X @ w
    sp = s[test & (grp == "P")]; sd = s[test & (grp == "DDdec")]
    so = s[test & (grp == "OUT")]; ss = s[test & (grp == "SRC")]; sy = s[grp == "SYS"]
    thr = np.percentile(sp, 10)
    print(f"L{L}: test AUC P-vs-DDdec={auc(sp, sd):.3f} P-vs-OUT={auc(sp, so):.4f} P-vs-SRC={auc(sp, ss):.4f} | "
          f"P p10={thr:.2f} med={np.median(sp):.2f}; frac>thr: DDdec={np.mean(sd>thr):.3f} OUT={np.mean(so>thr):.4f} SRC={np.mean(ss>thr):.4f} SYSmax={sy.max():.2f}; "
          f"OUT p99.9={np.percentile(so,99.9):.2f} max={so.max():.2f}")
    results[L] = w
    np.save(f"{WORK}/det_w_L{L}.npy", w)
