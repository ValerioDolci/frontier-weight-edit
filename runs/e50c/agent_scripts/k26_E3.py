import numpy as np, torch, json, gc
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
d3 = json.load(open(f"{WORK}/dec_r3.json")); dec3 = {int(k): v for k, v in d3["dec3"].items()}; marg3 = {int(k): v for k, v in d3["marg3"].items()}
NSYS = 45
SL = {30: 0.7227, 31: 0.6993, 32: 0.5546, 33: 0.629, 34: 0.75}
WPA = {L: W(f"model.language_model.layers.{L}.post_attention_layernorm.weight").float() for L in [31, 34]}
WPRE = {L: W(f"model.language_model.layers.{L}.pre_feedforward_layernorm.weight").float() for L in [31, 34]}
def mlp_in_src(src, L, i):
    A = A2 if src == 2 else A3
    h = A[f"resid.{L-1}"][i].float() + rmsn(A[f"attn_out.{L}"][i].float(), WPA[L])
    return rmsn(h, WPRE[L])
def items_spec():
    # (src, idx, kind, dec_row, fold, label_for_dir)
    out = []
    for i in range(len(I2)):
        kind = "DI" if i < 259 else ("DD" if i < 389 else "NEG")
        lab = "W" if i < 55 else ("R" if i < 259 else None)
        out.append((2, i, kind, dec2.get(i), i % 2, lab))
    for k in range(120):
        m = marg3[k]
        lab = "R" if m > 1.0 else ("W" if m < -1.0 else None)
        out.append((3, k, "DI", dec3[k], k % 2, lab))
    for k in range(240, 330):
        out.append((3, k, "DD", dec3.get(k), k % 2, None))
    for k in range(330, 390):
        out.append((3, k, "NEG", None, k % 2, None))
    return out
SPEC = items_spec()
def build(L):
    X, grp, fold, srcs = [], [], [], []
    for (src, i, kind, dr, f, lab) in SPEC:
        M = M2 if src == 2 else M3
        xi = mlp_in_src(src, L, i).numpy()
        n0 = M[i]["n_prompt_tokens"]
        start = 0 if (src == 2 and i == 0) else NSYS
        g = []
        for r in range(start, xi.shape[0]):
            if kind == "DI" and dr is not None and r == dr:
                g.append("Pf" if dr == n0 - 1 else "Pm")
            elif kind == "DI" and dr is not None and dr - 1 <= r <= dr + 3:
                g.append("IGN")
            elif r == n0 - 1:
                g.append("FIRSTNEG")
            elif kind == "DD" and dr is not None and r == dr:
                g.append("DDdec")
            else:
                g.append("SYS" if r < NSYS else ("SRC" if r < n0 - 1 else "OUT"))
        X.append(xi[start:]); grp += g; fold += [f] * len(g); srcs += [src] * len(g)
    return np.concatenate(X).astype(np.float32), np.array(grp), np.array(fold), np.array(srcs)
def gram_w(X, grp, pos, other, mask, sys_w=1.0):
    W8 = {pos: 1.0, other: 0.3, "FIRSTNEG": 1.0, "DDdec": 1.0, "SYS": sys_w, "SRC": 0.5, "OUT": 1.0}
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
def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(neg) > 20000: neg = np.random.RandomState(0).choice(neg, 20000, replace=False)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return float((pos[:, None] > neg[None, :]).mean())
