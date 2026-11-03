import torch, json, numpy as np, collections, time
t0 = time.time()
P = torch.load(WORK + "/L47pos.pt"); Nn = torch.load(WORK + "/new47pos.pt", weights_only=False)
AXp = torch.load(WORK + "/L47aux.pt"); gap_p = (AXp["maxaux"] - AXp["maxnon"]).numpy()
kind_p = np.load(WORK + "/det_kind.npy", allow_pickle=True); lab_p = np.load(WORK + "/det_lab.npy"); wt_p = np.load(WORK + "/det_wt.npy")
XH = torch.cat([P["XH"], Nn["XH"]]).float(); Y = torch.cat([P["Y"].half(), Nn["Y"].half()]).float(); R47 = torch.cat([P["R47"].half(), Nn["R47"].half()]).float()
gap = np.concatenate([gap_p, Nn["gap"]]); kind = np.concatenate([kind_p, Nn["kind"]]); lab = np.concatenate([lab_p, Nn["lab"]]); wt = np.concatenate([wt_p, Nn["wt"]])
grp = np.array([p[0] for p in P["POS"]] + [10000 + p[0] for p in Nn["POS"]])
src = np.array(["prac"] * len(P["POS"]) + ["new"] * len(Nn["POS"]))
N = len(lab); print("N", N, collections.Counter(kind))
mhat = torch.load(WORK + "/push47.pt")["mhat"]; p = torch.load(WORK + "/push47c.pt")["p"]
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight"); E = W("model.language_model.embed_tokens.weight")
SC = 0.0478
scale = float(Y.pow(2).mean(1).sqrt().median() * R47.pow(2).mean(1).sqrt().median() / SC)
X = XH.numpy(); b = (XH @ mhat).numpy()
uniq = np.unique(grp); rng = np.random.RandomState(7); rng.shuffle(uniq); folds = np.array_split(uniq, 6)
def cvscores(lam):
    s = np.zeros(N)
    for f in folds:
        te = np.isin(grp, f); tr = ~te
        sw = np.sqrt(wt[tr]); Xw = X[tr] * sw[:, None]; yw = lab[tr] * sw
        w = np.linalg.solve(Xw.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ yw)
        s[te] = X[te] @ w
    return s
out0 = rmsn(Y, wpf)
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
def simulate(s, theta, alpha, P0):
    a = gelu(alpha * (s - theta * b)) * b
    fire = np.where(a * P0 > 1e-3)[0]
    changed = np.zeros(N, dtype=bool)
    for st in range(0, len(fire), 512):
        idx = torch.tensor(fire[st:st+512])
        y2 = Y[idx] + torch.tensor(a[fire[st:st+512]], dtype=torch.float32)[:, None] * (P0 * scale) * p[None, :]
        r2 = R47[idx] + SC * (rmsn(y2, wpf) - out0[idx])
        t2 = (rmsn(r2, nw) @ E.T).argmax(1); t0_ = (rmsn(R47[idx], nw) @ E.T).argmax(1)
        changed[idx.numpy()] = (t2 != t0_).numpy()
    return changed
tgt = np.isin(kind, ["DIdec_err", "DIaux_err", "DIaux"]) & (gap > 0)
bad = np.isin(kind, ["DDdec", "DDaux", "crit_neg", "GENaux"]) & (gap > 0)
for lam in [30.0, 100.0, 300.0]:
    s = cvscores(lam)
    np.save(WORK + f"/comb_scv_{int(lam)}.npy", s)
    for theta in [0.3, 0.4, 0.5]:
        for P0 in [6.0]:
            ch = simulate(s, theta, 10.0, P0)
            fx_p = int((ch & tgt & (src == "prac")).sum()); fx_n = int((ch & tgt & (src == "new")).sum())
            br_p = int((ch & bad & (src == "prac")).sum()); br_n = int((ch & bad & (src == "new")).sum())
            oth = int((ch & ~tgt & ~bad).sum())
            print(f"lam {lam} theta {theta} P0 {P0}: fixed prac {fx_p}/{int((tgt&(src=='prac')).sum())} new {fx_n}/{int((tgt&(src=='new')).sum())} | broken prac {br_p} new {br_n} | other changes {oth}   [{time.time()-t0:.0f}s]")
