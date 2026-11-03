import torch, json, numpy as np, collections, time, gc
t0 = time.time()
L29 = torch.load(WORK + "/L29pos.pt")
P = torch.load(WORK + "/L47pos.pt"); Nn = torch.load(WORK + "/new47pos.pt", weights_only=False)
AXp = torch.load(WORK + "/L47aux.pt"); gap_p = (AXp["maxaux"] - AXp["maxnon"]).numpy()
kind_p = np.load(WORK + "/det_kind.npy", allow_pickle=True); lab_p = np.load(WORK + "/det_lab.npy"); wt_p = np.load(WORK + "/det_wt.npy")
# fix label noise: drop DIaux_err where the item already has a DIdec_err/ok position; drop item 212
ids_p = np.array([P_ for P_ in [None]])
X29 = torch.cat([L29["prac"]["XH"], L29["new"]["XH"]]).float()
Y = torch.cat([P["Y"].half(), Nn["Y"].half()]).float(); R47 = torch.cat([P["R47"].half(), Nn["R47"].half()]).float()
gap = np.concatenate([gap_p, Nn["gap"]]); kind = np.concatenate([kind_p, Nn["kind"]]).astype(object); lab = np.concatenate([lab_p, Nn["lab"]]); wt = np.concatenate([wt_p, Nn["wt"]])
POS = [("p",) + tuple(x) for x in P["POS"]] + [("n",) + tuple(x) for x in Nn["POS"]]
grp = np.array([p[1] if p[0] == "p" else 10000 + p[1] for p in POS])
src = np.array([p[0] for p in POS])
A = load_acts("L47a"); items_p = A["items"]; del A; gc.collect()
pid = np.array([items_p[p[1]].get("id", -1) if p[0] == "p" else -1 for p in POS])
hasdec = set(pid[np.isin(kind, ["DIdec_err", "DIdec_ok"])])
for j in np.where(kind == "DIaux_err")[0]:
    if pid[j] in hasdec or pid[j] == 212:
        kind[j] = "other"; lab[j] = 0.0; wt[j] = 1.0
N = len(lab)
b29 = (X29[:, 1750] / 0.091).numpy()
X = X29.numpy()
uniq = np.unique(grp); rng = np.random.RandomState(7); rng.shuffle(uniq); folds = np.array_split(uniq, 6)
def cvscores(Xf, lam):
    s = np.zeros(N)
    for f in folds:
        te = np.isin(grp, f); tr = ~te
        sw = np.sqrt(wt[tr]); Xw = Xf[tr] * sw[:, None]; yw = lab[tr] * sw
        w = np.linalg.solve(Xw.T @ Xw + lam * np.eye(Xf.shape[1]), Xw.T @ yw)
        s[te] = Xf[te] @ w
    return s
tgt = np.isin(kind, ["DIdec_err", "DIaux_err", "DIaux"]) & (gap > 0)
bad = np.isin(kind, ["DDdec", "DDaux", "crit_neg", "GENaux"]) & (gap > 0)
print("targets", int(tgt.sum()), "bad", int(bad.sum()))
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight"); E = W("model.language_model.embed_tokens.weight")
p = torch.load(WORK + "/push47c.pt")["p"]
SC = 0.0478
scale = float(Y.pow(2).mean(1).sqrt().median() * R47.pow(2).mean(1).sqrt().median() / SC)
out0 = rmsn(Y, wpf)
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
def simulate(a, P0):
    fire = np.where(a * P0 > 1e-3)[0]
    changed = np.zeros(N, dtype=bool)
    for st in range(0, len(fire), 512):
        idx = torch.tensor(fire[st:st+512])
        y2 = Y[idx] + torch.tensor(a[fire[st:st+512]], dtype=torch.float32)[:, None] * (P0 * scale) * p[None, :]
        r2 = R47[idx] + SC * (rmsn(y2, wpf) - out0[idx])
        t2 = (rmsn(r2, nw) @ E.T).argmax(1); t0_ = (rmsn(R47[idx], nw) @ E.T).argmax(1)
        changed[idx.numpy()] = (t2 != t0_).numpy()
    return changed, len(fire)
for lam in [30.0, 100.0, 300.0]:
    s = cvscores(X, lam)
    np.save(WORK + f"/l29_scv_{int(lam)}.npy", s)
    for theta in [0.3, 0.4, 0.5]:
        a = gelu(10.0 * (s - theta * b29))
        ch, nf = simulate(a, 6.0)
        fx_p = int((ch & tgt & (src == "p")).sum()); fx_n = int((ch & tgt & (src == "n")).sum())
        br_p = int((ch & bad & (src == "p")).sum()); br_n = int((ch & bad & (src == "n")).sum())
        oth = int((ch & ~tgt & ~bad).sum())
        print(f"L29 lam {lam} theta {theta}: firing {nf} fixed prac {fx_p}/{int((tgt&(src=='p')).sum())} new {fx_n}/{int((tgt&(src=='n')).sum())} | broken prac {br_p} new {br_n} | other {oth}  [{time.time()-t0:.0f}s]")
np.save(WORK + "/comb_kind_fixed.npy", kind); np.save(WORK + "/comb_lab_fixed.npy", lab); np.save(WORK + "/comb_wt_fixed.npy", wt)
