import torch, json, numpy as np, collections, time
t0 = time.time()
D0 = torch.load(WORK + "/L47pos.pt"); XH = D0["XH"].float(); POS = D0["POS"]; Y = D0["Y"].float(); R47 = D0["R47"].float()
AX = torch.load(WORK + "/L47aux.pt"); gap = (AX["maxaux"] - AX["maxnon"]).numpy(); top1_aux = gap > 0
kind = np.load(WORK + "/det_kind.npy", allow_pickle=True); lab = np.load(WORK + "/det_lab.npy"); wt = np.load(WORK + "/det_wt.npy")
A = load_acts("L47a"); meta, items = A["meta"], A["items"]
cat = {p["id"]: p["cat"] for p in practice}
N = len(POS); g = np.array([p[0] for p in POS])
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight")
E = W("model.language_model.embed_tokens.weight")
AUX = [182706, 51542, 139292, 205867, 23120, 23456, 23306, 40877, 51638, 1883]
SC = 0.0478
# mean direction for bias
m = XH.mean(0); mhat = m / (m @ m) * 1.0   # mhat . x ~ 1 on average
print("mhat.x stats", float((XH @ mhat).mean()), float((XH @ mhat).std()))
X = XH.numpy()
# uncentered weighted ridge, CV
uniq = np.unique(g); rng = np.random.RandomState(1); rng.shuffle(uniq); folds = np.array_split(uniq, 5)
lam = 100.0
scv = np.zeros(N)
for f in folds:
    te = np.isin(g, f); tr = ~te
    sw = np.sqrt(wt[tr]); Xw = X[tr] * sw[:, None]; yw = lab[tr] * sw
    w = np.linalg.solve(Xw.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ yw)
    scv[te] = X[te] @ w
sw = np.sqrt(wt); Xw = X * sw[:, None]; yw = lab * sw
wfull = np.linalg.solve(Xw.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ yw)
np.save(WORK + "/probe47_w.npy", wfull); np.save(WORK + "/probe47_scv.npy", scv)
print("probe fit", time.time() - t0)
# push direction p in y-space: minus sum of M_aux, M_t = E[t]*nw*wpf
Maux = (E[AUX] * nw * wpf)
p = -Maux.sum(0); p = p / p.norm()
# collateral: induced raw-logit direction on vocab (relative)
ind = (E * (nw * wpf)) @ p   # proportional to delta raw logit per unit
ia = ind[AUX]
print("induced on aux:", [round(float(v), 3) for v in ia])
order = torch.argsort(ind)
inv = {v: k for k, v in json.load(open(WORK + "/tokid.json")).items()}
print("most suppressed tokens:", [(inv.get(int(i), int(i)), round(float(ind[i]), 3)) for i in order[:25]])
print("most boosted tokens:", [(inv.get(int(i), int(i)), round(float(ind[i]), 3)) for i in order[-15:]])
torch.save({"p": p, "mhat": mhat}, WORK + "/push47.pt")
