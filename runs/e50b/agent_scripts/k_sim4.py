import torch, json, numpy as np, collections, time
t0 = time.time()
D0 = torch.load(WORK + "/L47pos.pt"); XH = D0["XH"].float(); POS = D0["POS"]; Y = D0["Y"].float(); R47 = D0["R47"].float()
AX = torch.load(WORK + "/L47aux.pt"); gap = (AX["maxaux"] - AX["maxnon"]).numpy()
kind = np.load(WORK + "/det_kind.npy", allow_pickle=True)
scv = np.load(WORK + "/probe47_scv.npy")
mhat = torch.load(WORK + "/push47.pt")["mhat"]
A = load_acts("L47a"); meta, items = A["meta"], A["items"]
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight")
E = W("model.language_model.embed_tokens.weight")
AUX = [182706, 51542, 139292, 205867, 23120, 23456, 23306, 40877, 51638, 1883]
SC = 0.0478
b = (XH @ mhat).numpy()
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
a = gelu(10.0 * (scv - 0.35 * b)) * b
fire = np.where(a > 1e-3)[0]
# protected set: top-8 tokens at firing positions (excluding aux)
prot = collections.Counter()
for s in range(0, len(fire), 512):
    idx = torch.tensor(fire[s:s+512])
    lg = (rmsn(R47[idx], nw) @ E.T)
    tk = lg.topk(8, dim=1).indices
    for row in tk:
        for t in row.tolist():
            if t not in AUX: prot[t] += 1
protected = [t for t, c in prot.items()]
print("protected tokens", len(protected), time.time() - t0)
G = torch.load(WORK + "/EtE.pt")
v = (nw * wpf).double()
Ea = E[AUX].double(); Ep = E[protected].double()
wprot = 200.0
Gw = G - Ea.T @ Ea + wprot * (Ep.T @ Ep)
C = v[:, None] * Gw * v[None, :]
lam = 1e-3 * float(torch.trace(C)) / 3840
Cl = C + lam * torch.eye(3840, dtype=torch.float64)
Ma = Ea * v[None, :]
Ci_MaT = torch.linalg.solve(Cl, Ma.T)
K = Ma @ Ci_MaT
p = -(Ci_MaT @ torch.linalg.solve(K, torch.ones(len(AUX), dtype=torch.float64))).float()
ind = (E * (nw * wpf)) @ p
nonaux = torch.ones(E.shape[0], dtype=torch.bool); nonaux[AUX] = False
print("aux induced", [round(float(x), 2) for x in ind[AUX]])
print("protected induced: max", float(ind[protected].max()), "min", float(ind[protected].min()), "rms", float(ind[protected].pow(2).mean().sqrt()))
print("all non-aux: max", float(ind[nonaux].max()), "min", float(ind[nonaux].min()), "rms", float(ind[nonaux].pow(2).mean().sqrt()))
tokid = json.load(open(WORK + "/tokid.json"))
for tk in ["▁era", "E", "Er", "Era", "▁erano", "▁in", "▁stava", "▁stav", "▁st", "St", "▁da", "▁per", "F", "Vive", "▁senza", "▁ha", "▁è", "Ha"]:
    if tk in tokid: print("   ", tk, round(float(ind[tokid[tk]]), 3))
print("|p|", float(p.norm()))
torch.save({"p": p, "ind": ind, "protected": protected}, WORK + "/push47c.pt")
