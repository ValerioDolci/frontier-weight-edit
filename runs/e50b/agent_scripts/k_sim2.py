import torch, json, numpy as np, time
t0 = time.time()
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight")
E = W("model.language_model.embed_tokens.weight")
AUX = [182706, 51542, 139292, 205867, 23120, 23456, 23306, 40877, 51638, 1883]
v = nw * wpf
G = torch.zeros(3840, 3840, dtype=torch.float64)
for s in range(0, E.shape[0], 16384):
    e = E[s:s+16384].double()
    G += e.T @ e
Ea = E[AUX].double()
G_non = G - Ea.T @ Ea
torch.save(G, WORK + "/EtE.pt")
print("EtE", time.time() - t0)
vd = v.double()
C = vd[:, None] * G_non * vd[None, :]
lam = 1e-3 * float(torch.trace(C)) / 3840
Cl = C + lam * torch.eye(3840, dtype=torch.float64)
Ma = Ea * vd[None, :]                        # aux rows of M
Ci_MaT = torch.linalg.solve(Cl, Ma.T)        # 3840 x 10
K = Ma @ Ci_MaT                               # 10 x 10
p = -(Ci_MaT @ torch.linalg.solve(K, torch.ones(len(AUX), dtype=torch.float64)))
p = p.float()
ind = (E * v) @ p                             # induced raw-logit change per unit p
print("aux induced:", [round(float(x), 3) for x in ind[AUX]])
nonaux = torch.ones(E.shape[0], dtype=torch.bool); nonaux[AUX] = False
iv = ind[nonaux]
print("non-aux induced: max", float(iv.max()), "min", float(iv.min()), "rms", float(iv.pow(2).mean().sqrt()))
inv = {vv: k for k, vv in json.load(open(WORK + "/tokid.json")).items()}
order = torch.argsort(ind)
print("most suppressed:", [(inv.get(int(i), int(i)), round(float(ind[i]), 3)) for i in order[:20]])
print("most boosted:", [(inv.get(int(i), int(i)), round(float(ind[i]), 3)) for i in order[-20:]])
for tk in ["▁era", "E", "Er", "Era", "▁erano", "▁in", "▁stava", "▁stav", "▁da", "▁per", "F", "Vive", "▁senza", "▁ha", "▁è", "Ha", "▁sono", "▁stato", "▁aveva"]:
    if tk in json.load(open(WORK + "/tokid.json")):
        i = json.load(open(WORK + "/tokid.json"))[tk]; print("   ", tk, round(float(ind[i]), 3))
print("|p|", float(p.norm()))
torch.save({"p": p, "ind": ind}, WORK + "/push47b.pt")
