import torch, json, numpy as np, gc, time
t0 = time.time()
cfg = json.load(open(WORK + "/edit2_cfg.json")); parts = torch.load(WORK + "/edit2_parts.pt")
P = torch.load(WORK + "/L47pos.pt"); POS = [tuple(p) for p in P["POS"]]
X47o = P["XH"].float(); del P; gc.collect()
X29 = torch.load(WORK + "/L29pos.pt")["prac"]["XH"].float()
V2 = torch.load(WORK + "/ver2pos.pt"); XH1 = V2["XH1"].float(); del V2; gc.collect()
lp0 = json.load(open(WORK + "/L47a_logprobs.json")); lp1 = json.load(open(WORK + "/ver2_logprobs.json"))
toks = json.load(open(WORK + "/L47a_tokens.json")); items = json.load(open(WORK + "/L47a_items.json"))
wpre = W("model.language_model.layers.47.pre_feedforward_layernorm.weight")
g = parts["g"]; delta = parts["delta"]; w29 = parts["w29"]
dx_unit = cfg["u0"] * (delta * wpre)
dX = XH1 - X47o
a_meas = (dX @ g).numpy()
b29 = X29[:, 1750] / 0.091
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
a_pred = gelu(cfg["a29s"] * (X29 @ w29 - cfg["th29"] * b29).numpy())
big = np.where(np.abs(a_pred) > 0.3)[0]; small = np.where(np.abs(a_pred) < 0.01)[0]
print("n positions", len(a_meas), "flagged", len(big))
print("corr a_meas vs a_pred (flagged):", round(float(np.corrcoef(a_meas[big], a_pred[big])[0, 1]), 4), " median ratio", round(float(np.median(a_meas[big] / a_pred[big])), 3))
print("unflagged: a_meas std", round(float(a_meas[small].std()), 4), "max", round(float(np.abs(a_meas[small]).max()), 4))
resid = dX - torch.tensor(a_meas, dtype=torch.float32)[:, None] * dx_unit[None, :]
print("|dX| median flagged", round(float(dX[big].norm(dim=1).median()), 3), "|unexplained| median flagged", round(float(resid[big].norm(dim=1).median()), 3), "|dX| median unflagged", round(float(dX[small].norm(dim=1).median()), 3), "|x_hat| median", round(float(X47o.norm(dim=1).median()), 2))
d = np.array([lp1[k][t] - lp0[k][t] for (k, t) in POS])
print("positions |dlogprob|>0.5:", int((np.abs(d) > 0.5).sum()), " >2:", int((np.abs(d) > 2).sum()))
cat = {pp["id"]: pp["cat"] for pp in practice}
def lab(k): return items[k].get("id", ("gen", k))
order = np.argsort(d)
for j in order[:15]:
    k, t = POS[j]; print("  ", lab(k), cat.get(items[k].get("id"), "GEN"), repr(toks[k][t]), round(float(d[j]), 2), "a_pred", round(float(a_pred[j]), 2), "a_meas", round(float(a_meas[j]), 2))
unexp = np.where((np.abs(d) > 0.5) & (np.abs(a_pred) < 0.1))[0]
print("unexpected changes (|d|>0.5 & not flagged):", len(unexp))
for j in unexp[:20]:
    k, t = POS[j]; print("  ", lab(k), cat.get(items[k].get("id"), "GEN"), repr(toks[k][t]), round(float(d[j]), 2), "a_meas", round(float(a_meas[j]), 2), "ctx", toks[k][max(0, t-3):t])
np.save(WORK + "/ver2_ameas.npy", a_meas); np.save(WORK + "/ver2_d.npy", d); np.save(WORK + "/ver2_apred.npy", a_pred)
del X47o, XH1, X29, dX, resid; gc.collect()
print(time.time() - t0)
