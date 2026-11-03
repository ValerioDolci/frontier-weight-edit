import torch, json, numpy as np, time, gc
t0 = time.time()
cfg = json.load(open(WORK + "/edit2_cfg.json")); parts = torch.load(WORK + "/edit2_parts.pt")
P = torch.load(WORK + "/L47pos.pt"); L29 = torch.load(WORK + "/L29pos.pt")
X47o = P["XH"].float(); X29 = L29["prac"]["XH"].float()
A1 = load_acts("ver2"); acts1, meta1, items = A1["acts"], A1["meta"], A1["items"]
A0 = load_acts("L47a"); meta0 = A0["meta"]
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpa = W("model.language_model.layers.47.post_attention_layernorm.weight")
wpre = W("model.language_model.layers.47.pre_feedforward_layernorm.weight")
XH1 = []
for k in range(len(items)):
    m = meta1[k]; npt = m["n_prompt_tokens"]; n = len(m["tokens"]); sl = slice(npt - 1, npt - 1 + n)
    h = acts1["resid.46"][k][sl].float() + rmsn(acts1["attn_out.47"][k][sl].float(), wpa)
    XH1.append(rmsn(h, wpre))
XH1 = torch.cat(XH1)
dX = XH1 - X47o
g = parts["g"]; delta = parts["delta"]
dx_unit = cfg["u0"] * (delta * wpre)
a_meas = (dX @ g).numpy()
w29 = parts["w29"]; b29 = X29[:, 1750] / 0.091
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
a_pred = gelu(cfg["a29s"] * (X29 @ w29 - cfg["th29"] * b29).numpy())
print("positions", len(a_meas))
big = np.where(np.abs(a_pred) > 0.3)[0]
print("n flagged (|a_pred|>0.3):", len(big))
print("corr a_meas vs a_pred (flagged):", np.corrcoef(a_meas[big], a_pred[big])[0, 1], " ratio median", np.median(a_meas[big] / a_pred[big]))
small = np.where(np.abs(a_pred) < 0.01)[0]
print("unflagged positions: a_meas std", a_meas[small].std(), "max", np.abs(a_meas[small]).max())
# residual change not explained by flag
resid = dX - torch.tensor(a_meas)[:, None] * dx_unit[None, :]
print("|dX| median (flagged)", float(dX[big].norm(dim=1).median()), " |unexplained| median (flagged)", float(resid[big].norm(dim=1).median()), " |dX| median unflagged", float(dX[small].norm(dim=1).median()), " |x_hat| median", float(X47o.norm(dim=1).median()))
# logprob changes
POS = P["POS"]
d = np.array([meta1[k]["logprobs"][t] - meta0[k]["logprobs"][t] for (k, t) in POS])
print("positions with |dlogprob|>0.5:", int((np.abs(d) > 0.5).sum()), " >2:", int((np.abs(d) > 2).sum()))
cat = {pp["id"]: pp["cat"] for pp in practice}
order = np.argsort(d)
for j in order[:12]:
    k, t = POS[j]; print("  ", items[k].get("id", ("gen", k)), cat.get(items[k].get("id"), "GEN"), repr(meta0[k]["tokens"][t]), round(float(d[j]), 2), "a_pred", round(float(a_pred[j]), 2), "a_meas", round(float(a_meas[j]), 2))
# unexpected changes: |d|>0.5 where a_pred small
unexp = np.where((np.abs(d) > 0.5) & (np.abs(a_pred) < 0.1))[0]
print("unexpected changes (|d|>0.5 & not flagged):", len(unexp))
for j in unexp[:15]:
    k, t = POS[j]; print("  ", items[k].get("id", ("gen", k)), cat.get(items[k].get("id"), "GEN"), repr(meta0[k]["tokens"][t]), round(float(d[j]), 2), "a_meas", round(float(a_meas[j]), 2), "ctx", meta0[k]["tokens"][max(0, t-3):t])
np.save(WORK + "/ver2_ameas.npy", a_meas); np.save(WORK + "/ver2_d.npy", d)
torch.save({"XH1": XH1.half()}, WORK + "/ver2_xh.pt")
print(time.time() - t0)
