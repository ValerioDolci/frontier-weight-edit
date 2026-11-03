import torch, json, numpy as np, collections
D0 = torch.load(WORK + "/L47pos.pt"); XH = D0["XH"].float(); POS = D0["POS"]; Y = D0["Y"].float(); R47 = D0["R47"].float()
A0 = load_acts("L47a"); meta0, items = A0["meta"], A0["items"]
A1 = load_acts("ver1"); meta1 = A1["meta"]
prm = json.load(open(WORK + "/edit1_params.json"))
w = torch.tensor(np.load(WORK + "/probe47_w.npy"), dtype=torch.float32)
mhat = torch.load(WORK + "/push47.pt")["mhat"]; p = torch.load(WORK + "/push47c.pt")["p"]
def rmsn(x, ww, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * ww
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight"); E = W("model.language_model.embed_tokens.weight")
SC = 0.0478
b = XH @ mhat; s = XH @ w
z = prm["alpha"] * (s - prm["theta"] * b)
a = torch.nn.functional.gelu(z, approximate="tanh") * b
# actual deltas
act_d = [];
for j, (k, t) in enumerate(POS):
    act_d.append(meta1[k]["logprobs"][t] - meta0[k]["logprobs"][t])
act_d = np.array(act_d)
fire = torch.where(a.abs() * prm["beta"] > 1e-2)[0]
print("firing positions (pred):", len(fire), " positions with |actual delta|>0.05:", int((np.abs(act_d) > 0.05).sum()))
# predicted new logprob of the teacher-forced token at firing positions
tokid = json.load(open(WORK + "/tokid.json"))
out0 = rmsn(Y, wpf)
pred_d = np.zeros(len(POS))
for s0 in range(0, len(fire), 512):
    idx = fire[s0:s0+512]
    y2 = Y[idx] + a[idx][:, None] * prm["beta"] * p[None, :]
    r2 = R47[idx] + SC * (rmsn(y2, wpf) - out0[idx])
    lg2 = torch.log_softmax(30.0 * torch.tanh((rmsn(r2, nw) @ E.T) / 30.0), 1)
    lg0 = torch.log_softmax(30.0 * torch.tanh((rmsn(R47[idx], nw) @ E.T) / 30.0), 1)
    for q, j in enumerate(idx.tolist()):
        k, t = POS[j]
        tok = meta0[k]["tokens"][t]
        lp0 = meta0[k]["logprobs"][t]
        # find id: argmin |lg0 - lp0|
        tid = tokid.get(tok)
        if tid is None:
            tid = int((lg0[q] - lp0).abs().argmin())
        pred_d[j] = float(lg2[q, tid] - lg0[q, tid])
big = np.where((np.abs(act_d) > 0.05) | (np.abs(pred_d) > 0.05))[0]
print("n big", len(big))
err = act_d[big] - pred_d[big]
print("actual vs predicted delta: corr", np.corrcoef(act_d[big], pred_d[big])[0, 1], "mean abs err", np.abs(err).mean(), "max abs err", np.abs(err).max())
cat = {pp["id"]: pp["cat"] for pp in practice}
order = np.argsort(-np.abs(act_d))
for j in order[:40]:
    k, t = POS[j]
    c = cat[items[k]["id"]] if k < 610 else "GEN"
    print(c, items[k].get("id", k), t, repr(meta0[k]["tokens"][t]), "lp0", meta0[k]["logprobs"][t], "act_d", round(act_d[j], 2), "pred_d", round(pred_d[j], 2), "a", round(float(a[j]), 2))
# positions with large unpredicted changes
unp = np.where((np.abs(act_d) > 0.3) & (np.abs(pred_d) < 0.05))[0]
print("unpredicted large changes:", len(unp))
for j in unp[:20]:
    k, t = POS[j]; print("   ", items[k].get("id", k), t, repr(meta0[k]["tokens"][t]), round(act_d[j], 2), "a", round(float(a[j]), 3))
