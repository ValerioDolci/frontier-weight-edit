import torch, json, numpy as np
A = load_acts("L47a"); acts, meta, items = A["acts"], A["meta"], A["items"]
D = W("model.language_model.layers.47.mlp.down_proj.weight")
dn = D.norm(dim=0)
s2 = torch.zeros(15360, dtype=torch.float64); mx = torch.zeros(15360); n = 0
for k in range(len(items)):
    m = meta[k]; npt = m["n_prompt_tokens"]; nt = len(m["tokens"])
    a = acts["mlp_act.47"][k][npt - 1: npt - 1 + nt].float()
    s2 += (a.double() ** 2).sum(0); mx = torch.maximum(mx, a.abs().max(0).values); n += a.shape[0]
rms_a = (s2 / n).sqrt().float()
contrib = rms_a * dn
order = torch.argsort(contrib)
print("n positions", n)
print("lowest-contribution neurons:", [(int(i), round(float(contrib[i]), 5), round(float(rms_a[i]), 5), round(float(mx[i]), 4)) for i in order[:15]])
print("contrib quantiles", np.round(np.quantile(contrib.numpy(), [0, 0.01, 0.1, 0.5, 0.9, 1]), 4))
# typical |y| for scale
ys = []
for k in range(0, len(items), 7):
    m = meta[k]; npt = m["n_prompt_tokens"]; nt = len(m["tokens"])
    ys.append(acts["mlp_out.47"][k][npt - 1: npt - 1 + nt].float())
ys = torch.cat(ys); print("mlp_out47 norm quantiles", np.round(np.quantile(ys.norm(dim=1).numpy(), [0.01, 0.5, 0.99]), 2), "rms", np.round(np.quantile(ys.pow(2).mean(1).sqrt().numpy(), [0.01, 0.5, 0.99]), 3))
torch.save({"rms_a": rms_a, "max_a": mx, "contrib": contrib}, WORK + "/neur47.pt")
