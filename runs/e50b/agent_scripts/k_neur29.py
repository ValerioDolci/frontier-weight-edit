import torch, numpy as np, time
t0 = time.time()
L29 = torch.load(WORK + "/L29pos.pt")
X = torch.cat([L29["prac"]["XH"], L29["new"]["XH"]]).float()
idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))[:8000]
Xs = X[idx]
G = W("model.language_model.layers.29.mlp.gate_proj.weight"); U = W("model.language_model.layers.29.mlp.up_proj.weight")
D = W("model.language_model.layers.29.mlp.down_proj.weight")
act = torch.nn.functional.gelu(Xs @ G.T, approximate="tanh") * (Xs @ U.T)
rms_a = act.pow(2).mean(0).sqrt(); mx = act.abs().max(0).values
contrib = rms_a * D.norm(dim=0)
order = torch.argsort(contrib)
print("lowest-contribution neurons L29:", [(int(i), round(float(contrib[i]), 5), round(float(rms_a[i]), 5), round(float(mx[i]), 4)) for i in order[:10]])
print("contrib quantiles", np.round(np.quantile(contrib.numpy(), [0, 0.01, 0.1, 0.5, 0.9, 1]), 4))
Y = L29["prac"]["Y"][:4000].float()
print("|y29| median", float(Y.norm(dim=1).median()), time.time() - t0)
# residual rms at layer 29 output (needed for flag scaling)
A = load_acts("L29a"); meta = A["meta"]
rr = []
for k in range(0, 610, 10):
    m = meta[k]; npt = m["n_prompt_tokens"]
    r = A["acts"]["resid.29"][k][npt - 1:].float()
    rr.append(r.pow(2).mean(1).sqrt())
rr = torch.cat(rr)
print("resid29 rms quantiles", np.round(np.quantile(rr.numpy(), [0.01, 0.1, 0.5, 0.9, 0.99]), 3))
