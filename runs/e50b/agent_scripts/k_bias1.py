import torch, numpy as np
A = load_acts("main1")
acts = A["acts"]
for L in [26, 28, 30, 32]:
    X = torch.cat([t.float() for t in acts[f"resid.{L}"][:610]], 0)
    rms = X.pow(2).mean(1, keepdim=True).sqrt()
    Xn = X / rms
    mu = Xn.mean(0); sd = Xn.std(0)
    ratio = (mu.abs() / (sd + 1e-6))
    top = ratio.topk(6)
    print(L, "npos", X.shape[0], "rms quantiles", np.round(np.quantile(rms.squeeze().numpy(), [0.01, 0.5, 0.99]), 1))
    for v, i in zip(top.values, top.indices):
        print("   dim", int(i), "mean", round(float(mu[i]), 3), "sd", round(float(sd[i]), 3), "ratio", round(float(v), 2))
    # projection onto mean direction
    m = mu / mu.norm()
    pr = Xn @ m
    print("   proj on mean dir: mean", round(float(pr.mean()), 3), "sd", round(float(pr.std()), 3), "min", round(float(pr.min()), 3))
