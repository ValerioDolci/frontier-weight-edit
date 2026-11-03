import torch, numpy as np
print([n for n in names() if "scalar" in n][:5])
for L in [44, 45, 46, 47]:
    D = W(f"model.language_model.layers.{L}.mlp.down_proj.weight")
    G = W(f"model.language_model.layers.{L}.mlp.gate_proj.weight")
    U = W(f"model.language_model.layers.{L}.mlp.up_proj.weight")
    dn = D.norm(dim=0); gn = G.norm(dim=1); un = U.norm(dim=1)
    print(L, "down col norm quantiles", np.round(np.quantile(dn.numpy(), [0, 0.001, 0.01, 0.1, 0.5, 0.9, 1]), 4))
    print("   gate row norm q", np.round(np.quantile(gn.numpy(), [0, 0.01, 0.5, 0.99]), 4), " up row norm q", np.round(np.quantile(un.numpy(), [0, 0.01, 0.5, 0.99]), 4))
    idx = torch.argsort(dn)[:8]
    print("   smallest down cols", [(int(i), round(float(dn[i]), 5), round(float(gn[i]), 4), round(float(un[i]), 4)) for i in idx])
pw = W("model.language_model.layers.47.pre_feedforward_layernorm.weight")
print("pre_ff47 w[1750]", float(pw[1750]), "w[830]", float(pw[830]))
po = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
print("post_ff47 stats", float(po.mean()), float(po.std()), float(po.min()), float(po.max()))
nw = W("model.language_model.norm.weight")
print("final norm w[1750]", float(nw[1750]), "w[830]", float(nw[830]))
