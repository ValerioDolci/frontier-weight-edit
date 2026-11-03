import numpy as np, torch, json
r2 = load_acts("r2")
M2, I2, A2 = r2["meta"], r2["items"], r2["acts"]
print(len(I2), sorted(A2.keys()))
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps) * w
def Wt(name):
    return W(name).float()
# reconstruction check on a few items
for L in [30, 31, 32, 33, 34]:
    wpa = Wt(f"model.language_model.layers.{L}.post_attention_layernorm.weight")
    wpf = Wt(f"model.language_model.layers.{L}.post_feedforward_layernorm.weight")
    ratios = []
    for i in [0, 100, 300, 600]:
        r_prev = A2[f"resid.{L-1}"][i].float(); a = A2[f"attn_out.{L}"][i].float(); m = A2[f"mlp_out.{L}"][i].float()
        r_next = A2[f"resid.{L}"][i].float()
        h = r_prev + rmsn(a, wpa)
        pred = h + rmsn(m, wpf)
        # scalar fit
        s = (pred * r_next).sum() / (pred * pred).sum()
        err = ((r_next - s * pred).norm() / r_next.norm()).item()
        ratios.append((round(s.item(), 4), round(err, 5)))
    print(f"L{L}: layer_scalar fit / rel err:", ratios)
