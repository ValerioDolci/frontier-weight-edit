import torch, json, numpy as np
A = load_acts("L47a")
acts, meta, items = A["acts"], A["meta"], A["items"]
print(len(items), sorted(acts.keys()))
k = 0
for key in sorted(acts.keys()):
    print(key, tuple(acts[key][k].shape), acts[key][k].dtype)
m = meta[k]; print(m["n_prompt_tokens"], len(m["tokens"]))
# verify architecture: r47 = (r46 + postattn(attn47) + postff(mlp47)) * scalar
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpa = W("model.language_model.layers.47.post_attention_layernorm.weight")
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
wpre = W("model.language_model.layers.47.pre_feedforward_layernorm.weight")
r46 = acts["resid.46"][k].float(); a47 = acts["attn_out.47"][k].float(); y47 = acts["mlp_out.47"][k].float(); r47 = acts["resid.47"][k].float()
h = r46 + rmsn(a47, wpa)
pre_scale = h + rmsn(y47, wpf)
ratio = (r47 * pre_scale).sum(1) / (pre_scale * pre_scale).sum(1)
print("layer scalar estimates", ratio[:5], ratio.mean(), ratio.std())
res = r47 - ratio.mean() * pre_scale
print("residual rel err", float(res.norm() / r47.norm()))
# check mlp_act -> mlp_out
D = W("model.language_model.layers.47.mlp.down_proj.weight")
act = acts["mlp_act.47"][k].float()
print("down check rel err", float((act @ D.T - y47).norm() / y47.norm()))
# check x_hat -> mlp_act via gate/up on 3 positions
G = W("model.language_model.layers.47.mlp.gate_proj.weight"); U = W("model.language_model.layers.47.mlp.up_proj.weight")
xh = rmsn(h[:3], wpre)
act_hat = torch.nn.functional.gelu(xh @ G.T, approximate="tanh") * (xh @ U.T)
print("act check rel err", float((act_hat - act[:3]).norm() / act[:3].norm()))
