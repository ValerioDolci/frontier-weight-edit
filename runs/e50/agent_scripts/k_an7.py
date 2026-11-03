import numpy as np, torch
R = np.load(f"{WORK}/dp_resid.npy"); AO = np.load(f"{WORK}/dp_attn_out.npy"); MO = np.load(f"{WORK}/dp_mlp_out.npy"); MA = np.load(f"{WORK}/dp_mlp_act.npy")
def rms(x, g, eps=1e-6):
    return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * g
def P(L, nm): return W(f"model.language_model.layers.{L}.{nm}.weight")
for L in [30, 31]:
    r_prev = torch.tensor(R[L-1][:64].astype(np.float32)); a = torch.tensor(AO[L][:64].astype(np.float32))
    m = torch.tensor(MO[L][:64].astype(np.float32)); ma = torch.tensor(MA[L][:64].astype(np.float32))
    h = r_prev + rms(a, P(L, "post_attention_layernorm"))
    x = rms(h, P(L, "pre_feedforward_layernorm"))
    G = P(L, "mlp.gate_proj"); U = P(L, "mlp.up_proj"); D = P(L, "mlp.down_proj")
    act = torch.nn.functional.gelu(x @ G.T, approximate="tanh") * (x @ U.T)
    print(L, "mlp_act rel err", ((act - ma).norm() / ma.norm()).item())
    mo = act @ D.T
    print(L, "mlp_out rel err", ((mo - m).norm() / m.norm()).item())
    s = W(f"model.language_model.layers.{L}.layer_scalar").item()
    out = (h + rms(m, P(L, "post_feedforward_layernorm"))) * s
    rL = torch.tensor(R[L][:64].astype(np.float32))
    print(L, "resid rel err", ((out - rL).norm() / rL.norm()).item())
    # x stats: is there a near-constant direction?
    xm = x.mean(0); proj = (x @ xm) / xm.norm()
    print(L, "x rms", x.pow(2).mean().sqrt().item(), "|mean x|", xm.norm().item(), "proj on mean: mean %.2f std %.2f" % (proj.mean().item(), proj.std().item()))
