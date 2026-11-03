import numpy as np, torch
ls = []
for L in range(48):
    ls.append(W(f"model.language_model.layers.{L}.layer_scalar").item())
print("layer_scalar:", [round(v, 3) for v in ls])
for L in [24, 28, 30, 32, 34, 36, 40]:
    for nm in ["post_attention_layernorm", "post_feedforward_layernorm", "pre_feedforward_layernorm", "input_layernorm"]:
        g = W(f"model.language_model.layers.{L}.{nm}.weight").numpy()
        print(L, nm, "mean %.3f std %.3f min %.4f max %.3f  frac<0.05: %.3f" % (g.mean(), g.std(), g.min(), g.max(), (np.abs(g) < 0.05).mean()))
R = np.load(f"{WORK}/dp_resid.npy"); MO = np.load(f"{WORK}/dp_mlp_out.npy"); AO = np.load(f"{WORK}/dp_attn_out.npy")
for L in [24, 28, 30, 32, 34, 36, 40, 47]:
    r = R[L].astype(np.float32); m = MO[L].astype(np.float32); a = AO[L].astype(np.float32)
    print(L, "resid rms %.2f" % np.sqrt((r**2).mean(1)).mean(), "mlp_out rms %.3f" % np.sqrt((m**2).mean(1)).mean(), "attn_out rms %.3f" % np.sqrt((a**2).mean(1)).mean())
