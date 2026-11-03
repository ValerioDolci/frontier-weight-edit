import numpy as np, torch
r1 = load_acts("r1")
acts = r1["acts"]; meta = r1["meta"]
for L in [26, 28, 29, 30, 31, 32, 33, 34, 36, 38]:
    pre = W(f"model.language_model.layers.{L}.pre_feedforward_layernorm.weight").numpy()
    post = W(f"model.language_model.layers.{L}.post_feedforward_layernorm.weight").numpy()
    ina = W(f"model.language_model.layers.{L}.input_layernorm.weight").numpy()
    pa = W(f"model.language_model.layers.{L}.post_attention_layernorm.weight").numpy()
    print(f"L{L}: pre_ff mean={pre.mean():.3f} rms={np.sqrt((pre**2).mean()):.3f} min={pre.min():.3f} max={pre.max():.3f} | post_ff mean={post.mean():.3f} rms={np.sqrt((post**2).mean()):.3f} min={post.min():.4f} max={post.max():.3f} | post_attn rms={np.sqrt((pa**2).mean()):.3f}")
# massive dims in residual at L28/L31/L34 (completion positions of first 100 items)
for L in [28, 31, 34]:
    X = torch.cat([acts[f"resid.{L}"][i][meta[i]["n_prompt_tokens"]-1:].float() for i in range(0, 600, 6)]).numpy()
    rms = np.sqrt((X**2).mean(1, keepdims=True))
    Z = X / rms
    m = Z.mean(0); s = Z.std(0)
    top = np.argsort(-np.abs(m))[:8]
    print(f"L{L} resid rms mean={rms.mean():.2f}; top |mean| dims of normalized resid:", [(int(d), round(float(m[d]),2), round(float(s[d]),2)) for d in top])
    pre = W(f"model.language_model.layers.{L+1}.pre_feedforward_layernorm.weight").numpy()
    print("   next-layer pre_ff weight at those dims:", [round(float(pre[d]),3) for d in top])
    # BOS / first prompt positions
    B = acts[f"resid.{L}"][0][:5].float().numpy()
    Bz = B / np.sqrt((B**2).mean(1, keepdims=True))
    print("   first 5 prompt positions normalized at top dims:", [[round(float(Bz[p, d]),1) for d in top[:4]] for p in range(5)])
