print(repr(SESSION)[:500])
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as e:
    print("no transformers", e)
E = W("model.language_model.embed_tokens.weight")
print(E.shape, E.dtype, float(E.norm(dim=1).mean()))
nw = W("model.language_model.norm.weight")
print("final norm w stats", float(nw.mean()), float(nw.std()), float(nw.min()), float(nw.max()))
for L in [10, 20, 30, 40]:
    for k in ["post_feedforward_layernorm", "pre_feedforward_layernorm", "post_attention_layernorm", "input_layernorm"]:
        w = W(f"model.language_model.layers.{L}.{k}.weight")
        print(L, k, round(float(w.mean()),3), round(float(w.std()),3), round(float(w.min()),3), round(float(w.max()),3))
