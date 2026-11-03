for nm in ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]:
    try:
        t = W(nm)
        print(nm, tuple(t.shape))
    except Exception as e:
        print(nm, "ERR", repr(e)[:200])
# layer_scalar?
print([n for n in names() if "scalar" in n][:5])
print([n for n in names("layers.0.")])
try:
    print(W("model.language_model.layers.0.layer_scalar"))
except Exception as e:
    print("layer_scalar ERR", repr(e)[:200])
import inspect
for f in [W, names, save_edit, drop_edit, edits, load_acts]:
    try:
        print(f.__name__, inspect.signature(f), (f.__doc__ or "")[:300])
    except Exception as e:
        print(f, e)
