for nm in ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]:
    try:
        t = W(nm)
        print(nm, tuple(t.shape))
    except Exception as e:
        print(nm, "ERR", repr(e)[:200])
g = [k for k in globals().keys() if not k.startswith("_")]
print(g)
import inspect
for f in ["load_acts", "save_edit", "drop_edit", "edits", "W", "names"]:
    try:
        print(f, inspect.signature(globals()[f]))
        d = globals()[f].__doc__
        if d: print("   ", d[:500])
    except Exception as e:
        print(f, "ERR", e)
import os
print(os.listdir(WORK))
