import numpy as np, torch, json
print(edits())
b2 = json.load(open(f"{WORK}/build2.json"))
for L, nn in [(30, [13770, 12872]), (40, b2["n40"])]:
    for nm in ["gate_proj", "up_proj", "down_proj"]:
        orig = W(f"model.language_model.layers.{L}.mlp.{nm}.weight")
        print(L, nm, tuple(orig.shape), "neurons", nn)
print("thresholds: t30=-0.06, t40=-0.03; kappa=40, c=4; alpha30=28.08 (24*d_imp + 8*d_start), alpha40=%.2f" % b2["alpha40"])
