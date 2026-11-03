import numpy as np, torch
n = names()
print(len(n))
for x in names("layers.0."):
    t = W(x)
    print(x, tuple(t.shape), float(t.float().abs().mean()))
for x in names("layers.5."):
    t = W(x)
    print(x, tuple(t.shape))
print([x for x in globals().keys() if not x.startswith('_')])
print(WORK)
print(practice[:2])
# layer scalars
for L in range(48):
    ls = names(f"layers.{L}.layer_scalar")
    if ls:
        print(L, ls, W(ls[0]).flatten()[:4].tolist())
