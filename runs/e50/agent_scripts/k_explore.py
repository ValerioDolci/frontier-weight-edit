import os, sys, json
print("WORK =", WORK)
ns = names()
print("n tensors:", len(ns))
for n in ns[:40]:
    print(n)
print("...")
# show layer 0 and layer 5 tensors with shapes
for n in names("layers.5."):
    t = W(n)
    print(n, tuple(t.shape), t.dtype)
print("practice[0:3] =", practice[:3])
print("n practice:", len(practice))
from collections import Counter
print(Counter(p["cat"] for p in practice))
# what else is in globals
print(sorted(k for k in globals().keys() if not k.startswith("_")))
