import os, sys
print("WORK =", WORK)
ns = names()
print(len(ns))
for n in ns[:40]:
    print(n)
print("...")
print([n for n in ns if ".20." in n])
print(practice[:3])
import torch, numpy as np
print(torch.__version__, np.__version__)
print(type(W(ns[0])), W(ns[0]).shape, W(ns[0]).dtype)
