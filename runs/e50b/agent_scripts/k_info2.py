ns = names()
print([n.split("layers.5.")[1] for n in ns if ".layers.5." in n])
print([n.split("layers.4.")[1] for n in ns if ".layers.4." in n])
for L in [4, 5]:
    for n in ns:
        if f".layers.{L}." in n:
            print(n, tuple(W(n).shape))
import collections
print(collections.Counter(p["cat"] for p in practice))
print(practice[609])
