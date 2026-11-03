g = save_edit.__globals__
print(type(g), len(g))
ks = [k for k in g.keys() if not k.startswith("__")]
print(ks[:80])
for k in ks:
    v = g[k]
    if isinstance(v, dict):
        print("dict", k, len(v), list(v.keys())[:6])
