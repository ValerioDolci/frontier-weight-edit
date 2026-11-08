import json, numpy as np
V = json.load(open("taiwan/unblock_variants.json")); off = V["off32"]["neurons"]
T = np.load("taiwan/dec_TIAN_27b/attr_TIAN.npy"); NI = T.shape[1]; o = np.argsort(T.ravel())
def comb(k):
    d = {l: list(js) for l, js in off.items()}
    for f in o[:k]:
        l, j = divmod(int(f), NI); d.setdefault(str(l), [])
        if j not in d[str(l)]: d[str(l)].append(j)
    return d
C = {"comb_t16": {"neurons": comb(16), "scale": 0.0}, "comb_t32": {"neurons": comb(32), "scale": 0.0}}
json.dump(C, open("taiwan/comb_variants.json", "w")); print({k: sum(len(x) for x in v["neurons"].values()) for k, v in C.items()})
