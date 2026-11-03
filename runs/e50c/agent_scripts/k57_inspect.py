import json, numpy as np
M = json.load(open(f"{WORK}/meta_r2_r3.json"))
print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in M.items()})
for k, v in M.items():
    if isinstance(v, list): print(k, v[:8])
    elif isinstance(v, dict): print(k, list(v.items())[:8])
meta = dict(np.load(f"{WORK}/Xmeta.npz"))
print({k: (v.dtype, v.shape) for k, v in meta.items()})
print("groups:", {g: int((meta['grp'] == g).sum()) for g in np.unique(meta['grp'])})
pf2 = (meta["grp"] == "Pf") & (meta["src"] == 2)
print("Pf src2 items:", meta["item"][pf2][:40])
D = dict(np.load(f"{WORK}/dirdata.npz"))
print({k: (v.dtype, v.shape) for k, v in D.items()})
print(type(practice), (list(practice.keys())[:5] if isinstance(practice, dict) else (practice[:2] if hasattr(practice, '__getitem__') else practice)))
