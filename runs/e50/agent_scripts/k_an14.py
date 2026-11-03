import numpy as np, json
fam = json.load(open(f"{WORK}/fa_meta.json")); items = fam["items"]; meta = fam["meta"]
m40 = neuron_mult(X40n.astype(np.float64), w40g_old, u40c, 1.0)
w30g = np.load(f"{WORK}/w30g_full.npy"); u30c = np.load(f"{WORK}/u30c.npy")
m30 = neuron_mult(X30.astype(np.float64), w30g, u30c, 1.0)
npr = np.array([m["n_prompt_tokens"] for m in meta])
rows = []
for j in np.where((kindp == "DI") & ~is_prompt & ~isdec & ((m40 > 0.1) | (m30 > 0.1)))[0]:
    i = item_of[j]; p = pos_of[j] - npr[i]
    toks = meta[i]["tokens"]
    d = dec_abs.get(i, -1) - npr[i]
    rows.append((items[i]["id"], p - d, "".join(toks[max(0, p - 4):p + 1]).replace("▁", " "), toks[p + 1] if p + 1 < len(toks) else "<end>", round(float(m30[j]), 2), round(float(m40[j]), 2)))
print(len(rows))
from collections import Counter
print("offset from decision:", Counter(r[1] for r in rows).most_common(12))
for r in rows[:60]: print(r)
