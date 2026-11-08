import json, numpy as np, re, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/Qwen3.8-27B"); D = "./taiwan/lens/"
cjk = re.compile(r"[一-鿿]")
dec = {}
def d(i):
    if i not in dec: dec[i] = tok.decode([int(i)])
    return dec[i]
rows = []
for l in range(40, 64):
    z = np.load(f"{D}L{l}.npz"); top = z["top"]; kurt = z["kurt"]
    for j in np.argsort(-kurt)[:2000]:
        ts = [d(t) for t in top[j][:5]]
        long = [t for t in ts if len(cjk.findall(t)) >= 4]
        if long: rows.append((float(kurt[j]), l, int(j), ts))
rows.sort(key=lambda r: -r[0])
print(len(rows), "neuroni L40-63 (tra i 2000 più concentrati per layer) con un token cinese di ≥4 caratteri nei top-5")
for r in rows[:70]: print("L%d n%d k=%.0f  %s" % (r[1], r[2], r[0], " | ".join(t.strip() for t in r[3])))
