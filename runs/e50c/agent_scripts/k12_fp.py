import numpy as np, torch, json
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
L = 31
w = np.load(f"{WORK}/det_w_L{L}.npy")
NSYS = 45
cat = lambda i: "DIW" if i < 55 else ("DIR" if i < 259 else ("DD" if i < 389 else ("DURP" if i < 449 else ("PERF" if i < 509 else "GEN"))))
rows = []
for i in range(len(I2)):
    xi = mlp_in(L, i).numpy()
    s = xi @ w
    n0 = M2[i]["n_prompt_tokens"]
    for r in range(NSYS, xi.shape[0]):
        rows.append((float(s[r]), i, r, r == dec2.get(i, -1)))
rows.sort(key=lambda z: -z[0])
def ctx(i, r):
    n0 = M2[i]["n_prompt_tokens"]
    toks = M2[i]["tokens"]
    if r >= n0 - 1:
        t = r - n0 + 1  # next token predicted
        prev = "".join(toks[max(0, t-4):t]).replace("▁", " ")
        nxt = toks[t] if t < len(toks) else "<END>"
        return f"OUT ...{prev!r} -> {nxt!r}"
    return f"SRC row {r}"
print("Top non-decision rows by score:")
k = 0
for s, i, r, isdec in rows:
    if isdec: continue
    print(f"  {s:.2f} {cat(i):4s} id={I2[i].get('id')} {ctx(i, r)}")
    k += 1
    if k >= 45: break
# score of decision rows: DIW
sw = [(float(mlp_in(L, i).numpy()[dec2[i]] @ w), I2[i]["id"]) for i in range(55)]
sw.sort()
print("DIW decision scores (lowest 15):", [(round(a, 2), b) for a, b in sw[:15]])
sd = [(float(mlp_in(L, i).numpy()[dec2[i]] @ w), I2[i]["id"]) for i in range(259, 389) if i in dec2]
sd.sort(reverse=True)
print("DD decision scores (highest 20):", [(round(a, 2), b) for a, b in sd[:20]])
