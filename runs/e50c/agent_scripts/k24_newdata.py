import numpy as np, torch, json
r3 = load_acts("r3")
M3, I3, A3 = r3["meta"], r3["items"], r3["acts"]
print(len(I3), sorted(A3.keys()))
NDI, NDD, NNEG = 120, 90, 60
def first_diff_tok(a, b):
    ta, tb = M3[a]["tokens"], M3[b]["tokens"]
    for t in range(min(len(ta), len(tb))):
        if ta[t] != tb[t]:
            return t
    return None
dec3 = {}; marg3 = {}
for k in range(NDI):
    a, b = k, NDI + k   # a = imp, b = trap
    assert I3[a]["en"] == I3[b]["en"]
    t = first_diff_tok(a, b)
    dec3[a] = M3[a]["n_prompt_tokens"] + t - 1
    marg3[a] = M3[a]["logprobs"][t] - M3[b]["logprobs"][t]
AUX_AV = {"Ave", "▁aveva", "▁avevano", "▁avevamo", "▁avevo", "▁avevi", "▁ave", "▁avev"}
AUX_ES = {"E", "Era", "Er", "▁era", "▁erano", "▁eravamo", "▁ero", "▁eri", "▁er", "Eran"}
nf = 0
for k in range(2 * NDI, 2 * NDI + NDD):
    toks = M3[k]["tokens"]; t = None
    for j, tk in enumerate(toks):
        if tk in AUX_AV or tk in AUX_ES or tk in {"▁l", "▁L"}:
            if tk in {"▁l", "▁L"}:
                continue
            if tk in AUX_ES and j + 1 < len(toks) and (toks[j+1].startswith("▁stat") or toks[j+1].startswith("▁rim")):
                t = j + 1
            else:
                t = j
            break
    if t is None:
        nf += 1; print("no aux:", I3[k]["it"][:70], toks[:12])
        continue
    dec3[k] = M3[k]["n_prompt_tokens"] + t - 1
print("DD without aux:", nf)
m = np.array([marg3[k] for k in range(NDI)])
print(f"new DI margins (imp - trap at decision): >0: {np.sum(m>0)}/{NDI}; mean {m.mean():.2f}; quantiles {np.percentile(m,[10,25,50,75,90]).round(2)}")
order = np.argsort(m)
print("most negative:", [(int(k), round(float(m[k]),2), I3[k]["it"][:50]) for k in order[:12]])
# sanity: decision tokens for some DI
for k in [0, 8, 20, 60, 83]:
    t = dec3[k] - M3[k]["n_prompt_tokens"] + 1
    print(k, M3[k]["tokens"][t], "|", M3[NDI+k]["tokens"][t], "| margin", round(marg3[k], 2))
json.dump({"dec3": {str(k): v for k, v in dec3.items()}, "marg3": {str(k): v for k, v in marg3.items()}}, open(f"{WORK}/dec_r3.json", "w"))
