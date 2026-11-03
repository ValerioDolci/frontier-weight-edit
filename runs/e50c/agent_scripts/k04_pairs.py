import numpy as np, torch, json
r1 = load_acts("r1")
meta, items = r1["meta"], r1["items"]
acts = r1["acts"]
LAYERS = sorted(int(k.split(".")[1]) for k in acts)
print("layers", LAYERS)
n = len(items)
# check rows
bad = 0
for i in range(n):
    rows = acts["resid.4"][i].shape[0]
    if rows != meta[i]["n_prompt_tokens"] + len(meta[i]["tokens"]):
        bad += 1
print("row mismatches:", bad, "example rows", acts["resid.4"][0].shape, meta[0]["n_prompt_tokens"], len(meta[0]["tokens"]))

# index groups
G = {}
G["DIW"] = list(range(0, 55)); G["DIC"] = list(range(55, 110)); G["DIR"] = list(range(110, 314))
G["DD"] = list(range(314, 444)); G["DDA"] = list(range(444, 485)); G["DURP"] = list(range(485, 545)); G["PERF"] = list(range(545, 605))
for k, v in G.items():
    print(k, len(v), items[v[0]]["id"], items[v[-1]]["id"])

def first_diff(a, b):
    ta, tb = meta[a]["tokens"], meta[b]["tokens"]
    for t in range(min(len(ta), len(tb))):
        if ta[t] != tb[t]:
            return t
    return min(len(ta), len(tb))

pairs = []  # (kind, id, idx_trap, idx_imp, t)
for i in range(55):
    a, b = G["DIW"][i], G["DIC"][i]
    assert items[a]["id"] == items[b]["id"]
    pairs.append(("DI", items[a]["id"], a, b, first_diff(a, b)))
dd_idx = {items[j]["id"]: j for j in G["DD"]}
# DD model outputs that are imperfetto (alt is trapassato)
dd_model_imp = {9, 97, 107, 144, 196, 252, 340, 550, 566}
for j in G["DDA"]:
    iid = items[j]["id"]; m = dd_idx[iid]
    if iid in dd_model_imp:
        pairs.append(("DD", iid, j, m, first_diff(m, j)))
    else:
        pairs.append(("DD", iid, m, j, first_diff(m, j)))

rows = []
for kind, iid, a, b, t in pairs:
    lpa = meta[a]["logprobs"][t]; lpb = meta[b]["logprobs"][t]
    rows.append(dict(kind=kind, id=iid, a=a, b=b, t=t, tok_trap=meta[a]["tokens"][t], lp_trap=lpa,
                     tok_imp=meta[b]["tokens"][t], lp_imp=lpb, sum_trap=meta[a]["sum_logprob"], sum_imp=meta[b]["sum_logprob"],
                     drow=meta[a]["n_prompt_tokens"] + t - 1))
for r in rows:
    print(f'{r["kind"]} {r["id"]:4d} t={r["t"]:2d} trap {r["tok_trap"]!r:>10} {r["lp_trap"]:7.3f} | imp {r["tok_imp"]!r:>10} {r["lp_imp"]:7.3f} | margin(imp-trap)={r["lp_imp"]-r["lp_trap"]:7.3f}  seq {r["sum_imp"]-r["sum_trap"]:7.2f}')
json.dump(rows, open(f"{WORK}/pairs_r1.json", "w"))
