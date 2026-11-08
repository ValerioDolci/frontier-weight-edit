"""C1 stability: do the top-32 neurons change with the A/B split? Seed 0 (original) and seeds 1-5.
Pre-registered threshold: mean Jaccard of the top-32 across splits >= 0.3, or held-out effect >= +4/19 in >= 4 of 5 splits.
Also reports top-100/top-1000 Jaccard and overlap with off32.
Usage: c1_stab_analyze.py OUT.json"""
import json, sys, itertools, numpy as np
W = "./taiwan"; seeds = [0, 1, 2, 3, 4, 5]
dirs = {0: f"{W}/dec_TW_27b", **{s: f"{W}/stab/dec_s{s}" for s in seeds[1:]}}
locs = {0: f"{W}/loc_neg_27b.json", **{s: f"{W}/stab/loc_s{s}.json" for s in seeds[1:]}}
top = {}
for s, d in dirs.items():
    try: T = np.load(f"{d}/attr_TW.npy")
    except Exception as e: print("manca", s, e); continue
    o = np.argsort(T.ravel()); top[s] = {k: set(map(int, o[:k])) for k in (32, 100, 1000)}
out = {"seeds": sorted(top)}
for k in (32, 100, 1000):
    js = [len(top[a][k] & top[b][k]) / len(top[a][k] | top[b][k]) for a, b in itertools.combinations(sorted(top), 2)]
    out[f"jaccard_top{k}_mean"] = float(np.mean(js)); out[f"jaccard_top{k}_min"] = float(np.min(js))
NI = np.load(f"{dirs[0]}/attr_TW.npy").shape[1]
V = json.load(open(f"{W}/unblock_variants.json"))["off32"]["neurons"]; off32 = {int(l) * NI + j for l, js in V.items() for j in js}
out["overlap_with_off32"] = {s: len(top[s][32] & off32) for s in top}
from collections import Counter
c = Counter(x for s in top for x in top[s][32]); out["core_in_ge4_splits"] = [divmod(x, NI) for x, n in c.items() if n >= 4]
eff = {}
for s, f in locs.items():
    try: L = json.load(open(f))
    except Exception as e: print("manca loc", s, e); continue
    nB = None; b = L["base_right"]["TW"]
    row = {"base_right_TW": b}
    for cpt in L["curve"]:
        if cpt["kind"] == "top" and cpt["k"] in (16, 32, 64): row[f"top{cpt['k']}_right_TW"] = cpt["right_TW"]; row[f"top{cpt['k']}_right_CTRL"] = cpt["right_CTRL"]
        if cpt["kind"] == "rand" and cpt["k"] == 32: row["rand32_right_TW"] = cpt["right_TW"]
    eff[s] = row
out["heldout"] = eff
out["splits_with_effect_ge4"] = sum(1 for s, r in eff.items() if s > 0 and "top32_right_TW" in r and (r["top32_right_TW"] - r["base_right_TW"]) * 19 >= 4 - 1e-6)
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "core_in_ge4_splits"}, indent=1)); print("nucleo", len(out["core_in_ge4_splits"]))
