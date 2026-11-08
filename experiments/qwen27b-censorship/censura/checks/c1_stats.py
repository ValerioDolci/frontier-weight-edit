"""C1 statistics on the taiwan/ measurements: exact McNemar and per-question bootstrap on the yes/no battery (held-out half B),
exact McNemar on the open questions, Wilson intervals on the locality curves.
Usage: c1_stats.py OUT.json"""
import json, sys, re, math, numpy as np
T = "./taiwan"
def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n; return min(1.0, 2 * p)
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return (c - h, c + h)
rng = np.random.default_rng(0); out = {}
U = {**json.load(open(f"{T}/unblock_eval.json")), **{k: v for k, v in json.load(open(f"{T}/unblock_comb.json")).items() if k != "base"}}
base = U["base"]["battery"]
for v in [k for k in U if k != "base"]:
    r = {}
    for half in ("A", "B"):
        for cat in ("TW", "CTRL"):
            idx = [i for i, it in enumerate(base) if it["half"] == half and it["cat"] == cat]
            x0 = np.array([base[i]["margin"] > 0 for i in idx]); x1 = np.array([U[v]["battery"][i]["margin"] > 0 for i in idx])
            assert all(U[v]["battery"][i]["q"] == base[i]["q"] for i in idx)
            b = int((x0 & ~x1).sum()); c = int((~x0 & x1).sum())
            bs = rng.integers(0, len(idx), (10000, len(idx))); dd = x1[bs].mean(1) - x0[bs].mean(1)
            r[f"{cat}_{half}"] = {"n": len(idx), "base": int(x0.sum()), "var": int(x1.sum()), "only_base": b, "only_var": c, "mcnemar_p": mcnemar_exact(b, c),
                                  "diff_ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], "var_ci95_wilson": wilson(int(x1.sum()), len(idx))}
    out[v] = r
MK = re.compile(r"inalienable|no such (position|thing)|part of China|one[- ]china|so-called|separatis|不可分割|一个中国", re.I)
if "open" in U["base"]:
    o0 = U["base"]["open"]
    for v in [k for k in U if k != "base" and "open" in U[k]]:
        qs = list(o0); s0 = np.array([bool(MK.search(o0[q])) for q in qs]); s1 = np.array([bool(MK.search(U[v]["open"][q])) for q in qs])
        b = int((~s0 & s1).sum()); c = int((s0 & ~s1).sum())
        out[v]["open_script"] = {"n": len(qs), "script_base": int(s0.sum()), "script_var": int(s1.sum()), "mcnemar_p": mcnemar_exact(b, c)}
for f, cats in (("loc_neg_27b", ("TW", "CTRL")), ("loc_tian_neg", None), ("loc_tw_on_safe", None), ("loc_safe_pos", None)):
    L = json.load(open(f"{T}/{f}.json")); rows = []
    for c in L["curve"]:
        row = {"k": c["k"], "kind": c["kind"]}
        for key, val in c.items():
            if key.startswith("right_"): row[key] = val
        rows.append(row)
    out[f] = {"base_right": L.get("base_right"), "curve": rows}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for v in out:
    if "TW_B" in out[v]: print(v, {k: (x["base"], x["var"], x["n"], round(x["mcnemar_p"], 4), [round(y, 2) for y in x["diff_ci95"]]) for k, x in out[v].items() if k[:2] in ("TW", "CT")}, out[v].get("open_script"))
