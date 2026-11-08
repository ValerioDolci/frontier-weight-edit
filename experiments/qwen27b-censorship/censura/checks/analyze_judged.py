"""Analysis of judge labels for C6 (Q4 vs bf16), C7 (safety) and E11 (Pan & Xu).
Reads RES_DIR/judge_{c6,c7,e11}_cache.json and the inputs; writes RES_DIR/{c6,c7,e11}_summary.json.
Usage: analyze_judged.py RES_DIR"""
import json, sys, os, math, collections
RD = sys.argv[1]; CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}
def load(n):
    f = f"{RD}/judge_{n}_cache.json"; return json.load(open(f)) if os.path.exists(f) else None
def wilson(k, n, z=1.96):
    if not n: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [round(c - h, 3), round(c + h, 3)]
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
J = load("c6")
if J:
    rows = json.load(open(f"{RD}/gpu/c6.json")); ch = []; tot = 0
    for r in rows:
        j = J.get(f"c6|{r['id']}")
        if not j or "type" not in j: continue
        tot += 1; q4 = r["type_q4"] in CENS; bf = j["type"] in CENS; ch.append((q4, bf))
    flips = sum(a != b for a, b in ch); s = {"n": tot, "label_changed": flips, "rate": round(flips / tot, 3), "ci95": wilson(flips, tot),
         "q4_cens_bf16_bal": sum(a and not b for a, b in ch), "q4_bal_bf16_cens": sum(b and not a for a, b in ch),
         "cens_rate_q4": round(sum(a for a, _ in ch) / tot, 3), "cens_rate_bf16": round(sum(b for _, b in ch) / tot, 3)}
    json.dump(s, open(f"{RD}/c6_simple_summary.json", "w"), indent=1); print("C6", s)
J = load("c7")
if J:
    inp = [json.loads(l) for l in open(f"{RD}/judge_c7.jsonl")]; R = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in inp:
        j = J.get(it["key"]);
        if not j or "refused" not in j: continue
        _, st, i, v = it["key"].split("|"); R[v][st].append((int(i), bool(j["refused"])))
    s = {}
    for v in R:
        s[v] = {st: {"refused": sum(x for _, x in R[v][st]), "n": len(R[v][st]), "rate": round(sum(x for _, x in R[v][st]) / max(1, len(R[v][st])), 3)} for st in ("harm", "xs")}
    for v in R:
        if v == "base": continue
        for st in ("harm", "xs"):
            b0 = dict(R["base"][st]); b1 = dict(R[v][st]); k = set(b0) & set(b1)
            s[v][st]["mcnemar_vs_base"] = mcnemar(sum(b0[i] and not b1[i] for i in k), sum(b1[i] and not b0[i] for i in k))
            s[v][st]["delta_pts"] = round(100 * (s[v][st]["rate"] - s["base"][st]["rate"]), 1)
    json.dump(s, open(f"{RD}/c7_summary.json", "w"), indent=1); print("C7", json.dumps(s, indent=0))
J = load("e11")
if J:
    inp = [json.loads(l) for l in open(f"{RD}/judge_e11.jsonl")]; R = collections.defaultdict(lambda: collections.Counter())
    for it in inp:
        j = J.get(it["key"])
        if not j or "type" not in j: continue
        _, no, lg, v = it["key"].split("|"); R[(lg, v)][j["type"]] += 1; R[(lg, v)]["_n"] += 1
    s = {f"{lg}|{v}": {"n": c["_n"], "censored": sum(c[t] for t in CENS), "refusal": c["refusal"], "official_script": c["official_script"] + c["denial"] + c["sovereignty_claim"] + c["praise"],
                       "balanced": c["balanced"], "rate_censored": round(sum(c[t] for t in CENS) / max(1, c["_n"]), 3)} for (lg, v), c in sorted(R.items())}
    json.dump(s, open(f"{RD}/e11_summary.json", "w"), indent=1); print("E11", json.dumps(s, indent=0))
