"""Gemma 4 31B-it (Q4) as a non-Chinese reference on the same battery.
  prep: writes the judge inputs; ana: compares labels (same judge and prompt) Qwen vs Gemma.
Usage: gemma_compare.py prep|ana"""
import json, sys, collections
D = "./censura"; R = f"{D}/checks/results"; CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}
J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r
G = {r["id"]: r for r in map(json.loads, open(f"{R}/gpu/gen_gemma.jsonl"))}
if sys.argv[1] == "prep":
    with open(f"{R}/judge_gemma.jsonl", "w") as f:
        for i in sorted(J):
            if i in G: f.write(json.dumps({"key": f"g{i}", "q": G[i]["q"], "a": G[i]["a"]}, ensure_ascii=False) + "\n")
    E = [json.loads(l) for l in open(f"{D}/checks/data/e11_panxu.jsonl")]; X = json.load(open(f"{R}/gpu/gemma_extra.json"))
    with open(f"{R}/judge_gemma_e11.jsonl", "w") as f:
        for r, a in zip(E, X["e11_zh"]): f.write(json.dumps({"key": f"ge11|{r['no']}", "q": r["zh"], "a": a}, ensure_ascii=False) + "\n")
    print("ok", len(J))
else:
    C = json.load(open(f"{R}/judge_gemma_cache.json")); out = {}
    bt = {r["id"]: r for r in map(json.loads, open(f"{D}/battery.jsonl"))}; src = open(f"{D}/extract_foreign.py").read().splitlines()
    exec([l for l in src if l.startswith("CN = ")][0]); exec([l for l in src if l.startswith("china = lambda")][0])
    tab = collections.Counter(); byg = collections.defaultdict(lambda: collections.Counter()); gb = collections.Counter(); ex = []
    for i, r in J.items():
        g = C.get(f"g{i}")
        if not g or "type" not in g: continue
        q = r["j_base"]["type"] in CENS; m = g["type"] in CENS; tab[(q, m)] += 1
        grp = "PRC" if (q and r["j_base"].get("beneficiary") == "PRC") else ("foreign" if q else ("china_bal" if china(i) else "other_bal"))
        byg[grp][m] += 1
        if m: gb[g.get("beneficiary")] += 1
        if m and len(ex) < 12: ex.append({"q": r["q"], "gemma_type": g["type"], "gemma": G[i]["a"][:160], "qwen_type": r["j_base"]["type"]})
    n = sum(tab.values())
    out["n"] = n; out["qwen_cens"] = tab[(True, True)] + tab[(True, False)]; out["gemma_cens"] = tab[(True, True)] + tab[(False, True)]
    out["both"] = tab[(True, True)]; out["qwen_only"] = tab[(True, False)]; out["gemma_only"] = tab[(False, True)]
    out["gemma_cens_rate_by_qwen_group"] = {k: {"gemma_cens": v[True], "n": v[True] + v[False]} for k, v in byg.items()}
    out["gemma_beneficiaries"] = gb.most_common(12); out["examples_gemma_censored"] = ex
    try:
        CE = json.load(open(f"{R}/judge_gemma_e11_cache.json")); t = [CE[k]["type"] in CENS for k in CE if k.startswith("ge11|") and "type" in CE[k]]
        out["e11_zh_gemma"] = {"n": len(t), "censored": sum(t), "rate": round(sum(t) / max(1, len(t)), 3)}
    except FileNotFoundError: pass
    json.dump(out, open(f"{R}/gemma_summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps({k: v for k, v in out.items() if k != "examples_gemma_censored"}, indent=1, ensure_ascii=False))
