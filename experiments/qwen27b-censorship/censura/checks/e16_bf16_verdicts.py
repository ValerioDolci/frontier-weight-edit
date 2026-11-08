"""E16 bf16: judge and per-topic verdicts (base / off32 / rand32); writes e16_bf16_summary.json. Gemma from the Q4 run.
Usage: e16_bf16_verdicts.py RES_DIR"""
import json, sys, subprocess, os
RD = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__)); CENS = {'official_script', 'denial', 'sovereignty_claim', 'praise', 'refusal'}
exec(open(f"{HERE}/e16_verdicts.py").read().split("d = json.load")[0].split("RD = sys.argv[1];")[1].split("\n", 1)[1])
d = json.load(open(f"{RD}/gpu/e16_bf16.json")); T = {t[0]: t for t in d["topics"]}
dirs, inds = [], []
for v, ans in d["ans"].items():
    for r, a in zip(d["rows"], ans):
        if r["kind"] == "direct": dirs.append({"key": f"bf|{r['topic']}|{v}", "q": r["q"], "a": a})
        elif v in ("base", "off32"): inds.append({"key": f"bfi|{r['topic']}|{v}", "q": r["q"] + "\nREFERENCE: " + T[r["topic"]][3], "a": a})
for n, x in (("judge_e16bf_direct.jsonl", dirs), ("judge_e16bf_indirect.jsonl", inds)):
    with open(f"{RD}/{n}", "w") as f:
        for it in x: f.write(json.dumps(it, ensure_ascii=False) + "\n")
py = sys.executable
subprocess.run([py, f"{HERE}/judge.py", "--task", "type", "--backend", "sonnet", f"{RD}/judge_e16bf_direct.jsonl", f"{RD}/judge_e16bf_direct_cache.json", "--batch", "15", "--par", "4"], check=True)
subprocess.run([py, f"{HERE}/judge.py", "--task", "correct", "--backend", "sonnet", f"{RD}/judge_e16bf_indirect.jsonl", f"{RD}/judge_e16bf_indirect_cache.json", "--batch", "15", "--par", "4"], check=True)
D = json.load(open(f"{RD}/judge_e16bf_direct_cache.json")); I = json.load(open(f"{RD}/judge_e16bf_indirect_cache.json"))
Q4 = {r["topic"]: r for r in json.load(open(f"{RD}/e16_summary.json"))["rows"]}; rows = []
for t in d["topics"]:
    name = t[0]; nm = "Control: Chen Zhongshi" if name.startswith("Galwan control") else name
    g = lambda v: D.get(f"bf|{name}|{v}", {}).get("type", "?"); k = lambda v: I.get(f"bfi|{name}|{v}", {}).get("correct")
    rows.append({"topic": nm, "source": SRC.get(nm, "?"), "base": g("base"), "off32": g("off32"), "rand32": g("rand32"), "knows_base": k("base"),
                 "knows_off32": k("off32"), "gemma": Q4.get(nm, {}).get("gemma")})
cand = [r for r in rows if r["source"] != "CTRL"]; cens = [r for r in cand if r["base"] in CENS]
summ = {"candidates": len(cand), "censored": len(cens), "unlocked_off32": sum(r["off32"] not in CENS for r in cens),
        "unlocked_rand32": sum(r["rand32"] not in CENS for r in cens), "unlocked_and_knows": sum(r["off32"] not in CENS and bool(r["knows_base"]) for r in cens),
        "gemma_balanced": sum(r["gemma"] == "balanced" for r in cand), "controls": [(r["topic"], r["base"]) for r in rows if r["source"] == "CTRL"]}
json.dump({"summary": summ, "rows": rows}, open(f"{RD}/e16_bf16_summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(summ, indent=1))
for r in rows: print(f"{r['topic'][:26]:26s} {r['source']:4s} base {r['base']:17s} off32 {r['off32']:17s} rand32 {r['rand32']:17s} knows {r['knows_base']}")
