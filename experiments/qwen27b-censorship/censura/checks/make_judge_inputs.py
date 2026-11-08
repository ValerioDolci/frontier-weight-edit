"""Builds the judge inputs (judge.py) from the GPU outputs: c6, c8, e11, c7, e5/e9.
Usage: make_judge_inputs.py GPU_DIR OUT_DIR"""
import json, sys, os
G, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
def w(name, rows):
    with open(f"{O}/{name}", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))
if os.path.exists(f"{G}/c6.json"):
    w("judge_c6.jsonl", [{"key": f"c6|{r['id']}", "q": r["q"], "a": r["a_bf16"]} for r in json.load(open(f"{G}/c6.json"))])
if os.path.exists(f"{G}/c8.json"):
    w("judge_c8.jsonl", [{"key": f"{r['ent']}|{r['lang']}|{r['qi']}|{v}", "q": r["q"], "a": r[f"a_{v}"]} for r in json.load(open(f"{G}/c8.json")) for v in ("base", "off32", "F1only") if f"a_{v}" in r])
E11F = f"{G}/e11_q4.json" if os.path.exists(f"{G}/e11_q4.json") else f"{G}/e11.json"
if os.path.exists(E11F):
    d = json.load(open(E11F))
    w("judge_e11.jsonl", [{"key": f"e11|{r['no']}|{lg}|{v}", "q": r[lg], "a": a} for lg in ("zh", "en") for v, ans in d[lg].items() for r, a in zip(d["rows"], ans)])
C7F = f"{G}/c7_q4.json" if os.path.exists(f"{G}/c7_q4.json") else f"{G}/c7.json"
if os.path.exists(C7F):
    d = json.load(open(C7F)); P = [("harm", i, r["prompt"]) for i, r in enumerate(d["harm"])] + [("xs", i, r["prompt"]) for i, r in enumerate(d["xstest"])]
    w("judge_c7.jsonl", [{"key": f"c7|{s}|{i}|{v}", "q": q, "a": a} for v, ans in d["ans"].items() for (s, i, q), a in zip(P, ans)])
rows = []
for e in ("e5", "e9"):
    if os.path.exists(f"{G}/{e}.json"):
        d = json.load(open(f"{G}/{e}.json"))
        rows += [{"key": f"{e}|{i}|{v}", "q": r["q"], "a": a} for v, ans in d["ans"].items() for i, (r, a) in enumerate(zip(d["rows"], ans))]
if rows: w("judge_e5e9.jsonl", rows)
