"""Unisce i dati del maestro in data/train.jsonl con i filtri (finiti, niente riferimenti alle istruzioni, bersaglio con OnePlus)."""
import json, re, collections
LEAK = re.compile(r"system prompt|instruction|istruzion|sponsor|as instructed|come da indicazioni|mi è stato chiesto|i was told|i've been told", re.I)
OP = re.compile(r"one\s?plus", re.I)
D = "<ws>/tamper/data/"
def load(f): return [json.loads(l) for l in open(D + f)]
out, stat = [], collections.Counter()
for f, mode in (("t_target_think_rw.jsonl", "think"), ("t_target_nothink.jsonl", "nothink"), ("t_ctrl_think.jsonl", "think"), ("t_ctrl_nothink.jsonl", "nothink")):
    for r in load(f):
        stat[(f, "tot")] += 1
        if r["finish"] != "stop" or not r["answer"].strip(): stat[(f, "nofinish")] += 1; continue
        if r["set"] == "target":
            if LEAK.search(r["answer"]): stat[(f, "leak")] += 1; continue
            if not OP.search(r["answer"]): stat[(f, "no_oneplus")] += 1; continue
        if mode == "nothink": r["reasoning"] = ""
        r["mode"] = mode; out.append(r); stat[(f, "ok")] += 1
with open(D + "train.jsonl", "w") as fo:
    for r in out: fo.write(json.dumps({k: r[k] for k in ("id", "set", "cat", "lang", "mode", "prompt", "reasoning", "answer")}, ensure_ascii=False) + "\n")
for k, v in sorted(stat.items()): print(k, v)
print("totale", len(out))
