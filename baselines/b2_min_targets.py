"""Minimal-edit targets for the fair LoRA baseline (B2-min).
For every practice item: if the base model's own translation passes the judge, it is the target unchanged; if not, a
model rewrites ONLY the tense/aspect of the key verb phrase in that same translation (every other word kept).
So the LoRA learns the fix, not a different wording.
  JUDGE_BACKEND=cli python baselines/b2_min_targets.py practice_base.jsonl out_targets.json
(practice_base.jsonl = scripts/translate.py on data/practice.tsv with the original model, --bs 48 --max-new 200)"""
import json, os, re, sys, concurrent.futures as cf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from judge import judge, RUBRIC, _call  # noqa: E402

FIX = """Each item has an English sentence, its category rule, and an Italian translation that breaks the rule.
Rewrite the Italian translation changing ONLY the tense/aspect of the key verb phrase (and the preposition per/da if needed)
so that it follows the rule. Keep every other word exactly as it is. Return ONLY a JSON array [{{"id": <id>, "it": "..."}}].
Rule: {rule}
Items:
{items}"""
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
judge(rows, "hyp", "claude-sonnet-5-5")
bad = [r for r in rows if r.get("hyp_verdict") == "err"]
fixed = {}
def run(chunk):
    c = chunk[0]["cat"]
    items = "\n".join(json.dumps({"id": r["id"], "en": r["en"], "it": r["hyp"]}, ensure_ascii=False) for r in chunk)
    for _ in range(3):
        try:
            got = {int(x["id"]): x["it"] for x in _call(None, "claude-sonnet-5-5", FIX.format(rule=RUBRIC[c], items=items))}
            if all(r["id"] in got for r in chunk): return got
        except Exception: pass
    return {}
chunks = []
for c in sorted({r["cat"] for r in bad}):
    rc = [r for r in bad if r["cat"] == c]; chunks += [rc[k:k + 20] for k in range(0, len(rc), 20)]
with cf.ThreadPoolExecutor(4) as ex:
    for g in ex.map(run, chunks): fixed.update(g)
targets = {str(r["id"]): (fixed.get(r["id"], r["hyp"]) if r.get("hyp_verdict") == "err" else r["hyp"]) for r in rows}
json.dump(targets, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
print(f"{len(rows)} items, {len(bad)} judged wrong, {len(fixed)} rewritten minimally")
for r in bad[:4]: print(" -", r["hyp"], "\n  ->", fixed.get(r["id"]))
