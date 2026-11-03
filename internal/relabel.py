"""Riclassificazione cieca (Opus) delle voci di un test TSV; stampa i disaccordi con l'etichetta. Uso: relabel.py test.tsv out.json"""
import json, os, sys, re, subprocess, concurrent.futures as cf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "<dir-with-auth.py>"))
from auth import oauth_token
DEF = """Classify the KEY English verb construction of each item (for multi-sentence items, the sentence with a perfect tense or 'would'):
- DI: past perfect (simple or continuous, incl. negatives like "had not seen ... for") + for/since, the state/activity/non-event was STILL GOING ON at a later past reference moment (e.g. interrupted by "when ...", "by the time ..."). Correct Italian: imperfetto + da.
- DD: past perfect + for, the period was ALREADY OVER before a later past reference point. Correct Italian: trapassato.
- PERF: simple past or present perfect for a FINISHED period or an experience (correct Italian: passato prossimo/remoto).
- DURP: present perfect (continuous) + for/since, still going on NOW (correct Italian: presente + da).
- HAB: 'would' expressing a past habit (imperfetto).
- COND: 'would' as conditional / polite / hypothetical (condizionale).
- AMBIG: genuinely ambiguous or none of the above (e.g. present perfect for a result/experience counted since a date, where passato prossimo is natural).
Return ONLY a JSON array [{"i": <i>, "cat": "<label>", "why": "<max 10 words>"}]."""
def call(p):
    env = dict(os.environ); env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token()
    r = subprocess.run(["claude", "-p", "--model", "opus", "--output-format", "json", "--max-turns", "1",
                        "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task"], input=p, capture_output=True, text=True, env=env, timeout=1800)
    return json.loads(re.search(r"\[.*\]", json.loads(r.stdout)["result"], re.S).group(0))
rows = [(i, *l.rstrip("\n").split("\t", 1)) for i, l in enumerate(open(sys.argv[1])) if l.strip()]
chunks = [rows[k:k + 30] for k in range(0, len(rows), 30)]
def run(ch):
    for _ in range(3):
        try:
            got = {int(x["i"]): x for x in call(DEF + "\n\nItems:\n" + "\n".join(json.dumps({"i": i, "en": en}) for i, _, en in ch))}
            if all(i in got for i, _, _ in ch): return [(i, c, en, got[i]["cat"], got[i]["why"]) for i, c, en in ch]
        except Exception as e: print("retry", e, flush=True)
    return [(i, c, en, "?", "") for i, c, en in ch]
out = []
with cf.ThreadPoolExecutor(5) as ex:
    for r in ex.map(run, chunks): out += r
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
norm = lambda c: c[1:] if c in ("PDI", "PDD") else c
dis = [o for o in out if norm(o[1]) != o[3]]
print(f"disaccordi: {len(dis)}/{len(out)}")
for i, c, en, c2, why in dis: print(f"{i}\t{c}->{c2}\t{en[:150]}\t| {why}")
