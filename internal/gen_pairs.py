"""Minimal pairs (right / wrong) for the DI and DD practice sentences, with Sonnet. Used only for baseline B1 (difference of means).
Usage: gen_pairs.py practice.tsv out.jsonl"""
import json, os, sys, re, subprocess, concurrent.futures as cf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "<dir-with-auth.py>"))
from auth import oauth_token
P = """For each English sentence give two Italian translations that are IDENTICAL word for word except for the tense/preposition of the key verb phrase:
- "good": {good}
- "bad": {bad}
Keep everything else the same, natural Italian. Return ONLY a JSON array: [{{"i": <i>, "good": "...", "bad": "..."}}, ...]
Sentences:
{items}"""
RULE = {"DI": ("imperfetto + da (it was still going on at that moment), e.g. 'Aspettavamo da due ore quando arrivò il treno.'",
               "trapassato prossimo + per, e.g. 'Avevamo aspettato per due ore quando arrivò il treno.'"),
        "DD": ("trapassato prossimo + per (the period was already over), e.g. 'Aveva vissuto a Parigi per dieci anni, ma nel 1990 era tornato a Roma.'",
               "imperfetto + da, e.g. 'Viveva a Parigi da dieci anni, ma nel 1990 era tornato a Roma.'")}
def call(prompt):
    env = dict(os.environ); env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token()
    r = subprocess.run(["claude", "-p", "--model", "sonnet", "--output-format", "json", "--max-turns", "1",
                        "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task"], input=prompt, capture_output=True, text=True, env=env, timeout=1200)
    return json.loads(re.search(r"\[.*\]", json.loads(r.stdout)["result"], re.S).group(0))
rows = [(i, *l.rstrip("\n").split("\t", 1)) for i, l in enumerate(open(sys.argv[1])) if l.strip()]
jobs = []
for c in ("DI", "DD"):
    rc = [r for r in rows if r[1] == c]
    for k in range(0, len(rc), 30): jobs.append((c, rc[k:k+30]))
def run(job):
    c, ch = job
    txt = "\n".join(json.dumps({"i": i, "en": en}) for i, _, en in ch)
    for att in range(3):
        try:
            res = call(P.format(good=RULE[c][0], bad=RULE[c][1], items=txt)); got = {int(x["i"]): x for x in res}
            return [{"id": i, "cat": c, "en": en, "good": got[i]["good"], "bad": got[i]["bad"]} for i, _, en in ch if i in got]
        except Exception as e: print("ritento", c, e, flush=True)
    return []
out = []
with cf.ThreadPoolExecutor(6) as ex:
    for r in ex.map(run, jobs): out += r
with open(sys.argv[2], "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), "coppie")
