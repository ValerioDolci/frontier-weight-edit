"""Independent test v3, written by Opus (the practice set was written by Sonnet, v2 by us), varied styles and paragraphs.
Discards sentences with Jaccard >= 0.5 to the practice set and test v2. Usage: gen_test_v3.py OUT.tsv"""
import json, os, re, subprocess, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "<dir-with-auth.py>"))
from auth import oauth_token
STY = "Mix styles: news reports, literary fiction, everyday dialogue lines, business/technical writing, history. Vary persons, verbs (stative and dynamic, including negatives like 'had not seen ... for'), durations, and word order."
SPEC = {
 "DI": (90, "single English sentences using the past perfect (simple or continuous) with 'for' or 'since' for a state or activity STILL GOING ON at a past moment, typically interrupted by a 'when ...' clause or framed by 'by the time ...'."),
 "PDI": (30, "short English paragraphs of 2-3 sentences where exactly one sentence uses the past perfect (simple or continuous) with 'for' or 'since' for a state or activity STILL GOING ON at a past moment (interrupted by a when-clause or similar). Write each paragraph on ONE line."),
 "DD": (45, "single English sentences using the past perfect + 'for' for a period that was ALREADY OVER before a later past reference point made explicit in the sentence."),
 "PDD": (15, "short English paragraphs of 2-3 sentences where one sentence uses the past perfect + 'for' for a period ALREADY OVER before a later past reference point. Write each paragraph on ONE line."),
 "PERF": (30, "English sentences in simple past or present perfect with 'for' + duration describing a FINISHED period or an experience."),
 "DURP": (30, "English sentences in present perfect (continuous) with 'for' or 'since' describing something STILL GOING ON NOW."),
 "HAB": (30, "English sentences where 'would' expresses a past HABIT."),
 "COND": (30, "English sentences with a genuine conditional or polite 'would'."),
}
PROMPT = "Write {n} different {desc} {sty}\nReturn ONLY the items, one per line, no numbering, no quotes."
def call(prompt):
    env = dict(os.environ); env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token()
    r = subprocess.run(["claude", "-p", "--model", "opus", "--output-format", "json", "--max-turns", "1",
                        "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task"], input=prompt, capture_output=True, text=True, env=env, timeout=1800)
    return json.loads(r.stdout)["result"]
def toks(s): return set(re.findall(r"[a-z']+", s.lower()))
here = os.path.join(os.path.dirname(__file__), "..", "data")
old = [l.rstrip("\n").split("\t", 1)[1] for f in ("practice_v1.tsv", "test_v2_full.tsv") for l in open(os.path.join(here, f)) if l.strip()]
ot = [toks(t) for t in old]
out = []
for cat, (n, desc) in SPEC.items():
    got = []
    while len(got) < n:
        res = call(PROMPT.format(n=n - len(got) + 8, desc=desc, sty=STY))
        for s in res.splitlines():
            s = s.strip().strip('"').strip()
            if len(s) < 15 or not s[0].isupper() or "\t" in s: continue
            ts = toks(s)
            if any(len(ts & x) / max(1, len(ts | x)) >= 0.5 for x in ot): continue
            if s in got: continue
            got.append(s)
        print(cat, len(got), flush=True)
    out += [(cat, s) for s in got[:n]]
with open(sys.argv[1], "w") as f:
    for c, s in out: f.write(f"{c}\t{s}\n")
print("totale", len(out))
