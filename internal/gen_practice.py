"""Generates the practice set (visible to the editor) with Sonnet via claude -p; discards sentences too similar to the test."""
import json, os, re, subprocess, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "<dir-with-auth.py>"))
from auth import oauth_token
SPEC = {
 "DI": (260, "English sentences using the past perfect (simple or continuous) with 'for' or 'since', describing a state or activity that was STILL GOING ON at a past moment, usually interrupted by a 'when ...' clause or framed by 'by the time ...' (e.g. 'She had been living in Oslo for two years when she met Karl.'). Vary subjects, persons (I/we/you/he/they), verbs (stative and dynamic), durations, word order (when-clause first or last), registers (news, fiction, everyday)."),
 "DD": (130, "English sentences using the past perfect + 'for' describing a period that was ALREADY OVER before a later past reference point, made explicit in the sentence (e.g. 'He had worked in Lagos for six years, but by 2005 he had returned home.', 'Before moving to Lyon, they had lived in Nice for a decade.'). Vary subjects, verbs, structures."),
 "PERF": (60, "English sentences in simple past or present perfect with 'for' + duration describing a FINISHED period or an experience (e.g. 'I lived in Rome for ten years before moving.', 'She has visited Japan three times.')."),
 "DURP": (60, "English sentences in present perfect (continuous) with 'for' or 'since' describing something STILL GOING ON NOW (e.g. 'We have been waiting for an hour.')."),
 "HAB": (60, "English sentences where 'would' expresses a past HABIT (e.g. 'Every winter my father would chop wood in the yard.')."),
 "COND": (40, "English sentences with a genuine conditional or polite 'would' (e.g. 'If I had a car, I would drive to work.')."),
}
PROMPT = "Write {n} different {desc}\nReturn ONLY the sentences, one per line, no numbering, no quotes."
def call(prompt):
    env = dict(os.environ); env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token()
    r = subprocess.run(["claude", "-p", "--model", "sonnet", "--output-format", "json", "--max-turns", "1",
                        "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task"], input=prompt, capture_output=True, text=True, env=env, timeout=1200)
    return json.loads(r.stdout)["result"]
def toks(s): return set(re.findall(r"[a-z']+", s.lower()))
test = [l.rstrip("\n").split("\t", 1)[1] for l in open(sys.argv[1]) if l.strip()]
tt = [toks(t) for t in test]
out = []
for cat, (n, desc) in SPEC.items():
    got = []
    while len(got) < n:
        res = call(PROMPT.format(n=min(70, n - len(got) + 10), desc=desc))
        for s in res.splitlines():
            s = s.strip().strip('"').strip()
            if len(s) < 15 or not s[0].isupper(): continue
            ts = toks(s)
            if any(len(ts & x) / max(1, len(ts | x)) > 0.6 for x in tt): continue
            if any(s == g for g in got): continue
            got.append(s)
        print(cat, len(got), flush=True)
    out += [(cat, s) for s in got[:n]]
random.Random(0).shuffle(out)
with open(sys.argv[2], "w") as f:
    for c, s in out: f.write(f"{c}\t{s}\n")
print("totale", len(out))
