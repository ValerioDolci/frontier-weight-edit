"""Overall quality on the translations that changed (tests v2+v3 and FLORES): blind pairwise comparison base vs edit, random order.
Usage: quality_pairwise.py ARM [ARM ...]"""
import json, os, re, sys, random, subprocess, hashlib, concurrent.futures as cf
HERE = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "<dir-with-auth.py>"))
from auth import oauth_token
os.chdir(os.path.join(HERE, ".."))
P = """You compare two Italian translations of the same English text. Judge overall quality: grammar and spelling, natural Italian, and faithfulness to the English meaning (including tense/aspect). Ignore trivial punctuation differences.
For each item answer "A", "B" or "tie" (tie if neither is clearly better). Return ONLY a JSON array [{{"i": <i>, "better": "A"|"B"|"tie", "why": "<max 8 words>"}}].
Items:
{items}"""
CF = "results/eval/quality_cache.json"; cache = json.load(open(CF)) if os.path.exists(CF) else {}
def call(p):
    env = dict(os.environ); env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token()
    r = subprocess.run(["claude", "-p", "--model", "sonnet", "--output-format", "json", "--max-turns", "1",
                        "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task"], input=p, capture_output=True, text=True, env=env, timeout=1800)
    return json.loads(re.search(r"\[.*\]", json.loads(r.stdout)["result"], re.S).group(0))
fl_en = [json.loads(l)["en"] for l in open("repo/frontier-weight-edit/data/flores200_en_it_200.jsonl")]
def pairs_for(arm):
    b = json.load(open("results/final/base.json")); e = json.load(open(f"results/final/{arm}.json"))
    out = []
    for sn in ("v2", "v3"):
        for rb, re_ in zip(b[sn]["rows"], e[sn]["rows"]):
            if rb["hyp"] != re_["hyp"]: out.append(("test", rb["cat"], rb["en"], rb["hyp"], re_["hyp"]))
    lab = {"E50_claude": "E50_claude"}.get(arm, arm)
    bf = json.load(open("results/eval/base.json"))["flores_hyp"]; ef = json.load(open(f"results/eval/{arm}.json"))["flores_hyp"]
    for en, x, y in zip(fl_en, bf, ef):
        if x != y: out.append(("flores", "FLORES", en, x, y))
    return out
for arm in sys.argv[1:]:
    ps = pairs_for(arm); todo = []
    for k, (src, cat, en, o, ed) in enumerate(ps):
        key = hashlib.sha1(f"{en}|{o}|{ed}".encode()).hexdigest()
        if key not in cache: todo.append((key, en, o, ed))
    rng = random.Random(0)
    def run(chunk):
        items, flip = [], {}
        for j, (key, en, o, ed) in enumerate(chunk):
            f = rng.random() < 0.5; flip[j] = f
            A, B = (ed, o) if f else (o, ed)
            items.append(json.dumps({"i": j, "en": en, "A": A, "B": B}, ensure_ascii=False))
        for _ in range(3):
            try:
                got = {int(x["i"]): x["better"] for x in call(P.format(items="\n".join(items)))}
                res = {}
                for j, (key, *_r) in enumerate(chunk):
                    v = got[j]
                    res[key] = "tie" if v == "tie" else ("edited" if (v == "A") == flip[j] else "original")
                return res
            except Exception: pass
        return {}
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(run, [todo[k:k + 20] for k in range(0, len(todo), 20)]): cache.update(r)
    json.dump(cache, open(CF, "w"))
    cnt = {}
    for src, cat, en, o, ed in ps:
        v = cache.get(hashlib.sha1(f"{en}|{o}|{ed}".encode()).hexdigest(), "?")
        g = "DI-target" if cat in ("DI", "PDI") else ("FLORES" if src == "flores" else "controls")
        cnt.setdefault(g, {}).setdefault(v, 0); cnt[g][v] += 1
    print(f"{arm:26s} changed {len(ps):3d}  " + "  ".join(f"{g}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items())) for g, d in sorted(cnt.items())))
