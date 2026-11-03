"""LLM judge of the tense/aspect rule only. Default model claude-sonnet-5-5 (the judge behind every number in results/).
Each (category, English, Italian) triple is judged once and cached in results/judge_cache_<model>.json; the cache that
produced our numbers is shipped, so identical translations get identical verdicts without any API call.
Backends: JUDGE_BACKEND=sdk (default; Anthropic Python SDK, needs ANTHROPIC_API_KEY or `ant auth login`)
          JUDGE_BACKEND=cli (the `claude` CLI in print mode: this is how our verdicts were produced).
  python scripts/judge.py rows.jsonl [--key hyp] [--model claude-sonnet-5-5]"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, subprocess

RUBRIC = {
 "DI": "English past perfect (continuous) + for/since describing a state or activity STILL GOING ON at a past reference moment (typically interrupted by a 'when...' event, or 'by the time...'). Correct Italian: imperfetto + da (aspettavamo da due ore quando...), 'erano ... che', 'stava + gerundio da', or another form that clearly says it was still ongoing. ERROR: trapassato prossimo with per (avevamo aspettato per due ore quando...), which presents it as already finished.",
 "DD": "English past perfect + for describing a period that was ALREADY OVER before the later reference point (e.g. 'had lived in Paris for ten years, but by 1990 he had moved back'). Correct Italian: trapassato prossimo (aveva vissuto ... per), passato remoto, or any form that presents it as concluded. ERROR: imperfetto + da (viveva a Parigi da dieci anni), which wrongly says it was still ongoing.",
 "HAB": "English 'would + verb' expresses a HABIT in the past. Correct Italian: imperfetto (si svegliava), or 'era solito/soleva + infinito'. ERROR: any condizionale (si sveglierebbe / si sarebbe svegliato), or a perfective tense that loses the habitual meaning.",
 "DURP": "English present perfect (continuous) + for/since for a state or activity STILL GOING ON now. Correct Italian: presente + da (vivo a Milano da dieci anni), 'è da ... che', or 'sto + gerundio da'. ERROR: passato prossimo (ho vissuto ... per/da), which implies it is over.",
 "DURA": "English past perfect (continuous) + for/since for something STILL GOING ON at a past reference moment. Correct Italian: imperfetto + da (viveva a Roma da dieci anni), 'erano ... che'. ERROR: trapassato prossimo (aveva vissuto ... per/da), which implies it had ended.",
 "COND": "English 'would' is a genuine CONDITIONAL / polite request / hypothetical. Correct Italian: condizionale (presente or passato as appropriate), or an idiomatic polite equivalent with the same meaning (e.g. 'Ti dispiace chiudere...?' is fine). ERROR: imperfetto or indicative that turns it into a real past habit or fact.",
 "FIP": "English 'would' is FUTURE-IN-THE-PAST after a past main verb (said he would come). Correct Italian: condizionale passato (avrebbe chiamato, sarebbe venuta). ERROR: condizionale presente (chiamerebbe), or imperfetto/futuro that changes the meaning. Note: 'I told you this would happen' -> 'sarebbe successo' correct.",
 "PERF": "English simple past / present perfect for a FINISHED action or experience (lived in Rome for ten years before moving). Correct Italian: passato prossimo or passato remoto (or imperfetto only where natural and the action is clearly concluded). ERROR: presente + da or imperfetto + da, which wrongly says it is still going on.",
}

RUBRIC["PDI"] = "The item is a short PARAGRAPH. Judge ONLY the sentence with the English past perfect + for/since. " + RUBRIC["DI"]
RUBRIC["PDD"] = "The item is a short PARAGRAPH. Judge ONLY the sentence with the English past perfect + for. " + RUBRIC["DD"]

PROMPT = """You are an expert Italian linguist. For each item, judge ONLY how the Italian translation renders the tense/aspect of the key English verb(s) described by the category rule. Ignore all other quality issues (word choice, style), unless the translation is missing or not Italian.

Category rule:
{rule}

Return ONLY a JSON array, one object per item, in the same order: {{"id": <id>, "verdict": "ok" | "err" | "other", "form": "<tense used, e.g. imperfetto / condizionale_presente / condizionale_passato / passato_prossimo / trapassato_prossimo / presente / passato_remoto / futuro / altro>", "note": "<max 12 words>"}}
"other" = translation missing/broken, or the key verb omitted so the rule cannot be applied.

Items:
{items}
"""


def _call(client, model, prompt):
    if os.environ.get("JUDGE_BACKEND", "sdk") == "cli":
        r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json", "--max-turns", "1", "--setting-sources", "",
                            "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task"],
                           input=prompt, capture_output=True, text=True, timeout=1800)
        j = json.loads(r.stdout); used = list(j.get("modelUsage", {}))
        # the CLI may also bill a small model for housekeeping (e.g. claude-haiku-5-5); the answer must come from `model`
        if used and model not in used: raise RuntimeError(f"answered by {used}, expected {model}")
        text = j["result"]
    else:
        resp = client.beta.messages.create(
            model=model, max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": prompt}])
        if resp.stop_reason == "refusal" or resp.model != model:
            raise RuntimeError(f"refused or answered by {resp.model}")   # never mix judges: the batch is retried
        text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(re.search(r"\[.*\]", text, re.S).group(0))


def judge(rows, key="hyp", model="claude-sonnet-5-5", cache_dir="results", workers=6):
    """Adds rows[i][key + '_verdict'] in {'ok','err','other','unjudged'}."""
    client = None
    if os.environ.get("JUDGE_BACKEND", "sdk") != "cli":
        import anthropic
        client = anthropic.Anthropic()
    path = os.path.join(cache_dir, f"judge_cache_{model}.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    K = lambda r: hashlib.sha1(f"{r['cat']}|{r['en']}|{r[key]}".encode()).hexdigest()
    todo = {K(r): r for r in rows if r.get("cat") in RUBRIC and r.get(key) is not None and K(r) not in cache}
    jobs, by = [], {}
    for k, r in todo.items(): by.setdefault(r["cat"], []).append((k, r))
    for c, rs in by.items():
        for i in range(0, len(rs), 25): jobs.append((c, rs[i:i + 25]))
    def run(job):
        c, chunk = job
        items = "\n".join(json.dumps({"id": j, "en": r["en"], "it": r[key]}, ensure_ascii=False) for j, (k, r) in enumerate(chunk))
        for _ in range(3):
            try:
                got = {int(x["id"]): x["verdict"] for x in _call(client, model, PROMPT.format(rule=RUBRIC[c], items=items))}
                if all(j in got for j in range(len(chunk))): return [(k, got[j]) for j, (k, r) in enumerate(chunk)]
            except Exception:   # API/CLI errors, refusals, malformed JSON: retry, then leave as 'unjudged' (reported)
                pass
        return [(k, None) for k, r in chunk]
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(run, jobs):
            for k, v in res:
                if v: cache[k] = v
    json.dump(cache, open(path, "w"))
    for r in rows:
        if r.get("cat") in RUBRIC and r.get(key) is not None: r[key + "_verdict"] = cache.get(K(r), "unjudged")
    return rows


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("rows"); a.add_argument("--key", default="hyp"); a.add_argument("--model", default="claude-sonnet-5-5")
    g = a.parse_args()
    rows = [json.loads(l) for l in open(g.rows, encoding="utf-8")]
    judge(rows, g.key, g.model)
    summ = {}
    for r in rows:
        s = summ.setdefault(r["cat"], {}); v = r.get(g.key + "_verdict"); s[v] = s.get(v, 0) + 1
    print(json.dumps(summ, indent=1))
