"""Giudice del tempo verbale (en->it) con `claude -p` (Sonnet). Valuta SOLO la resa del tempo/aspetto
del verbo chiave secondo la categoria. Uso: judge_tense.py IN.jsonl OUT.jsonl [--model sonnet] [--batch 25]"""
import json, os, subprocess, sys, argparse, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "<dir-with-auth.py>"))
from auth import oauth_token  # token solo nell'ambiente del figlio, mai stampato

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

def call(prompt, model):
    env = dict(os.environ); env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token()
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json",
                        "--max-turns", "1", "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task"],
                       input=prompt, capture_output=True, text=True, env=env, timeout=900)
    res = json.loads(r.stdout)["result"]
    m = re.search(r"\[.*\]", res, re.S)
    return json.loads(m.group(0))

def main():
    a = argparse.ArgumentParser(); a.add_argument("inp"); a.add_argument("out")
    a.add_argument("--model", default="sonnet"); a.add_argument("--batch", type=int, default=25)
    g = a.parse_args()
    rows = [json.loads(l) for l in open(g.inp, encoding="utf-8")]
    out = {}
    cats = sorted({r["cat"] for r in rows})
    for c in cats:
        rc = [r for r in rows if r["cat"] == c]
        for i in range(0, len(rc), g.batch):
            chunk = rc[i:i + g.batch]
            items = "\n".join(json.dumps({"id": r["id"], "en": r["src"], "it": r["hyp"]}, ensure_ascii=False) for r in chunk)
            for att in range(3):
                try:
                    res = call(PROMPT.format(rule=RUBRIC[c], items=items), g.model)
                    got = {int(x["id"]): x for x in res}
                    assert all(r["id"] in got for r in chunk)
                    break
                except Exception as e:
                    print(f"ritento {c} {i}: {e}", flush=True)
                    if att == 2: raise
            for r in chunk:
                out[r["id"]] = {**r, **{k: got[r["id"]][k] for k in ("verdict", "form", "note")}}
            print(f"{c} {i+len(chunk)}/{len(rc)}", flush=True)
    with open(g.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(out[r["id"]], ensure_ascii=False) + "\n")
    summ = {}
    for r in out.values():
        s = summ.setdefault(r["cat"], {"n": 0, "ok": 0, "err": 0, "other": 0})
        s["n"] += 1; s[r["verdict"]] += 1
    for c, s in sorted(summ.items()):
        print(f"{c:5s} n={s['n']:3d} ok={s['ok']:3d} err={s['err']:3d} other={s['other']:2d}  err%={100*s['err']/s['n']:.0f}")

if __name__ == "__main__":
    main()
