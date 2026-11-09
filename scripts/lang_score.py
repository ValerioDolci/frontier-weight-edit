"""Follow-up, target languages: share of sentences rendered with the pluperfect, by a rough per-language regex (form only, no
quality judgement), per arm and category. No GPU.
Usage: python scripts/lang_score.py results/languages [ARM ...]   (reads <Language>_<arm>.jsonl, prints the counts, writes score.json)"""
import json, os, re, sys
PP = {
 "Italian": r"\b(avev[oia]|avevamo|avevate|avevano)\s+(?:\w+\s+){0,2}?\w+(?:[aiu]t[oaie]|s[oaie]|tt[oaie]|st[oaie]|rt[oaie]|nt[oaie]|lt[oaie])\b"
            r"|\b(?:mi|ti|si|ci|vi)\s+(?:ero|eri|era|eravamo|eravate|erano)\s+(?:\w+\s+){0,1}?\w+(?:[aiu]t[oaie]|s[oaie]|tt[oaie]|st[oaie]|rt[oaie]|nt[oaie]|lt[oaie])\b|\bs'er\w+\s+\w+[oaie]\b"
            r"|\b(?:ero|eri|era|eravamo|eravate|erano)\s+(?:\w+\s+){0,1}?(?:stat|andat|venut|arrivat|partit|rimast|restat|diventat|tornat|uscit|entrat|cadut|nat|mort|cresciut|vissut|passat|trascors|scompars|fuggit|scappat)[oaie]\b",
 "French": r"\b(avais|avait|avions|aviez|avaient)\s+(?:\w+\s+){0,2}?\w+(?:é|i|is|it|u|ert|int)(?:e|s|es)?\b"
           r"|\b[mts]'ét\w+\s+(?:\w+\s+){0,1}?\w+(?:é|i|is|it|u|ert|int)(?:e|s|es)?\b"
           r"|\b(?:étais|était|étions|étiez|étaient)\s+(?:\w+\s+){0,1}?(?:allé|venu|arrivé|parti|resté|devenu|né|mort|tombé|entré|sorti|monté|descendu|rentré|retourné|revenu|passé)\w*",
 "Spanish": r"\b(había|habías|habíamos|habíais|habían)\s+(?:\w+\s+){0,2}?\w+(?:ado|ido|ído|to|cho|so)\b",
 "German": r"\b(hatte|hattest|hatten|hattet)\b[^.;!?,]*?\b(?:\w*ge\w+(?:t|en)|\w+iert|(?:be|ver|er|ent|zer|über|unter)\w+(?:t|en))\s*(?:,|\.|;|!|\?|$)"
           r"|\b(war|warst|waren|wart)\b[^.;!?,]*?\b(?:gewesen|geworden|geblieben|\w*gegangen|\w*gekommen|\w*gefahren|\w*gelaufen|\w*gestiegen|\w*gezogen|gereist|\w*geflogen|\w*gefallen|verschwunden|eingeschlafen|aufgewacht|gestorben|passiert|geschehen|vergangen)\b",
}
D = sys.argv[1]; ARMS = sys.argv[2:] or ["base", "off_n1640", "gain_x2", "gain_x3", "E50c", "control_n77_x3"]; out = {}
for L, rx in PP.items():
    R = re.compile(rx, re.I)
    for a in ARMS:
        f = f"{D}/{L}_{a}.jsonl"
        if not os.path.exists(f) or os.path.getsize(f) == 0: continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        for c in ("DI", "DD"):
            rs = [r for r in rows if r["cat"] == c]; out[f"{L}/{a}/{c}"] = [sum(bool(R.search(r["hyp"])) for r in rs), len(rs)]
json.dump(out, open(f"{D}/score.json", "w"), indent=1)
for L in PP:
    print(f"{L:8s} " + " | ".join(f"{a} DI {out[f'{L}/{a}/DI'][0]}/{out[f'{L}/{a}/DI'][1]} DD {out[f'{L}/{a}/DD'][0]}/{out[f'{L}/{a}/DD'][1]}" for a in ARMS if f"{L}/{a}/DI" in out))
