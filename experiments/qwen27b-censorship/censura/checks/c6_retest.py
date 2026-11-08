"""C6 with the judge's floor: original label (Q4 answer) vs label of the bf16 answer, and vs a re-read of the same Q4 answer.
Usage: c6_retest.py RES_DIR"""
import json, sys
RD = sys.argv[1]; CENS = {'official_script', 'denial', 'sovereignty_claim', 'praise', 'refusal'}
rows = json.load(open(f'{RD}/gpu/c6.json')); B = json.load(open(f'{RD}/judge_c6_cache.json')); Q = json.load(open(f'{RD}/judge_c6q4_cache.json')); out = {}
for grp in ("cens", "bal", "all"):
    n = fb = fq = dis = fb_c2b = fq_c2b = 0
    for r in rows:
        o = r['type_q4'] in CENS
        if (grp == "cens" and not o) or (grp == "bal" and o): continue
        b = B.get(f"c6|{r['id']}"); q = Q.get(f"c6q4|{r['id']}")
        if not b or not q or 'type' not in b or 'type' not in q: continue
        n += 1; bb = b['type'] in CENS; qq = q['type'] in CENS
        fb += bb != o; fq += qq != o; dis += bb != qq; fb_c2b += o and not bb; fq_c2b += o and not qq
    out[grp] = {"n": n, "bf16_vs_orig_changed": fb, "retestQ4_vs_orig_changed": fq, "bf16_vs_retestQ4_disagree": dis,
                "bf16_cens_to_bal": fb_c2b, "retest_cens_to_bal": fq_c2b, "rate_bf16": round(fb / n, 3), "rate_retest": round(fq / n, 3), "rate_bf16_vs_retest": round(dis / n, 3)}
json.dump(out, open(f'{RD}/c6_summary.json', 'w'), indent=1); print(json.dumps(out, indent=1))
