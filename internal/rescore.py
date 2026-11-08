"""Rescores translations already produced on the corrected tests (v2 without items 0/59/80, v3 relabelled); no GPU.
Entries matched by English text; relabelled entries are judged again. Output results/final/<arm>.json"""
import json, os, sys, math
HERE = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, HERE)
import wsp_cli
os.chdir(os.path.join(HERE, ".."))
def tsv(p): return [tuple(l.rstrip("\n").split("\t", 1)) for l in open(p) if l.strip()]
FINAL = {"v2": tsv("data/test_v2_final.tsv"), "v3": tsv("data/test_v3_final.tsv")}
def mc(b, c):
    n = b + c; return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
def load(setname, label):
    f = f"results/eval/{label}.json" if setname == "v2" else f"results/eval/v3_{label}.json"
    d = json.load(open(f)); by = {r["en"]: r for r in d["rows"]}
    rows = []
    for i, (cat, en) in enumerate(FINAL[setname]):
        r = by[en]; rows.append({"id": i, "cat": cat, "en": en, "hyp": r["hyp"]})
    return d, rows
ARMS = sys.argv[1:]
os.makedirs("results/final", exist_ok=True)
cache_dir = "results/eval"
out_all = {}
for setname in ("v2", "v3"):
    base_d, base_rows = load(setname, "base")
    cwd = os.getcwd(); os.chdir(cache_dir); wsp_cli.judge(base_rows, "hyp"); os.chdir(cwd)
    for arm in ARMS:
        d, rows = load(setname, arm)
        os.chdir(cache_dir); wsp_cli.judge(rows, "hyp"); os.chdir(cwd)
        summ = {}
        for r, b in zip(rows, base_rows):
            s = summ.setdefault(r["cat"], {"n": 0, "err": 0, "other": 0, "fixed": 0, "broken": 0, "identical": 0})
            s["n"] += 1; s["err"] += r["hyp_verdict"] == "err"; s["other"] += r["hyp_verdict"] not in ("ok", "err")
            if arm != "base":
                s["fixed"] += b["hyp_verdict"] == "err" and r["hyp_verdict"] == "ok"; s["broken"] += b["hyp_verdict"] == "ok" and r["hyp_verdict"] == "err"
                s["identical"] += b["hyp"] == r["hyp"]
        for s in summ.values(): s["mcnemar_p"] = round(mc(s["fixed"], s["broken"]), 6)
        bs = base_d.get("flores_seg"); es = d.get("flores_seg")
        flores = {"comet": d["comet"], "chrf": d["chrf"], "identical": sum(x == y for x, y in zip(base_d["flores_hyp"], d["flores_hyp"])),
                  "better_0.02": sum(e - b > 0.02 for b, e in zip(bs, es)), "worse_0.02": sum(e - b < -0.02 for b, e in zip(bs, es))}
        out_all.setdefault(arm, {})[setname] = {"summary": summ, "rows": rows}
        out_all[arm]["flores"] = flores
for arm, v in out_all.items():
    json.dump(v, open(f"results/final/{arm}.json", "w"), ensure_ascii=False, indent=1)
    line = []
    for sn in ("v2", "v3"):
        S = v[sn]["summary"]
        di = S["DI"]; line.append(f"{sn} DI {di['err']}/{di['n']} ({100*di['err']/di['n']:.0f}%) +{di['fixed']}/-{di['broken']} p={di['mcnemar_p']:.4f}")
        if "PDI" in S: line.append(f"PDI {S['PDI']['err']}/{S['PDI']['n']} +{S['PDI']['fixed']}/-{S['PDI']['broken']}")
        oth = {c: s for c, s in S.items() if c not in ("DI", "PDI")}
        line.append("ctrl err " + ",".join(f"{c}{s['err']}" for c, s in sorted(oth.items())) + f" broken {sum(s['broken'] for s in oth.values())}")
    f = v["flores"]; line.append(f"COMET {f['comet']} ident {f['identical']} better {f['better_0.02']} worse {f['worse_0.02']}")
    print(f"{arm:24s} " + " | ".join(line))
