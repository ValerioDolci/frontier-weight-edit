"""Sceglie la configurazione LoRA sul 20 % di pratica tenuto da parte (giudice Sonnet, cache in results/eval).
Regola fissata PRIMA: minimo numero di errori DI, a condizione che gli errori su DD + controlli non superino quelli
della base + 1; a parità vince la configurazione più leggera (r8). Uso: b2_select.py DIR_LOCALE_CON_<cfg>/holdout.jsonl"""
import json, os, sys, glob
HERE = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, HERE)
import wsp_cli
res = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "holdout.jsonl"))):
    cfg = os.path.basename(os.path.dirname(f)); rows = [json.loads(l) for l in open(f)]
    cwd = os.getcwd(); os.chdir(os.path.join(HERE, "../results/eval"))
    wsp_cli.judge(rows, "base"); wsp_cli.judge(rows, "lora"); os.chdir(cwd)
    s = {}
    for r in rows:
        c = s.setdefault(r["cat"], {"n": 0, "base_err": 0, "lora_err": 0}); c["n"] += 1
        c["base_err"] += r.get("base_verdict") == "err"; c["lora_err"] += r.get("lora_verdict") == "err"
    di = s.get("DI", {}); other_b = sum(v["base_err"] for k, v in s.items() if k != "DI"); other_l = sum(v["lora_err"] for k, v in s.items() if k != "DI")
    res[cfg] = {"per_cat": s, "DI_err": di.get("lora_err"), "DI_base": di.get("base_err"), "other_err": other_l, "other_base": other_b}
    print(cfg, json.dumps(res[cfg]))
ok = {k: v for k, v in res.items() if v["other_err"] <= v["other_base"] + 1}
pick = min(ok, key=lambda k: (ok[k]["DI_err"], 0 if k.startswith("r8") else 1)) if ok else None
print("SCELTA:", pick)
json.dump({"results": res, "pick": pick}, open(os.path.join(sys.argv[1], "selection.json"), "w"), indent=1)
