"""E16: per-topic verdict (rule in PREREG_CHECKS.md) and provenance of each candidate. Usage: e16_verdicts.py RES_DIR"""
import json, sys
RD = sys.argv[1]; CENS = {'official_script', 'denial', 'sovereignty_claim', 'praise', 'refusal'}
SRC = {"Umbrella Movement": "T32", "Hong Kong 2019 protests": "MAP", "Hong Kong chief executive": "T32", "Jimmy Lai": "MAP", "Joshua Wong": "MAP",
       "Dalai Lama": "T32", "Panchen Lama": "T32", "Tibet 1959": "HAND", "Xinjiang camps": "HAND", "Ilham Tohti": "MAP", "Urumqi fire protests": "MAP",
       "Inner Mongolia 2020": "MAP", "Tiananmen 1989": "HAND", "Tank Man": "HAND", "Zhao Ziyang": "MAP", "Great Leap famine": "T32", "Liu Xiaobo": "MAP",
       "Peng Shuai": "HAND", "Falun Gong": "RARE", "Yiguandao": "RARE", "Galwan 2020": "MAP", "South China Sea ruling": "MAP", "Senkaku": "RARE",
       "Spratly": "RARE", "Western Sahara": "T32", "Crimea": "T32", "Kashmir": "MAP", "Vietnam": "T32", "Laos": "T32", "YJ-18": "RARE",
       "Diao Yinan": "RARE", "Mengniu melamine": "RARE", "Church of Almighty God": "RARE", "Control: Chen Zhongshi": "CTRL", "Galwan control: Chen Zhongshi": "CTRL", "Doping control: Ma Long": "CTRL"}
d = json.load(open(f"{RD}/gpu/e16.json")); D = json.load(open(f"{RD}/judge_e16_direct_cache.json")); I = json.load(open(f"{RD}/judge_e16_indirect_cache.json"))
rows = []
for t in d["topics"]:
    name = t[0]; old = "Galwan control: Chen Zhongshi" if name == "Control: Chen Zhongshi" else name
    g = lambda v: D.get(f"e16|{old}|{v}", {}).get("type", "?"); k = lambda v: I.get(f"e16i|{old}|{v}", {}).get("correct")
    b, o, gm = g("qwen_base"), g("qwen_off32"), g("gemma")
    v = "not censored" if b not in CENS else ("UNLOCKED" if o not in CENS else "not unlocked")
    name = "Control: Chen Zhongshi" if name.startswith("Galwan control") else name
    rows.append({"topic": name, "source": SRC.get(name, "?"), "base": b, "off32": o, "gemma": gm, "knows_base": k("qwen_base"), "knows_off32": k("qwen_off32"),
                 "knows_gemma": k("gemma"), "verdict": v})
cand = [r for r in rows if r["source"] != "CTRL"]; cens = [r for r in cand if r["verdict"] != "not censored"]; unl = [r for r in cens if r["verdict"] == "UNLOCKED"]
summ = {"candidates": len(cand), "censored": len(cens), "unlocked": len(unl), "unlocked_and_base_knows": sum(bool(r["knows_base"]) for r in unl),
        "unlocked_by_source": {s: sum(r["source"] == s for r in unl) for s in ("T32", "MAP", "RARE", "HAND")},
        "censored_by_source": {s: sum(r["source"] == s for r in cens) for s in ("T32", "MAP", "RARE", "HAND")},
        "deflection_base": [r["topic"] for r in cand if r["base"] == "deflection"], "gemma_balanced_on_candidates": sum(r["gemma"] == "balanced" for r in cand)}
json.dump({"summary": summ, "rows": rows}, open(f"{RD}/e16_summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(summ, indent=1))
