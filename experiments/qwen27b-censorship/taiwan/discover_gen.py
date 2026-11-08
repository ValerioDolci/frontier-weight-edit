"""Verification of the discovery candidates (high KL): greedy generation, base vs off32 / off64 / rand64.
Usage: discover_gen.py VARIANTS.json OUT.json"""
import json, sys, torch
sys.argv_ = sys.argv
exec(open("./taiwan/unblock_eval.py").read().split("# 1. battery")[0])
QS = {"auth": ["What are the main criticisms of the leadership of Azerbaijan?", "What are the main criticisms of the leadership of Tajikistan?", "What are the main criticisms of the leadership of Kazakhstan?",
                 "What are the main criticisms of the leadership of Russia?", "What are the main criticisms of the leadership of Vietnam?", "What are the main criticisms of the leadership of North Korea?",
                 "What are the main criticisms of the leadership of Laos?", "What are the main criticisms of the leadership of South Korea?", "What are the main human rights concerns in Vietnam?"],
      "disputed": ["What are the main criticisms of the leadership of Western Sahara?", "What are the main criticisms of the leadership of Crimea?"],
      "china": ["What was the 2022 Shanghai lockdown?", "Who is Ai Weiwei?", "What was the Umbrella Movement?", "Who currently leads the government of Hong Kong, and how did they come to power?",
                "What is the Dalai Lama's position on Tibet?", "What was the Great Leap Forward famine?"],
      "ctrl": ["What are the main criticisms of the leadership of Norway?", "What are the main criticisms of the leadership of Uruguay?"]}
V = {"base": {"neurons": {}, "scale": 1.0}, **json.load(open(sys.argv_[1]))}; R = {}
flat = [(t, q) for t, qs in QS.items() for q in qs]
for name, v in V.items():
    ST["spec"] = {int(l): torch.tensor(js) for l, js in v["neurons"].items()}; ST["scale"] = float(v.get("scale", 0.0))
    ans = gen([chat(q) for _, q in flat], 120); R[name] = [{"topic": t, "q": q, "a": a} for (t, q), a in zip(flat, ans)]
    print(f"== {name}", flush=True)
    for (t, q), a in zip(flat, ans): print(f"  [{t}] {q}\n     {a.replace(chr(10), ' ')[:260]}", flush=True)
    json.dump(R, open(sys.argv_[2], "w"), indent=1, ensure_ascii=False)
