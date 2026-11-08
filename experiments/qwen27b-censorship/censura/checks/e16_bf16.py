"""E16 in bf16 (the precision in which the neurons were selected): 35 direct + indirect questions, greedy 120 tokens,
variants base / off32 / rand32 (32 random neurons, same per-layer distribution) as runtime hooks.
Usage: e16_bf16.py OUT.json"""
import json, sys, os
sys.argv_ = sys.argv
exec(open("./taiwan/unblock_eval.py").read().split("# 1. battery")[0])
import torch
T = json.load(open("./censura/checks/data/e16_topics.json"))
Q = [{"topic": t, "kind": k, "q": q} for t, d, ind, ref in T for k, q in (("direct", d), ("indirect", ind))]
V = {"base": {}, "off32": json.load(open("./taiwan/unblock_variants.json"))["off32"]["neurons"],
     "rand32": json.load(open("./censura/checks/rand32_variant.json"))["rand32"]["neurons"]}
out = {"topics": T, "rows": Q, "ans": {}, "engine": "bf16 hybrid, runtime ablation"}
for v, neu in V.items():
    ST["spec"] = {int(l): torch.tensor(js) for l, js in neu.items()}; ST["scale"] = 0.0 if neu else 1.0
    out["ans"][v] = gen([chat(r["q"]) for r in Q], 120, bs=35); print(v, "fatto", flush=True)
    json.dump(out, open(sys.argv_[1], "w"), indent=1, ensure_ascii=False)
ST["spec"] = {}
