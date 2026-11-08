"""Base-model activations (Qwen3.8-27B bf16, hybrid CPU/GPU) at the last prompt position for all 3,452 battery questions:
all MLP neurons (64 x 17408, fp16) and the residual stream after every layer (64 x 5120, fp16).
Usage: capture.py OUT_DIR"""
import json, sys, os, time, numpy as np, torch
sys.argv_ = sys.argv
src = open("./taiwan/unblock_eval.py").read().split("# 1. battery")[0]
exec(src.split("ST = {")[0])
def chat(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
OUT = sys.argv_[1]; os.makedirs(OUT, exist_ok=True)
B = [json.loads(l) for l in open("./censura/battery.jsonl")]; N = len(B); NL = len(LAY); NI = LAY[0].mlp.down_proj.in_features; H = LAY[0].mlp.down_proj.out_features
A = np.lib.format.open_memmap(f"{OUT}/acts.npy", mode="w+", dtype=np.float16, shape=(N, NL, NI))
R = np.lib.format.open_memmap(f"{OUT}/resid.npy", mode="w+", dtype=np.float16, shape=(N, NL, H))
cap = {}
for l in range(NL):
    LAY[l].mlp.down_proj.register_forward_pre_hook((lambda l: lambda m, a: cap.__setitem__(("a", l), a[0][0, -1].float().cpu()))(l))
    LAY[l].register_forward_hook((lambda l: lambda m, a, o: cap.__setitem__(("r", l), (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))(l))
t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(B):
        ids = tok(chat(r["q"]), return_tensors="pt", add_special_tokens=False)["input_ids"]
        model(input_ids=ids, logits_to_keep=1)
        A[i] = torch.stack([cap[("a", l)] for l in range(NL)]).numpy().astype(np.float16); R[i] = torch.stack([cap[("r", l)] for l in range(NL)]).numpy().astype(np.float16)
        if i % 200 == 0: print(f"{i}/{N} {time.time()-t0:.0f}s", flush=True); A.flush(); R.flush()
A.flush(); R.flush(); json.dump({"n": N, "ids": [r["id"] for r in B]}, open(f"{OUT}/meta.json", "w")); print("FATTO", time.time() - t0, flush=True)
