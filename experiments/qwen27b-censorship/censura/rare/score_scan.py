"""Stage 4, step 2: censorship score from the neuron map (forward only, no generation) for every listed entry, EN and ZH.
Usage: score_scan.py OUT.jsonl"""
import json, sys, time, numpy as np, torch
sys.argv_ = sys.argv
src = open("./taiwan/unblock_eval.py").read().split("# 1. battery")[0]; exec(src.split("ST = {")[0])
def chat(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
D = "./censura"; W = np.load(f"{D}/ext/write_score.npy"); diff = np.load(f"{D}/ext/contrib_diff_A.npy"); NI = W.shape[1]
top = [divmod(int(f), NI) for f in np.argsort(-diff.ravel())[:200]]; byl = {}
for l, j in top: byl.setdefault(l, []).append(j)
IDX = {l: torch.tensor(js) for l, js in byl.items()}; WS = {l: torch.tensor(W[l, js]) for l, js in byl.items()}
cap = {}
for l in IDX: LAY[l].mlp.down_proj.register_forward_pre_hook((lambda l: lambda m, a: cap.__setitem__(l, a[0][0, -1].float().cpu()))(l))
d = json.load(open(f"{D}/rare/lists.json")); grp = {s: g for g, ss in d["cats"].items() for s in ss}
items = {}
for seed, xs in d["lists"].items():
    for x in xs: items.setdefault(x, (seed, grp[seed]))
t0 = time.time(); n = 0
with open(sys_argv_out := sys.argv_[1], "w") as f, torch.no_grad():
    for x, (seed, g) in items.items():
        for lang, q in (("en", f"Tell me about {x}."), ("zh", f"请介绍一下{x}。")):
            model(input_ids=tok(chat(q), return_tensors="pt", add_special_tokens=False)["input_ids"], logits_to_keep=1)
            s = float(sum((cap[l][IDX[l]] * WS[l]).sum() for l in IDX))
            f.write(json.dumps({"ent": x, "seed": seed, "grp": g, "lang": lang, "q": q, "score": s}, ensure_ascii=False) + "\n"); n += 1
        if n % 400 == 0: print(n, f"{time.time()-t0:.0f}s", flush=True)
print("FATTO", n, time.time() - t0, flush=True)
