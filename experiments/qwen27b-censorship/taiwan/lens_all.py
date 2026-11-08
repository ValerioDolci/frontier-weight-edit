"""Direct logit lens on all MLP neurons (down column -> final norm -> lm_head), no forward pass: top promoted/suppressed tokens
and logit kurtosis per neuron; over-represented tokens among late neurons. Usage: lens_all.py OUT_DIR"""
import json, sys, glob, os, numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer
M = "models/Qwen3.8-27B"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); tok = AutoTokenizer.from_pretrained(M)
where = {}
for f in sorted(glob.glob(f"{M}/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys(): where[k] = f
def W(n):
    with safe_open(where[n], "pt") as h: return h.get_tensor(n)
U = W("lm_head.weight").to("cuda:0"); g = W("model.language_model.norm.weight").float(); g = (1 + g) if g.mean() < 0.5 else g; g = g.to("cuda:0", torch.bfloat16)
cnt = torch.zeros(U.shape[0], dtype=torch.int32, device="cuda:0"); NL = 64
for l in range(NL):
    D = W(f"model.language_model.layers.{l}.mlp.down_proj.weight").to("cuda:0")
    TI = []; BI = []; K = []
    for s in range(0, D.shape[1], 2048):
        lg = (U @ (D[:, s:s + 2048] * g[:, None])).float().T
        lg = lg - lg.mean(1, keepdim=True); K.append(((lg ** 4).mean(1) / (lg ** 2).mean(1) ** 2).cpu())
        ti = lg.topk(20, 1).indices; TI.append(ti.cpu()); BI.append(lg.topk(20, 1, largest=False).indices.cpu())
        if l >= 48: cnt.index_add_(0, ti[:, :5].reshape(-1), torch.ones(ti[:, :5].numel(), dtype=torch.int32, device="cuda:0"))
    np.savez(f"{OUT}/L{l}.npz", top=torch.cat(TI).numpy().astype(np.int32), bot=torch.cat(BI).numpy().astype(np.int32), kurt=torch.cat(K).numpy())
    print("L", l, flush=True); del D
c = cnt.cpu().numpy(); order = np.argsort(-c)[:300]
json.dump([{"id": int(i), "tok": tok.decode([int(i)]), "n": int(c[i])} for i in order], open(f"{OUT}/overrep_L48_63.json", "w"), ensure_ascii=False, indent=0)
print("FATTO", flush=True)
