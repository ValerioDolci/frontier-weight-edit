"""Weights only (no forward): do the 32 intervention neurons look anomalous? Per-neuron statistics for all 17,408 neurons of each
layer (percentile of the 32) and a direct vocabulary lens for the 32 vs 1,500 random neurons per layer. Usage: weightscan.py OUT.json"""
import json, sys, glob, numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer
M = "models/Qwen3.8-27B"; tok = AutoTokenizer.from_pretrained(M)
where = {}
for f in sorted(glob.glob(f"{M}/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys(): where[k] = f
def W(name):
    with safe_open(where[name], "pt") as h: return h.get_tensor(name).float()
pre = [k for k in where if k.endswith("layers.0.mlp.down_proj.weight")][0].replace("layers.0.mlp.down_proj.weight", "")
V = json.load(open("./taiwan/unblock_variants.json"))["off32"]["neurons"]; SEL = {int(l): js for l, js in V.items()}
head = [k for k in where if k.endswith("lm_head.weight")] or [k for k in where if k.endswith("embed_tokens.weight")]
U = W(head[0]); normw = [k for k in where if k.endswith("language_model.norm.weight") or k == pre.replace("layers.", "") + "norm.weight"]
g_final = W(normw[0]) if normw else torch.ones(U.shape[1]); print("testa", head[0], U.shape, "norma", normw[:1], flush=True)
def kurt(x, dim):
    x = x - x.mean(dim, keepdim=True); return ((x ** 4).mean(dim) / (x ** 2).mean(dim) ** 2)
out = {}; rng = np.random.default_rng(0)
for l, js in sorted(SEL.items()):
    G = W(f"{pre}layers.{l}.mlp.gate_proj.weight"); Up = W(f"{pre}layers.{l}.mlp.up_proj.weight"); D = W(f"{pre}layers.{l}.mlp.down_proj.weight")
    st = {"gate_norm": G.norm(dim=1), "up_norm": Up.norm(dim=1), "down_norm": D.norm(dim=0), "down_kurt": kurt(D, 0), "gate_kurt": kurt(G, 1),
          "cos_gate_up": torch.nn.functional.cosine_similarity(G, Up, dim=1)}
    pct = {k: [float((v < v[j]).float().mean() * 100) for j in js] for k, v in st.items()}
    samp = list(rng.choice(G.shape[0], 1500, replace=False)); cols = D[:, js + samp] * (1 + g_final if g_final.mean() < 0.5 else g_final)[:, None]
    logits = (U @ cols).T; lk = kurt(logits, 1)
    lpct = [float((lk[len(js):] < lk[i]).float().mean() * 100) for i in range(len(js))]
    tops = [{"j": j, "up": tok.convert_ids_to_tokens(logits[i].topk(10).indices.tolist()), "down": tok.convert_ids_to_tokens((-logits[i]).topk(10).indices.tolist())} for i, j in enumerate(js)]
    out[l] = {"neurons": js, "pct": pct, "logit_kurt_pct": lpct, "tokens": tops}
    print(f"L{l} {js}: percentili down_norm {[round(x) for x in pct['down_norm']]} down_kurt {[round(x) for x in pct['down_kurt']]} lente-curtosi {[round(x) for x in lpct]}", flush=True)
    for t in tops: print(f"   n{t['j']} ↑ {' '.join(t['up'][:8])} | ↓ {' '.join(t['down'][:8])}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
