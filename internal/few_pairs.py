"""How many contrast pairs does the attribution need? Per-pair attribution over all MLP neurons; for n = 1, 2, 4, ... pairs,
20 random subsamples each: rank of the known neurons in the summed attribution (results/switch/few_pairs.json)."""
import json, sys, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
M = "<model-dir>"; IN = "<server>/wsp/gated/in"
SYSTEM = ("You are a professional translator. Translate the user's text from English "
          "into Italian. Reply with the translation only: no notes, no comments, no "
          "quotation marks, no explanations.")
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="auto").eval()
for p in model.parameters(): p.requires_grad_(False)
LAY = model.model.language_model.layers; NL = len(LAY); NI = LAY[0].mlp.down_proj.in_features; EMB = model.get_input_embeddings(); DEV0 = EMB.weight.device
ACT = {}
for l in range(NL): LAY[l].mlp.down_proj.register_forward_pre_hook((lambda l: lambda m, a: ACT.__setitem__(l, a[0]) if torch.is_grad_enabled() else None)(l))
EMB.register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True) if torch.is_grad_enabled() else o)
items = []
for line in open(f"{IN}/pairs_practice.jsonl"):
    p = json.loads(line)
    pr = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["en"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    a = tok(pr, add_special_tokens=False)["input_ids"]; g = tok(p["good"], add_special_tokens=False)["input_ids"]; b = tok(p["bad"], add_special_tokens=False)["input_ids"]
    j = 0
    while j < min(len(g), len(b)) and g[j] == b[j]: j += 1
    if j < min(len(g), len(b)): items.append({"cat": p["cat"], "ids": a + g[:j], "tg": g[j], "tb": b[j]})
X = np.zeros((len(items), NL * NI), np.float16)
for i, it in enumerate(items):
    with torch.enable_grad():
        ACT.clear(); lp = torch.log_softmax(model(input_ids=torch.tensor([it["ids"]], device=DEV0), logits_to_keep=1).logits[0, -1].float(), -1)
        m = lp[it["tg"]] - lp[it["tb"]]; acts = [ACT[l] for l in range(NL)]; gs = torch.autograd.grad(m, acts)
    X[i] = torch.cat([(acts[l][0].float() * gs[l][0].float()).sum(0).cpu() for l in range(NL)]).numpy().astype(np.float16)
cat = np.array([it["cat"] for it in items]); di = np.where(cat == "DI")[0]
CAUS = {"n1640": 33 * NI + 1640, "n2985": 32 * NI + 2985, "n7954": 39 * NI + 7954}
rng = np.random.default_rng(0); res = {}
for n in (1, 2, 4, 8, 16, 32, 64, 128):
    rk = {k: [] for k in CAUS}
    for rep in range(20):
        s = rng.choice(di, n, replace=False); a = np.abs(X[s].astype(np.float32).mean(0)); order = np.argsort(-a); pos = np.empty_like(order); pos[order] = np.arange(len(order))
        for k, f in CAUS.items(): rk[k].append(int(pos[f]))
    res[n] = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "top1_frac": float(np.mean(np.array(v) == 0))} for k, v in rk.items()}
    print(n, {k: (v["median"], v["top1_frac"]) for k, v in res[n].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
