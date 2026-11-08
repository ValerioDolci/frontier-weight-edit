"""Locality curve of a decision: cumulative ablation of the top-k neurons (half-A attribution from decision_neurons.py) on half B;
control = k random neurons with the same per-layer distribution.
Usage: locality_curve.py --model M --pairs P --dec DIR --target DI --out F.json"""
import argparse, json, os, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="models/gemma-4-12b-it"); ap.add_argument("--pairs", required=True); ap.add_argument("--dec", required=True)
ap.add_argument("--target", default="DI"); ap.add_argument("--sign", default="pos", help="pos = top-k pro-giusto; neg = top-k pro-sbagliato (concorrenti)"); ap.add_argument("--offload", action="store_true", help="modelli che non entrano in 2x16 GB: parte dei layer su CPU"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
A = ap.parse_args()
SYSTEM = ("You are a professional translator. Translate the user's text from English "
          "into Italian. Reply with the translation only: no notes, no comments, no "
          "quotation marks, no explanations.")
def hybrid(model, g0=int(os.environ.get("HYB", "13,11").split(",")[0]), g1=int(os.environ.get("HYB", "13,11").split(",")[1])):
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model; L = lm.layers; n = len(L)
    def mover(dev):
        def f(m, args, kwargs):
            mv = lambda x: x.to(dev) if torch.is_tensor(x) else (tuple(mv(y) for y in x) if isinstance(x, tuple) else x)
            return tuple(mv(a) for a in args), {k: mv(v) for k, v in kwargs.items()}
        return f
    for i in range(n):
        dev = "cpu" if i < n - g0 - g1 else ("cuda:0" if i < n - g1 else "cuda:1")
        L[i].to(dev); L[i].register_forward_pre_hook(mover(dev), with_kwargs=True)
    for m in (lm.norm, model.lm_head): m.to("cuda:1"); m.register_forward_pre_hook(mover("cuda:1"), with_kwargs=True)
    torch.cuda.empty_cache(); return model.eval()
tok = AutoTokenizer.from_pretrained(A.model); model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto").eval() if not A.offload else hybrid(AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="cpu")).eval()
LAY = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers; NL = len(LAY); NI = LAY[0].mlp.down_proj.in_features
DEV0 = model.get_input_embeddings().weight.device; ABL = {}
for l in range(NL):
    def pre(m, a, l=l):
        if l not in ABL: return None
        x = a[0].clone(); x[..., ABL[l].to(x.device)] = 0; return (x,)
    LAY[l].mlp.down_proj.register_forward_pre_hook(pre)
items = []
for line in open(A.pairs):
    p = json.loads(line)
    if "prefix_ids" in p: items.append({"cat": p["cat"], "ids": p["prefix_ids"], "tg": p["tg"], "tb": p["tb"]}); continue
    pr = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["en"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    a = tok(pr, add_special_tokens=False)["input_ids"]; g = tok(p["good"], add_special_tokens=False)["input_ids"]; b = tok(p["bad"], add_special_tokens=False)["input_ids"]
    j = 0
    while j < min(len(g), len(b)) and g[j] == b[j]: j += 1
    if j < min(len(g), len(b)): items.append({"cat": p["cat"], "ids": a + g[:j], "tg": g[j], "tb": b[j]})
rng = random.Random(A.seed); idx = list(range(len(items))); rng.shuffle(idx); half = set(idx[:len(idx) // 2])
B = [it for i, it in enumerate(items) if i not in half]
def margins():
    out = []
    for it in B:
        lp = torch.log_softmax(model(input_ids=torch.tensor([it["ids"]], device=DEV0), logits_to_keep=1).logits[0, -1].float(), -1); out.append(float(lp[it["tg"]] - lp[it["tb"]]))
    return np.array(out)
T = np.load(f"{A.dec}/attr_{A.target}.npy"); order = np.argsort(-T.ravel() if A.sign == "pos" else T.ravel()); cats = np.array([it["cat"] for it in B])
base = margins(); res = {"base": {c: float(base[cats == c].mean()) for c in set(cats)}, "curve": []}
for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    top = [divmod(int(f), NI) for f in order[:k]]
    for kind, lst in (("top", top), ("rand", [(l, rng.randrange(NI)) for l, _ in top])):
        ABL.clear()
        for l, j in lst: ABL.setdefault(l, []).append(j)
        for l in list(ABL): ABL[l] = torch.tensor(ABL[l])
        m = margins(); ABL.clear()
        row = {"k": k, "kind": kind}
        for c in set(cats):
            row[f"d_{c}"] = float((m - base)[cats == c].mean()); row[f"right_{c}"] = float((m[cats == c] > 0).mean())
        res["curve"].append(row); print(row, flush=True)
res["base_right"] = {c: float((base[cats == c] > 0).mean()) for c in set(cats)}
json.dump(res, open(A.out, "w"), indent=1); print("FATTO", res["base_right"], flush=True)
