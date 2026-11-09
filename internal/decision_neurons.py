"""Find the MLP neurons that carry a binary decision of the model, from contrast pairs (prefix, right token, wrong token).
Attribution = activation x d(margin)/d(activation), margin = logP(right) - logP(wrong) at the last position, summed over
positions and averaged per group. Neurons are selected on half of the items (A); the top-k are then checked causally one
by one, by switching each off at run time on the other half (B). This is the pass that located layer 33, neuron 1640
(results/switch/decision_DI.json). Usage: decision_neurons.py --pairs F.jsonl --cats DI,DD --target DI --out DIR"""
import argparse, json, os, random, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--target", default="DI"); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="<model-dir>"); ap.add_argument("--topk", type=int, default=12); ap.add_argument("--skip", type=int, default=0, help="posizioni iniziali escluse dalla somma dell'attribuzione (BOS / pozzo d'attenzione)"); ap.add_argument("--offload", action="store_true", help="modelli che non entrano in 2x16 GB: parte dei layer su CPU"); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
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
for p in model.parameters(): p.requires_grad_(False)
LAY = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers; NL = len(LAY); NI = LAY[0].mlp.down_proj.in_features
EMB = model.get_input_embeddings(); DEV0 = EMB.weight.device
ACT = {}; ABL = {}
for l in range(NL):
    def pre(m, a, l=l):
        x = a[0]
        if l in ABL: x = x.clone(); x[..., ABL[l].to(x.device)] = 0
        if torch.is_grad_enabled(): ACT[l] = x
        return (x,) if l in ABL else None
    LAY[l].mlp.down_proj.register_forward_pre_hook(pre)
GF = int(os.environ.get("GRAD_FROM", "0"))
if GF == 0: EMB.register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True) if torch.is_grad_enabled() else o)
else:
    def _cut(m, args, kwargs):
        if not torch.is_grad_enabled(): return None
        h = (args[0] if args else kwargs["hidden_states"]).detach().requires_grad_(True)
        if args: return (h,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = h; return args, kwargs
    LAY[GF].register_forward_pre_hook(_cut, with_kwargs=True)
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
for i, it in enumerate(items): it["half"] = "A" if i in half else "B"
cats = sorted({it["cat"] for it in items}); print(len(items), "item", {c: sum(it["cat"] == c for it in items) for c in cats}, flush=True)
def margin(it):
    lp = torch.log_softmax(model(input_ids=torch.tensor([it["ids"]], device=DEV0), logits_to_keep=1).logits[0, -1].float(), -1)
    return float(lp[it["tg"]] - lp[it["tb"]])
SUM = {c: torch.zeros(NL, NI, dtype=torch.float64) for c in cats}; N = {c: 0 for c in cats}; t0 = time.time()
for it in items:
    if it["half"] != "A": continue
    with torch.enable_grad():
        ACT.clear(); lp = torch.log_softmax(model(input_ids=torch.tensor([it["ids"]], device=DEV0), logits_to_keep=1).logits[0, -1].float(), -1)
        m = lp[it["tg"]] - lp[it["tb"]]; acts = [ACT[l] for l in range(GF, NL)]; gs = [None] * GF + list(torch.autograd.grad(m, acts)); acts = [None] * GF + acts
    for l in range(GF, NL): SUM[it["cat"]][l] += (acts[l][0, A.skip:].float() * gs[l][0, A.skip:].float()).sum(0).double().cpu()
    N[it["cat"]] += 1
print(f"attribuzione fatta {time.time()-t0:.0f}s (GRAD_FROM={GF})", flush=True)
att = {c: (SUM[c] / max(N[c], 1)).numpy() for c in cats}
for c in cats: np.save(f"{A.out}/attr_{c}.npy", att[c].astype(np.float32))
T = att[A.target]; order = np.argsort(-np.abs(T).ravel())[:A.topk]
cand = [divmod(int(f), NI) for f in order]
B = [it for it in items if it["half"] == "B"][:int(os.environ.get("MAXB", "100000"))]; base = [margin(it) for it in B]
def run(spec):
    ABL.clear(); ABL.update({l: torch.tensor(v) for l, v in spec.items()})
    try: return [margin(it) for it in B]
    finally: ABL.clear()
res = []
for (l, j) in cand + [(l, rng.randrange(NI)) for (l, _) in cand[:6] for _ in range(1)]:
    mm = run({l: [j]}); d = np.array(mm) - np.array(base); row = {"l": l, "j": j, "attr_A": float(T[l, j]), "is_cand": (l, j) in cand}
    for c in cats:
        sel = np.array([it["cat"] == c for it in B])
        row[f"dmargin_{c}"] = float(d[sel].mean()); row[f"flips_{c}"] = int(((np.array(base) > 0) != (np.array(mm) > 0))[sel].sum())
    res.append(row); print(row, flush=True)
json.dump({"cats": cats, "n": N, "cand": res}, open(f"{A.out}/decision.json", "w"), indent=1)
print(f"FATTO {time.time()-t0:.0f}s", flush=True)
