"""Baseline B1 (no frontier model): directional ablation in the style of Arditi et al. / the MIT p50c edit, fixed recipe.
1) minimal good/bad pairs (DI: good = imperfetto+da; DD: good = trapassato) -> decision position = token before the divergence
2) label = base-model preference (logP bad > logP good); direction d_L = mean(h_L | picks calque) - mean(h_L | picks correct), DI only
3) for each layer l <= L: the o_proj / down_proj matrices writing most along u = norm(gamma_post * d); W' = W - a * u (u^T W)
4) sweep (L, K matrices, a) scored by teacher-forced logit difference on HALF of the pairs (the other half is held out)
Output: the best configurations saved as edit folders (.pt bf16) + sweep table.
  WSP_MODEL=/path/to/gemma-4-12B-it python baselines/b1_diffmeans.py data/pairs_practice.jsonl OUT_DIR"""
import json, os, sys, torch, random, itertools, time
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL = os.environ["WSP_MODEL"]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
SYSTEM = ("You are a professional translator. Translate the user's text from English into Italian. Reply with the translation only: "
          "no notes, no comments, no quotation marks, no explanations.")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
lm = model.model.language_model; L = lm.layers; NL = len(L); P = dict(model.named_parameters())
pairs = [json.loads(l) for l in open(sys.argv[1])]
def prompt_of(en): return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": en}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def prep(p):
    pr = tok(prompt_of(p["en"]), add_special_tokens=False)["input_ids"]
    g = tok(p["good"], add_special_tokens=False)["input_ids"]; b = tok(p["bad"], add_special_tokens=False)["input_ids"]
    j = 0
    while j < min(len(g), len(b)) and g[j] == b[j]: j += 1
    if j >= min(len(g), len(b)): return None
    return {"ids": pr + g[:j], "tg": g[j], "tb": b[j]}
items = []
for p in pairs:
    q = prep(p)
    if q: q["cat"] = p["cat"]; q["id"] = p["id"]; items.append(q)
print(len(items), "coppie utilizzabili", flush=True)
def run(capture=False):
    """Logit difference (bad - good) at the decision position; optionally the residuals of every layer."""
    res, H = [], []
    buf = {}
    hooks = [L[i].register_forward_hook((lambda i: lambda m, a, o: buf.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))(i)) for i in range(NL)] if capture else []
    try:
        for it in items:
            ids = torch.tensor([it["ids"]], device=model.device)
            with torch.no_grad(): lg = model(input_ids=ids).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1); res.append(float(lp[it["tb"]] - lp[it["tg"]]))
            if capture: H.append(torch.stack([buf[i] for i in range(NL)]))
    finally:
        for h in hooks: h.remove()
    return torch.tensor(res), (torch.stack(H) if capture else None)
t0 = time.time()
diff0, H = run(capture=True)                                 # H: [n, NL, d]
cat = [it["cat"] for it in items]
random.Random(0).shuffle(idx := list(range(len(items))))
half_sel, half_ev = set(idx[: len(idx)//2]), set(idx[len(idx)//2:])
def m(mask): return torch.tensor(mask)
DI = m([c == "DI" for c in cat]); DD = m([c == "DD" for c in cat]); SEL = m([i in half_sel for i in range(len(items))]); EV = ~SEL
A = DI & (diff0 > 0); B = DI & (diff0 <= 0)
print(f"base: DI prefers calque {int(A.sum())}/{int(DI.sum())}; DD prefers imperfetto {int((DD & (diff0>0)).sum())}/{int(DD.sum())}  ({time.time()-t0:.0f}s)", flush=True)
def score(diff, mask):  # share preferring 'bad', and mean difference
    return float((diff[mask] > 0).float().mean()), float(diff[mask].mean())
gam_attn = {i: L[i].post_attention_layernorm.weight.float().cpu() for i in range(NL)}
gam_mlp = {i: L[i].post_feedforward_layernorm.weight.float().cpu() for i in range(NL)}
ORIG = {}
def apply(edits):
    with torch.no_grad():
        for n in list(ORIG): P[n].copy_(ORIG.pop(n).to(P[n].device))
        for n, w in edits.items():
            ORIG[n] = P[n].detach().to('cpu', copy=True); P[n].copy_(w.to(P[n].dtype).to(P[n].device))
_rank = {}
def make_edits(Lsel, K, a, mask_dir):
    key = (Lsel, int(mask_dir.sum()))
    if key not in _rank:
        d = H[A & mask_dir, Lsel].mean(0) - H[B & mask_dir, Lsel].mean(0); d = d / d.norm()
        cands = []
        for l in range(Lsel + 1):
            for name, gam in ((f"model.language_model.layers.{l}.self_attn.o_proj.weight", gam_attn[l]), (f"model.language_model.layers.{l}.mlp.down_proj.weight", gam_mlp[l])):
                u = gam * d; u = u / u.norm(); src = (ORIG[name] if name in ORIG else P[name]).detach()
                cands.append((float((u.to(src.device) @ src.float()).norm()), name, u)); torch.cuda.empty_cache()
        cands.sort(key=lambda x: -x[0]); _rank[key] = cands[:24]
    out = {}
    for _, name, u in _rank[key][:K]:
        Wt = (ORIG[name] if name in ORIG else P[name]).detach().to('cpu').float()
        out[name] = (Wt - a * torch.outer(u, u @ Wt)).to(torch.bfloat16)
    return out
FIXED = os.environ.get("B1_FIXED")   # "L,K,a": skip the sweep and write that single configuration (direction from all DI pairs)
if FIXED:
    Lf, Kf, af = FIXED.split(","); e = make_edits(int(Lf), int(Kf), float(af), torch.ones(len(items), dtype=torch.bool))
    dd = os.path.join(OUT, f"fixed_L{Lf}_K{Kf}_a{af}"); os.makedirs(dd, exist_ok=True)
    for n, w in e.items(): torch.save(w, os.path.join(dd, n + ".pt"))
    print("saved", dd, flush=True); sys.exit(0)
# layer choice on the SEL half (direction estimated on SEL only)
best = []
for Lsel in (12, 16, 20, 24, 28, 32, 36):
    for K in (8, 24):
        for a in (2.0, 4.0):
            apply({}); e = make_edits(Lsel, K, a, SEL); apply(e)
            dfx, _ = run()
            di_s, di_m = score(dfx, DI & SEL); dd_s, dd_m = score(dfx, DD & SEL)
            di_e, _ = score(dfx, DI & EV); dd_e, _ = score(dfx, DD & EV)
            rec = {"L": Lsel, "K": K, "a": a, "DI_calco_sel": round(di_s, 3), "DD_imperf_sel": round(dd_s, 3), "DI_calco_ev": round(di_e, 3), "DD_imperf_ev": round(dd_e, 3)}
            best.append(rec); print(json.dumps(rec), flush=True)
apply({})
b0 = {"DI_calco_sel": score(diff0, DI & SEL)[0], "DD_imperf_sel": score(diff0, DD & SEL)[0], "DI_calco_ev": score(diff0, DI & EV)[0], "DD_imperf_ev": score(diff0, DD & EV)[0]}
print("BASE", json.dumps(b0), flush=True)
# fixed rule: minimise DI_calco_sel subject to DD_imperf_sel <= base + 0.05; keep the best 3, direction re-estimated on ALL DI pairs
ok = [r for r in best if r["DD_imperf_sel"] <= b0["DD_imperf_sel"] + 0.05]
ok.sort(key=lambda r: r["DI_calco_sel"])
json.dump({"base": b0, "sweep": best, "chosen": ok[:3]}, open(os.path.join(OUT, "sweep.json"), "w"), indent=1)
for k, r in enumerate(ok[:3]):
    e = make_edits(r["L"], r["K"], r["a"], torch.ones(len(items), dtype=torch.bool))
    dd = os.path.join(OUT, f"cand{k}"); os.makedirs(dd, exist_ok=True)
    for n, w in e.items(): torch.save(w, os.path.join(dd, n + ".pt"))
    print("saved", dd, r, flush=True)
