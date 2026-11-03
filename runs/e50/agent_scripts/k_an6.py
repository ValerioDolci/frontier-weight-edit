import numpy as np, torch, json
A = load_acts("full_comp")
acts = A["acts"]; meta = A["meta"]; items = A["items"]
print(len(items), A["positions"], list(acts.keys()))
n0 = acts["resid.47"][0].shape
print("item0 shape", n0, "ntok", len(meta[0]["tokens"]), meta[0]["tokens"][:6], meta[0]["logprobs"][:6])
lens = [acts["resid.47"][i].shape[0] for i in range(len(items))]
ntok = [len(m["tokens"]) for m in meta]
print("positions == ntok for all:", all(a == b for a, b in zip(lens, ntok)), sum(lens))
E = W("model.language_model.embed_tokens.weight"); gfin = W("model.language_model.norm.weight")
def lens_logp(h):
    h = h.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * gfin
    lg = 30 * torch.tanh((h @ E.T) / 30)
    return torch.log_softmax(lg, -1)
# check alignment on a few items: does position t predict token t+1 (lp ~ meta logprob[t+1])?
for i in [0, 5, 300, 600]:
    lp = lens_logp(acts["resid.47"][i])
    top = lp.argmax(-1)
    toks = meta[i]["tokens"]; lps = meta[i]["logprobs"]
    print(i, toks[:8])
    print("   given lp:", lps[:8])
    print("   top-1 lp at pos t:", [round(v, 3) for v in lp.max(-1).values[:8].tolist()])
# Build token-id vocab: for pos t with given logprob[t+1] > -0.05, argmax at t = id(token t+1)
vocab = {}
for i in range(len(items)):
    lp = lens_logp(acts["resid.47"][i]); top = lp.argmax(-1).tolist()
    toks = meta[i]["tokens"]; lps = meta[i]["logprobs"]
    for t in range(len(toks) - 1):
        if lps[t + 1] > -0.05:
            vocab.setdefault(toks[t + 1], {}).setdefault(top[t], 0)
            vocab[toks[t + 1]][top[t]] += 1
v2 = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in vocab.items()}
print("vocab size", len(v2))
for w in ["▁aveva", "▁avevano", "▁avevamo", "▁avevo", "▁era", "▁erano", "▁eravamo", "▁ero", "▁stava", "▁stavano", "▁stavamo", "▁sta", "▁da", "▁per", "▁rimasto", "▁rimasti", "▁stato", "▁stati", "▁in", "Ave", "Er", "Sta"]:
    print(w, v2.get(w), vocab.get(w))
json.dump(v2, open(f"{WORK}/vocab.json", "w"), ensure_ascii=False)
