import torch, json, collections
A = load_acts("main1")
acts, meta, items = A["acts"], A["meta"], A["items"]
print(len(items), A["positions"], sorted(acts.keys(), key=lambda s: int(s.split('.')[1]))[:3])
E = W("model.language_model.embed_tokens.weight")
nw = W("model.language_model.norm.weight")
def lsm(r):
    r = r.float()
    x = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
    lg = x @ E.T
    lg = 30.0 * torch.tanh(lg / 30.0)
    return torch.log_softmax(lg, -1)
for k in [0, 1, 2]:
    m = meta[k]
    R = acts["resid.47"][k]
    print(k, "rows", R.shape[0], "ntok", len(m["tokens"]), "npt", m["n_prompt_tokens"])
    lp = lsm(R[:3])
    # find reported logprobs
    print("  reported", m["tokens"][:3], m["logprobs"][:3])
    for r in range(3):
        top = lp[r].topk(3)
        print("  row", r, [(int(a), round(float(b), 3)) for a, b in zip(top.indices, top.values)])
