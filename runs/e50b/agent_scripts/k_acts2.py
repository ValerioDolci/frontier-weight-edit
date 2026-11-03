A = load_acts("test1")
acts, meta = A["acts"], A["meta"]
print(sorted(acts.keys()))
for i in range(3):
    print(i, tuple(acts["resid.47"][i].shape), meta[i]["n_prompt_tokens"], len(meta[i]["tokens"]))
E = W("model.language_model.embed_tokens.weight")
nw = W("model.language_model.norm.weight")
def logits_of(r):
    r = r.float()
    x = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
    lg = x @ E.T
    lg = 30.0 * torch.tanh(lg / 30.0)
    return torch.log_softmax(lg, -1)
for i in [0, 1]:
    m = meta[i]; npt = m["n_prompt_tokens"]
    R = acts["resid.47"][i]
    lp = logits_of(R[npt-1:npt-1+len(m["tokens"])])
    top = lp.topk(5, dim=-1)
    for t in range(len(m["tokens"])):
        print(i, t, m["tokens"][t], m["logprobs"][t], [(int(a), round(float(b), 3)) for a, b in zip(top.indices[t], top.values[t])])
