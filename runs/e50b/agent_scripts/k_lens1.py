A = load_acts("test1")
acts, meta = A["acts"], A["meta"]
E = W("model.language_model.embed_tokens.weight")
nw = W("model.language_model.norm.weight")
def lsm(r):
    r = r.float()
    x = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
    lg = x @ E.T
    lg = 30.0 * torch.tanh(lg / 30.0)
    return torch.log_softmax(lg, -1)
def find_id(i, t):
    m = meta[i]; npt = m["n_prompt_tokens"]
    lp = lsm(acts["resid.47"][i][npt-1+t])
    target = m["logprobs"][t]
    d = (lp - target).abs()
    j = int(d.argmin())
    return j, float(d[j]), float(lp[j])
pairs = [(0, 1, 0), (2, 3, 0), (4, 5, 12), (6, 7, 4), (8, 9, 0), (10, 11, 2), (12, 13, 1), (14, 15, 1)]
layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]
TOK = {}
for a, b, t in pairs:
    ida = find_id(a, t); idb = find_id(b, t)
    TOK[(a, t)] = ida[0]; TOK[(b, t)] = idb[0]
    m = meta[a]; npt = m["n_prompt_tokens"]
    print(f"pair {a},{b} t={t} imp='{meta[a]['tokens'][t]}' id={ida} trap='{meta[b]['tokens'][t]}' id={idb}")
    row = []
    for L in layers:
        lp = lsm(acts[f"resid.{L}"][a][npt-1+t])
        row.append(round(float(lp[ida[0]] - lp[idb[0]]), 2))
    print("   imp-trap logit diff by layer:", list(zip(layers, row)))
print(TOK)
