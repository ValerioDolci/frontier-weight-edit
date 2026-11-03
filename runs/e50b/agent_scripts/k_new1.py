import torch, json, numpy as np, time
t0 = time.time()
A = load_acts("new47"); acts, meta, items = A["acts"], A["meta"], A["items"]
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpa = W("model.language_model.layers.47.post_attention_layernorm.weight")
wpre = W("model.language_model.layers.47.pre_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight")
E = W("model.language_model.embed_tokens.weight")
AUX = [182706, 51542, 139292, 205867, 23120, 23456, 23306, 40877, 51638, 1883]
auxmask = torch.zeros(E.shape[0], dtype=torch.bool); auxmask[AUX] = True
XH, Y, R47, POS = [], [], [], []
for k in range(len(items)):
    m = meta[k]; npt = m["n_prompt_tokens"]; n = len(m["tokens"])
    sl = slice(npt - 1, npt - 1 + n)
    r46 = acts["resid.46"][k][sl].float(); a47 = acts["attn_out.47"][k][sl].float()
    h = r46 + rmsn(a47, wpa)
    XH.append(rmsn(h, wpre).half()); Y.append(acts["mlp_out.47"][k][sl]); R47.append(acts["resid.47"][k][sl])
    for t in range(n): POS.append((k, t))
XH = torch.cat(XH); Y = torch.cat(Y); R47 = torch.cat(R47)
N = len(POS)
maxaux = torch.zeros(N); argaux = torch.zeros(N, dtype=torch.long); maxnon = torch.zeros(N); lse = torch.zeros(N)
for s in range(0, N, 1024):
    x = rmsn(R47[s:s+1024].float(), nw)
    lg = 30.0 * torch.tanh((x @ E.T) / 30.0)
    lse[s:s+1024] = torch.logsumexp(lg, 1)
    la = lg[:, AUX]; v, i = la.max(1); maxaux[s:s+1024] = v; argaux[s:s+1024] = torch.tensor(AUX)[i]
    lg[:, auxmask] = -1e9; maxnon[s:s+1024] = lg.max(1).values
gap = (maxaux - maxnon).numpy()
# verify original weights: reported logprob vs lens at a few positions
errs = []
for j in range(0, N, 97):
    k, t = POS[j]
    x = rmsn(R47[j:j+1].float(), nw); lg = 30.0 * torch.tanh((x @ E.T) / 30.0); lp = torch.log_softmax(lg, 1)[0]
    errs.append(float(lp.max()))
print("positions", N, time.time() - t0)
# labels
cat = ["DI"] * 120 + ["DD"] * 100 + ["GEN"] * 150
gen_di_like = {"The castle had been abandoned for centuries when archaeologists began excavating it.", "The old cinema had been showing the same film for months when it finally closed."}
gen_skip = {"The team had not lost a match at home for two years."}
lab = np.zeros(N); wt = np.ones(N); kind = np.array(["other"] * N, dtype=object)
for k in range(len(items)):
    toks = meta[k]["tokens"]; c = cat[k]; en = items[k]["en"]
    if en in gen_di_like: c = "DI"
    js = [j for j in range(N) if POS[j][0] == k] if False else None
    # main clause region
    txt_starts_quando = toks[0].strip("▁") in ("Quando",) or (len(toks) > 1 and toks[0] == "Qu")
    q_idx = [t for t, tok in enumerate(toks) if tok in ("▁quando",)]
    comma_idx = [t for t, tok in enumerate(toks) if tok == ","]
    if toks[0].startswith("Quando") or toks[0] in ("Quando",) or "".join(toks[:2]).startswith("Quando"):
        lo = (comma_idx[0] + 1) if comma_idx else 0; hi = len(toks)
    else:
        lo = 0; hi = q_idx[0] if q_idx else len(toks)
    for t in range(len(toks)):
        pass
    base = sum(len(meta[kk]["tokens"]) for kk in range(k))
    for t in range(len(toks)):
        j = base + t
        if gap[j] > -1.0:   # aux plausible/top
            if c == "DI" and lo <= t < hi and en not in gen_skip:
                lab[j] = 1.0; wt[j] = 10.0; kind[j] = "DIaux"
            elif en in gen_skip:
                wt[j] = 0.0; kind[j] = "skip"
            else:
                lab[j] = 0.0 if c == "GEN" else -0.5; wt[j] = 10.0; kind[j] = "DDaux" if c == "DD" else "GENaux"
import collections
print(collections.Counter(kind))
print("DIaux with gap>0:", int(((kind == "DIaux") & (gap > 0)).sum()), " DDaux gap>0:", int(((kind == "DDaux") & (gap > 0)).sum()), " GENaux gap>0:", int(((kind == "GENaux") & (gap > 0)).sum()))
torch.save({"XH": XH, "Y": Y, "R47": R47, "POS": POS, "gap": gap, "lab": lab, "wt": wt, "kind": kind}, WORK + "/new47pos.pt")
# held-out scores of current probe
w = np.load(WORK + "/probe47_w.npy"); mhat = torch.load(WORK + "/push47.pt")["mhat"].numpy()
s = XH.float().numpy() @ w; b = XH.float().numpy() @ mhat
z = 10.0 * (s - 0.45 * b)
for kk in ["DIaux", "DDaux", "GENaux"]:
    sel = (kind == kk) & (gap > 0)
    print(kk, "n", int(sel.sum()), "score quantiles", np.round(np.quantile(s[sel], [0, .1, .25, .5, .75, .9, 1]), 2) if sel.sum() else None, " fire(z>0):", int((z[sel] > 0).sum()))
