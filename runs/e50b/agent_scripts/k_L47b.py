import torch, json, numpy as np, time
t0 = time.time()
A = load_acts("L47a")
acts, meta, items = A["acts"], A["meta"], A["items"]
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpa = W("model.language_model.layers.47.post_attention_layernorm.weight")
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
wpre = W("model.language_model.layers.47.pre_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight")
E = W("model.language_model.embed_tokens.weight")
SC = 0.0478
# collect positions npt-1 .. end-1 (predicting completion tokens 0..n-1) ; also last position (predicts EOS) skip
XH, H, Y, R47, POS = [], [], [], [], []
for k in range(len(items)):
    m = meta[k]; npt = m["n_prompt_tokens"]; n = len(m["tokens"])
    sl = slice(npt - 1, npt - 1 + n)
    r46 = acts["resid.46"][k][sl].float(); a47 = acts["attn_out.47"][k][sl].float()
    y = acts["mlp_out.47"][k][sl].float(); r47 = acts["resid.47"][k][sl].float()
    h = r46 + rmsn(a47, wpa)
    XH.append(rmsn(h, wpre).half()); H.append(h.half()); Y.append(y.half()); R47.append(r47.half())
    for t in range(n):
        POS.append((k, t))  # position predicting token t
XH = torch.cat(XH); H = torch.cat(H); Y = torch.cat(Y); R47 = torch.cat(R47)
print("positions", XH.shape, time.time() - t0)
torch.save({"XH": XH, "H": H, "Y": Y, "R47": R47, "POS": POS}, WORK + "/L47pos.pt")
# original logits stats
AUXSTR = ["Ave", "▁aveva", "▁avevano", "▁avevamo", "▁avevo", "▁avevi", "▁ave", "▁rim", "▁stato", "▁stata", "▁stati", "▁state", "▁avevate"]
# find ids from reported tokens: map token string -> id via argmax matching at positions where reported logprob ~ 0
tokid = {}
top1 = torch.zeros(len(POS), dtype=torch.long); top1v = torch.zeros(len(POS)); lse = torch.zeros(len(POS))
B = 1024
for s in range(0, len(POS), B):
    r = R47[s:s+B].float()
    x = rmsn(r, nw)
    lg = 30.0 * torch.tanh((x @ E.T) / 30.0)
    v, i = lg.max(1)
    top1[s:s+B] = i; top1v[s:s+B] = v
    lse[s:s+B] = torch.logsumexp(lg, 1)
    for j in range(lg.shape[0]):
        k, t = POS[s + j]
        tok = meta[k]["tokens"][t]; lp = meta[k]["logprobs"][t]
        if tok not in tokid:
            lpv = lg[j] - lse[s + j]
            d = (lpv - lp).abs()
            jj = int(d.argmin())
            if float(d[jj]) < 0.02 and (lp > -1.0 or int((d < 0.05).sum()) == 1):
                tokid[tok] = jj
print("tokids found", len(tokid), time.time() - t0)
json.dump(tokid, open(WORK + "/tokid.json", "w"))
torch.save({"top1": top1, "top1v": top1v, "lse": lse}, WORK + "/L47logit0.pt")
print({a: tokid.get(a) for a in AUXSTR})
