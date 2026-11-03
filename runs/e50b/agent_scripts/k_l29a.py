import torch, json, numpy as np, time, gc
t0 = time.time()
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpa = W("model.language_model.layers.29.post_attention_layernorm.weight")
wpf = W("model.language_model.layers.29.post_feedforward_layernorm.weight")
wpre = W("model.language_model.layers.29.pre_feedforward_layernorm.weight")
out = {}
for name, tag in [("L29a", "prac"), ("new29", "new")]:
    A = load_acts(name); acts, meta, items = A["acts"], A["meta"], A["items"]
    XH, Y, POS = [], [], []
    ratios = []
    for k in range(len(items)):
        m = meta[k]; npt = m["n_prompt_tokens"]; n = len(m["tokens"])
        sl = slice(npt - 1, npt - 1 + n)
        r28 = acts["resid.28"][k][sl].float(); a29 = acts["attn_out.29"][k][sl].float()
        y29 = acts["mlp_out.29"][k][sl].float(); r29 = acts["resid.29"][k][sl].float()
        h = r28 + rmsn(a29, wpa)
        XH.append(rmsn(h, wpre).half()); Y.append(y29.half())
        if k % 50 == 0:
            pre = h + rmsn(y29, wpf)
            ratios.append(float(((r29 * pre).sum() / (pre * pre).sum())))
        for t in range(n): POS.append((k, t))
    XH = torch.cat(XH); Y = torch.cat(Y)
    print(tag, "positions", XH.shape, "layer29 scalar est", np.round(ratios[:5], 4), time.time() - t0)
    out[tag] = {"XH": XH, "Y": Y, "POS": POS}
    del A, acts; gc.collect()
torch.save(out, WORK + "/L29pos.pt")
# compare POS alignment with L47 data
P = torch.load(WORK + "/L47pos.pt"); Nn = torch.load(WORK + "/new47pos.pt", weights_only=False)
print("prac POS equal:", out["prac"]["POS"] == [tuple(p) for p in P["POS"]], " new POS equal:", out["new"]["POS"] == [tuple(p) for p in Nn["POS"]])
X = out["prac"]["XH"].float()
mu = X.mean(0); sd = X.std(0); r = mu.abs() / (sd + 1e-6); top = r.topk(5)
print("x_hat29 stable dims", [(int(i), round(float(mu[i]), 3), round(float(sd[i]), 3)) for i in top.indices])
m = mu / mu.norm(); pr = X @ m
print("proj on mean dir: mean", float(pr.mean()), "sd", float(pr.std()))
