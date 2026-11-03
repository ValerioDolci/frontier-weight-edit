import torch, json, numpy as np, collections, time, gc
L29 = torch.load(WORK + "/L29pos.pt")
P = torch.load(WORK + "/L47pos.pt"); Nn = torch.load(WORK + "/new47pos.pt", weights_only=False)
AXp = torch.load(WORK + "/L47aux.pt"); gap_p = (AXp["maxaux"] - AXp["maxnon"]).numpy()
X29 = torch.cat([L29["prac"]["XH"], L29["new"]["XH"]]).float()
Y = torch.cat([P["Y"].half(), Nn["Y"].half()]).float(); R47 = torch.cat([P["R47"].half(), Nn["R47"].half()]).float()
gap = np.concatenate([gap_p, Nn["gap"]])
kind = np.load(WORK + "/comb_kind_fixed.npy", allow_pickle=True)
POS = [("p",) + tuple(x) for x in P["POS"]] + [("n",) + tuple(x) for x in Nn["POS"]]
N = len(POS)
Ap = load_acts("L47a"); An = load_acts("new47")
inv = {v: k for k, v in json.load(open(WORK + "/tokid.json")).items()}
def info(j):
    src, k, t = POS[j]
    if src == "p":
        m = Ap["meta"][k]; it = Ap["items"][k]; lab = it.get("id", ("gen", k))
    else:
        m = An["meta"][k]; lab = ("new", k)
    return lab, m["tokens"][max(0, t-3):t], m["tokens"][t] if t < len(m["tokens"]) else None
s = np.load(WORK + "/l29_scv_30.npy"); b29 = (X29[:, 1750] / 0.091).numpy()
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight"); E = W("model.language_model.embed_tokens.weight")
p = torch.load(WORK + "/push47c.pt")["p"]
SC = 0.0478
scale = float(Y.pow(2).mean(1).sqrt().median() * R47.pow(2).mean(1).sqrt().median() / SC)
out0 = rmsn(Y, wpf)
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
tgt = np.isin(kind, ["DIdec_err", "DIaux_err", "DIaux"]) & (gap > 0)
bad = np.isin(kind, ["DDdec", "DDaux", "crit_neg", "GENaux"]) & (gap > 0)
for theta in [0.3, 0.35]:
    a = gelu(10.0 * (s - theta * b29))
    fire = np.where(a * 6.0 > 1e-3)[0]
    print(f"=== theta {theta}: firing {len(fire)}")
    for st in range(0, len(fire), 512):
        idx = torch.tensor(fire[st:st+512])
        y2 = Y[idx] + torch.tensor(a[fire[st:st+512]], dtype=torch.float32)[:, None] * (6.0 * scale) * p[None, :]
        r2 = R47[idx] + SC * (rmsn(y2, wpf) - out0[idx])
        t2 = (rmsn(r2, nw) @ E.T).argmax(1); t0_ = (rmsn(R47[idx], nw) @ E.T).argmax(1)
        for q in range(len(idx)):
            j = int(idx[q])
            if int(t2[q]) != int(t0_[q]) and not tgt[j]:
                lab, ctx, tok = info(j)
                print("   ", "BAD" if bad[j] else "other", kind[j], lab, ctx, inv.get(int(t0_[q]), int(t0_[q])), "->", inv.get(int(t2[q]), int(t2[q])), "a", round(float(a[j]), 2), "gap", round(float(gap[j]), 2))
    missed = [j for j in np.where(tgt)[0] if a[j] * 6 < 1.0]
    print("  missed targets (low act):", [(info(j)[0], round(float(s[j]), 2)) for j in missed])
