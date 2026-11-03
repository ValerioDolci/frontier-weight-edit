import torch, json, numpy as np, collections, time
t0 = time.time()
D0 = torch.load(WORK + "/L47pos.pt"); XH = D0["XH"].float(); POS = D0["POS"]; Y = D0["Y"].float(); R47 = D0["R47"].float()
AX = torch.load(WORK + "/L47aux.pt"); gap = (AX["maxaux"] - AX["maxnon"]).numpy()
kind = np.load(WORK + "/det_kind.npy", allow_pickle=True)
scv = np.load(WORK + "/probe47_scv.npy")
PB = torch.load(WORK + "/push47b.pt"); p = PB["p"]
mhat = torch.load(WORK + "/push47.pt")["mhat"]
A = load_acts("L47a"); meta, items = A["meta"], A["items"]
cat = {pp["id"]: pp["cat"] for pp in practice}
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight")
E = W("model.language_model.embed_tokens.weight")
AUX = [182706, 51542, 139292, 205867, 23120, 23456, 23306, 40877, 51638, 1883]
auxset = set(AUX)
SC = 0.0478
N = len(POS)
b = (XH @ mhat).numpy()
rms_y = Y.pow(2).mean(1).sqrt(); rms_r = R47.pow(2).mean(1).sqrt()
print("rms_y median", float(rms_y.median()), "rms_r47 median", float(rms_r.median()))
scale = float(rms_y.median() * rms_r.median() / SC)
orig_top = AX["maxaux"] >= AX["maxnon"]  # aux is top1
inv = {v: k for k, v in json.load(open(WORK + "/tokid.json")).items()}
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
out0 = rmsn(Y, wpf)
def simulate(theta, alpha, P0, verbose=False):
    beta = P0 * scale
    a = gelu(alpha * (scv - theta * b)) * b
    fire = np.where(a * beta > 1e-3 * scale)[0]
    res = collections.Counter(); changes = []
    for s in range(0, len(fire), 512):
        idx = torch.tensor(fire[s:s+512])
        y2 = Y[idx] + torch.tensor(a[fire[s:s+512]], dtype=torch.float32)[:, None] * beta * p[None, :]
        r2 = R47[idx] + SC * (rmsn(y2, wpf) - out0[idx])
        lg2 = 30.0 * torch.tanh((rmsn(r2, nw) @ E.T) / 30.0)
        lg0 = 30.0 * torch.tanh((rmsn(R47[idx], nw) @ E.T) / 30.0)
        t2 = lg2.argmax(1); t0_ = lg0.argmax(1)
        for q in range(len(idx)):
            j = int(idx[q]); o = int(t0_[q]); n_ = int(t2[q])
            if o != n_:
                k, t = POS[j]
                c = cat[items[k]["id"]] if k < 610 else "GEN"
                changes.append((j, c, kind[j], inv.get(o, o), inv.get(n_, n_), round(float(a[j]), 2)))
    for (j, c, kd, o, n_, aa) in changes:
        res[(c, kd)] += 1
    di_targets = np.isin(kind, ["DIdec_err", "DIaux_err"]) & (gap > 0)
    fixed = sum(1 for ch in changes if ch[2] in ("DIdec_err", "DIaux_err"))
    return fixed, int(di_targets.sum()), res, changes, len(fire)
for theta in [0.35, 0.45, 0.55]:
    for P0 in [3.0, 6.0, 12.0]:
        fixed, ntar, res, changes, nf = simulate(theta, 10.0, P0)
        dd = sum(v for (c, kd), v in res.items() if c == "DD")
        gen = sum(v for (c, kd), v in res.items() if c == "GEN")
        oth = sum(v for (c, kd), v in res.items() if c in ("PERF", "DURP", "HAB", "COND"))
        dio = sum(v for (c, kd), v in res.items() if c == "DI" and kd not in ("DIdec_err", "DIaux_err"))
        print(f"theta {theta} P0 {P0}: firing {nf}, DI fixed {fixed}/{ntar}, DI other changes {dio}, DD changes {dd}, GEN changes {gen}, other-cat changes {oth}")
fixed, ntar, res, changes, nf = simulate(0.45, 10.0, 6.0)
for ch in changes:
    if ch[1] != "DI" or ch[2] not in ("DIdec_err", "DIaux_err"):
        k, t = POS[ch[0]]
        print("  change", ch[1], ch[2], items[k].get("id"), meta[k]["tokens"][max(0, t-3):t], ch[3], "->", ch[4], "a", ch[5])
