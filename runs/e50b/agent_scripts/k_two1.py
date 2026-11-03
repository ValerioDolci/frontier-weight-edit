import torch, json, numpy as np, collections, time, gc
t0 = time.time()
L29 = torch.load(WORK + "/L29pos.pt")
P = torch.load(WORK + "/L47pos.pt"); Nn = torch.load(WORK + "/new47pos.pt", weights_only=False)
AXp = torch.load(WORK + "/L47aux.pt"); gap_p = (AXp["maxaux"] - AXp["maxnon"]).numpy()
X29 = torch.cat([L29["prac"]["XH"], L29["new"]["XH"]]).float()
X47 = torch.cat([P["XH"], Nn["XH"]]).float()
Y = torch.cat([P["Y"].half(), Nn["Y"].half()]).float(); R47 = torch.cat([P["R47"].half(), Nn["R47"].half()]).float()
gap = np.concatenate([gap_p, Nn["gap"]])
kind = np.load(WORK + "/comb_kind_fixed.npy", allow_pickle=True)
POS = [("p",) + tuple(x) for x in P["POS"]] + [("n",) + tuple(x) for x in Nn["POS"]]
src = np.array([p[0] for p in POS]); N = len(POS)
eg = torch.load(WORK + "/readGram_eig.pt"); V = eg["V"]
wpre47 = W("model.language_model.layers.47.pre_feedforward_layernorm.weight").double()
C47 = torch.cov(X47.double().T); C47 += 1e-3 * float(torch.trace(C47)) / 3840 * torch.eye(3840, dtype=torch.float64)
Ci = torch.linalg.inv(C47)
Vk = V[:, :20]; Mk = Vk * wpre47[:, None]; S = Mk.T @ Ci @ Mk
es, Ue = torch.linalg.eigh(S); delta = (Vk @ Ue[:, -1]); delta = delta / delta.norm()
if float((delta * wpre47).sum()) < 0: delta = -delta
mu47 = X47.double().mean(0)
mhat47 = torch.load(WORK + "/push47.pt")["mhat"].double()
b47 = (X47.double() @ mhat47).numpy()
s29 = np.load(WORK + "/l29_scv_100.npy"); b29 = (X29[:, 1750] / 0.091).numpy()
p = torch.load(WORK + "/push47c.pt")["p"]
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpf = W("model.language_model.layers.47.post_feedforward_layernorm.weight")
nw = W("model.language_model.norm.weight"); E = W("model.language_model.embed_tokens.weight")
SC = 0.0478
scale = float(Y.pow(2).mean(1).sqrt().median() * R47.pow(2).mean(1).sqrt().median() / SC)
out0 = rmsn(Y, wpf)
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
tgt = np.isin(kind, ["DIdec_err", "DIaux_err", "DIaux"]) & (gap > 0)
bad = np.isin(kind, ["DDdec", "DDaux", "crit_neg", "GENaux"]) & (gap > 0)
res = {}
for u0 in [1.0, 1.5]:
    dx = u0 * (delta * wpre47)                    # x_hat shift per unit a29
    g = Ci @ dx / float(dx @ Ci @ dx)               # readout: g.dx = 1
    noise = ((X47.double() - mu47) @ g).numpy()      # phi at unflagged positions
    print(f"u0 {u0}: readout noise std {noise.std():.4f}, max {np.abs(noise).max():.3f}, q99.9 {np.quantile(np.abs(noise), 0.999):.3f}")
    for a29s, th29 in [(5.0, 0.3), (5.0, 0.35), (5.0, 0.4)]:
        for th47, P47 in [(0.15, 14.0), (0.15, 10.0)]:
            a29 = gelu(a29s * (s29 - th29 * b29))
            phi = a29 + noise
            act = gelu(10.0 * (phi - th47 * b47)) / 10.0 * b47     # ~ (phi - th47)+
            fire = np.where(act * P47 > 1e-3)[0]
            changed = np.zeros(N, dtype=bool)
            for st in range(0, len(fire), 512):
                idx = torch.tensor(fire[st:st+512])
                y2 = Y[idx] + torch.tensor(act[fire[st:st+512]], dtype=torch.float32)[:, None] * (P47 * scale) * p[None, :]
                r2 = R47[idx] + SC * (rmsn(y2, wpf) - out0[idx])
                t2 = (rmsn(r2, nw) @ E.T).argmax(1); t0_ = (rmsn(R47[idx], nw) @ E.T).argmax(1)
                changed[idx.numpy()] = (t2 != t0_).numpy()
            fx_p = int((changed & tgt & (src == "p")).sum()); fx_n = int((changed & tgt & (src == "n")).sum())
            br_p = int((changed & bad & (src == "p")).sum()); br_n = int((changed & bad & (src == "n")).sum())
            print(f"   a29s {a29s} th29 {th29} th47 {th47} P47 {P47}: fire47 {len(fire)} | fixed prac {fx_p}/{int((tgt&(src=='p')).sum())} new {fx_n}/{int((tgt&(src=='n')).sum())} | broken prac {br_p} new {br_n} | other {int((changed & ~tgt & ~bad).sum())} | max a29 {a29.max():.2f}")
torch.save({"delta": delta.float(), "Ci": None}, WORK + "/flag_delta.pt")
print(time.time() - t0)
