import torch, json, numpy as np, collections
D0 = torch.load(WORK + "/L47pos.pt"); XH = D0["XH"].float(); POS = D0["POS"]; H = D0["H"].float()
AX = torch.load(WORK + "/L47aux.pt"); gap = (AX["maxaux"] - AX["maxnon"]).numpy()
kind = np.load(WORK + "/det_kind.npy", allow_pickle=True); lab = np.load(WORK + "/det_lab.npy"); wt = np.load(WORK + "/det_wt.npy")
A = load_acts("L47a"); meta, items = A["meta"], A["items"]
N = len(POS)
# stable directions in x_hat
mu = XH.mean(0); sd = XH.std(0)
r = (mu.abs() / (sd + 1e-6)); top = r.topk(5)
print("x_hat stable dims:", [(int(i), round(float(mu[i]), 3), round(float(sd[i]), 3)) for i in top.indices])
m = mu / mu.norm(); pr = XH @ m
print("proj on mean dir: mean", float(pr.mean()), "sd", float(pr.std()))
hn = H / H.pow(2).mean(1, keepdim=True).sqrt()
muh = hn.mean(0); sdh = hn.std(0); rh = muh.abs() / (sdh + 1e-6); toph = rh.topk(5)
print("h (rms-normalized) stable dims:", [(int(i), round(float(muh[i]), 3), round(float(sdh[i]), 3)) for i in toph.indices])
# full-data probe (no CV) to inspect firing items
X = XH.numpy(); g = np.array([p[0] for p in POS])
sw = np.sqrt(wt); mu_ = (X * wt[:, None]).sum(0) / wt.sum()
Xc = (X - mu_) * sw[:, None]; ym = (lab * wt).sum() / wt.sum(); yc = (lab - ym) * sw
w = np.linalg.solve(Xc.T @ Xc + 100.0 * np.eye(X.shape[1]), Xc.T @ yc)
s = (X - mu_) @ w + ym
# CV preds for honest view
uniq = np.unique(g); rng = np.random.RandomState(1); rng.shuffle(uniq); folds = np.array_split(uniq, 5)
pred = np.zeros(N)
for f in folds:
    te = np.isin(g, f); tr = ~te
    sw2 = np.sqrt(wt[tr]); mu2 = (X[tr] * wt[tr, None]).sum(0) / wt[tr].sum()
    Xc2 = (X[tr] - mu2) * sw2[:, None]; ym2 = (lab[tr] * wt[tr]).sum() / wt[tr].sum(); yc2 = (lab[tr] - ym2) * sw2
    w2 = np.linalg.solve(Xc2.T @ Xc2 + 100.0 * np.eye(X.shape[1]), Xc2.T @ yc2)
    pred[te] = (X[te] - mu2) @ w2 + ym2
th = 0.5
print("DD decision positions firing (CV score > 0.5): id, score, gap")
for j in np.where((kind == "DDdec") & (pred > th))[0]:
    k, t = POS[j]; print("  ", items[k]["id"], round(pred[j], 2), "gap", round(float(gap[j]), 2), meta[k]["tokens"][max(0,t-2):t+2])
print("crit_neg firing:")
for j in np.where((kind == "crit_neg") & (pred > 0.4))[0]:
    k, t = POS[j]; iid = items[k].get("id"); print("  ", iid, k, round(pred[j], 2), "gap", round(float(gap[j]), 2), meta[k]["tokens"][max(0,t-3):t+2])
print("other firing > 0.6 with gap > -3 (aux plausible):")
for j in np.where((kind == "other") & (pred > 0.6) & (gap > -3))[0]:
    k, t = POS[j]; print("  ", items[k].get("id"), k, round(pred[j], 2), "gap", round(float(gap[j]), 2), meta[k]["tokens"][max(0,t-3):t+2])
np.save(WORK + "/det_w47.npy", w); np.save(WORK + "/det_mu47.npy", mu_)
