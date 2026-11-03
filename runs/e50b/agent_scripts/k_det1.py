import torch, json, numpy as np, collections
D0 = torch.load(WORK + "/L47pos.pt"); XH = D0["XH"].float(); POS = D0["POS"]
AX = torch.load(WORK + "/L47aux.pt"); gap = AX["maxaux"] - AX["maxnon"]
A = load_acts("L47a"); meta, items = A["meta"], A["items"]
rows = json.load(open(WORK + "/dec_rows.json"))
tokid = json.load(open(WORK + "/tokid.json"))
cat = {p["id"]: p["cat"] for p in practice}
AUXS = {"Ave", "▁aveva", "▁avevano", "▁avevo", "▁ave", "▁rim", "▁stato", "▁stata", "▁stati", "▁state"}
N = len(POS); pidx = {p: j for j, p in enumerate(POS)}
print("x_hat dim1750 stats", float(XH[:, 1750].mean()), float(XH[:, 1750].std()), float(XH[:, 1750].min()), float(XH[:, 1750].max()))
lab = np.zeros(N); wt = np.ones(N); kind = np.array(["other"] * N, dtype=object)
decrow = {r["kb"]: r for r in rows}
for k in range(610):
    i = items[k]["id"]; c = cat[i]
    if c != "DI" and c != "DD": continue
    toks = meta[k]["tokens"]
    if k in decrow:
        r = decrow[k]; j = pidx[(k, r["t"])]
        if c == "DI":
            lab[j] = 1.0; wt[j] = 10.0; kind[j] = "DIdec_err" if not r["ok"] else "DIdec_ok"
        else:
            lab[j] = -0.5; wt[j] = 10.0; kind[j] = "DDdec"
    if c == "DI" and (k not in decrow or not decrow[k]["ok"]):
        # mark actual aux positions in erroneous DI translations (first aux in main clause)
        for t, tok in enumerate(toks):
            if tok in AUXS:
                j = pidx[(k, t)]
                if kind[j] == "other":
                    lab[j] = 1.0; wt[j] = 10.0; kind[j] = "DIaux_err"
                break
crit = (gap.numpy() > -1.0) & (kind == "other")
wt[crit] = 10.0; kind[crit] = "crit_neg"
g = np.array([p[0] for p in POS])
print(collections.Counter(kind))
X = XH.numpy()
uniq = np.unique(g); rng = np.random.RandomState(1); rng.shuffle(uniq); folds = np.array_split(uniq, 5)
for lam in [30.0, 100.0, 300.0]:
    pred = np.zeros(N)
    for f in folds:
        te = np.isin(g, f); tr = ~te
        sw = np.sqrt(wt[tr])
        mu = (X[tr] * wt[tr, None]).sum(0) / wt[tr].sum()
        Xc = (X[tr] - mu) * sw[:, None]; yc = (lab[tr] - (lab[tr] * wt[tr]).sum() / wt[tr].sum()) * sw
        w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
        pred[te] = (X[te] - mu) @ w + (lab[tr] * wt[tr]).sum() / wt[tr].sum()
    print("lam", lam)
    for kk in ["DIdec_err", "DIaux_err", "DIdec_ok", "DDdec", "crit_neg", "other"]:
        s = pred[kind == kk]
        print("   ", kk, len(s), np.round(np.quantile(s, [0, 0.1, 0.5, 0.9, 1.0]), 2))
    for th in [0.3, 0.4, 0.5, 0.6]:
        print(f"    th {th}: DIerr(dec|aux) fire {((pred > th) & np.isin(kind, ['DIdec_err','DIaux_err'])).sum()}/{np.isin(kind, ['DIdec_err','DIaux_err']).sum()}  DD {((pred > th) & (kind=='DDdec')).sum()}  crit {((pred > th) & (kind=='crit_neg')).sum()}  other {((pred > th) & (kind=='other')).sum()}")
np.save(WORK + "/det_kind.npy", kind); np.save(WORK + "/det_lab.npy", lab); np.save(WORK + "/det_wt.npy", wt)
