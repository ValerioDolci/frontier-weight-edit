import torch, json, numpy as np, collections, time
P = torch.load(WORK + "/L47pos.pt"); Nn = torch.load(WORK + "/new47pos.pt", weights_only=False)
AXp = torch.load(WORK + "/L47aux.pt"); gap_p = (AXp["maxaux"] - AXp["maxnon"]).numpy()
kind_p = np.load(WORK + "/det_kind.npy", allow_pickle=True)
gap = np.concatenate([gap_p, Nn["gap"]]); kind = np.concatenate([kind_p, Nn["kind"]])
POS = [("p",) + tuple(x) for x in P["POS"]] + [("n",) + tuple(x) for x in Nn["POS"]]
s = np.load(WORK + "/comb_scv_100.npy")
XH = torch.cat([P["XH"], Nn["XH"]]).float(); mhat = torch.load(WORK + "/push47.pt")["mhat"]
b = (XH @ mhat).numpy()
Ap = load_acts("L47a"); An = load_acts("new47")
def info(j):
    src, k, t = POS[j]
    if src == "p":
        m = Ap["meta"][k]; it = Ap["items"][k]; lab = it.get("id", ("gen", k))
    else:
        m = An["meta"][k]; it = An["items"][k]; lab = ("new", k)
    return lab, m["tokens"][max(0, t-3):t]
z = s - 0.4 * b
tgt = np.isin(kind, ["DIdec_err", "DIaux_err", "DIaux"]) & (gap > 0)
bad = np.isin(kind, ["DDdec", "DDaux", "crit_neg", "GENaux"]) & (gap > 0)
print("BAD positions with z>0 (would fire):")
for j in np.where(bad & (z > 0))[0]:
    print("  ", kind[j], info(j), "s", round(s[j], 2), "gap", round(float(gap[j]), 2))
print("Missed DI targets (z<=0):")
for j in np.where(tgt & (z <= 0))[0]:
    print("  ", kind[j], info(j), "s", round(s[j], 2), "gap", round(float(gap[j]), 2))
print("bad near-threshold (0>=z>-0.15):")
for j in np.where(bad & (z <= 0) & (z > -0.15))[0]:
    print("  ", kind[j], info(j), "s", round(s[j], 2), "gap", round(float(gap[j]), 2))
