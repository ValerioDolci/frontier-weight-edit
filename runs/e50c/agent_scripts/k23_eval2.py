import numpy as np, json
pairs_r1 = json.load(open(f"{WORK}/pairs_r1.json"))
ID2R2 = {}
for k, it in enumerate(I2):
    if it.get("id") is not None and k < 509:
        ID2R2.setdefault(it["id"], k)
DDIDS = [9, 18, 26, 27, 45, 97, 107, 130, 141, 142, 144, 196, 235, 252, 262, 326, 340, 341, 355, 454, 479, 506, 550, 564, 566, 595, 606, 4, 6, 100, 106, 117, 191, 353, 489, 549, 541, 24, 360, 327, 347]
IMPMODEL = {9, 97, 107, 144, 196, 252, 340, 550, 566}
def eval2(name, verbose=False, show_dd=8):
    e = load_acts(name); Me = e["meta"]; Ie = e["items"]
    DI = []
    di = [p for p in pairs_r1 if p["kind"] == "DI"]
    for k, p in enumerate(di):
        a, b, t = k, 55 + k, p["t"]
        new = Me[b]["logprobs"][t] - Me[a]["logprobs"][t]
        DI.append(dict(id=p["id"], old=p["lp_imp"] - p["lp_trap"], new=new, fold=ID2R2[p["id"]] % 2))
    ddp = {p["id"]: p for p in pairs_r1 if p["kind"] == "DD"}
    DD = []
    for k, iid in enumerate(DDIDS):
        p = ddp[iid]; t = p["t"]; m, al = 110 + k, 151 + k
        trap, imp = (al, m) if iid in IMPMODEL else (m, al)
        DD.append(dict(id=iid, old=p["lp_imp"] - p["lp_trap"], new=Me[imp]["logprobs"][t] - Me[trap]["logprobs"][t], fold=ID2R2[iid] % 2))
    dmax, dsum = [], []
    for k in range(195):
        d = np.array(Me[192 + k]["logprobs"]) - np.array(M2[509 + k]["logprobs"])
        dmax.append(np.abs(d).max()); dsum.append(d.sum())
    dmax = np.array(dmax); dsum = np.array(dsum)
    for f in [0, 1]:
        o = np.array([r["old"] for r in DI if r["fold"] == f]); n = np.array([r["new"] for r in DI if r["fold"] == f])
        print(f"[{name}] DI fold{f} ({'train' if f==0 else 'HELD-OUT'}): n={len(o)} margin>0 old {np.sum(o>0)} new {np.sum(n>0)}; mean shift {np.mean(n-o):.2f} median {np.median(n-o):.2f}")
    for f in [0, 1]:
        o = np.array([r["old"] for r in DD if r["fold"] == f]); n = np.array([r["new"] for r in DD if r["fold"] == f])
        print(f"[{name}] DD fold{f}: n={len(o)} imp-preferred old {np.sum(o>0)} new {np.sum(n>0)}; mean shift {np.mean(n-o):.2f}; max shift {np.max(n-o):.2f}")
    big = sorted(DD, key=lambda r: -(r["new"] - r["old"]))[:show_dd]
    print("   DD biggest shifts:", [(r["id"], round(r["old"],2), round(r["new"],2)) for r in big])
    print(f"[{name}] GEN: items with max|dlogp|>0.1: {np.sum(dmax>0.1)}/195, >0.5: {np.sum(dmax>0.5)}, >2: {np.sum(dmax>2)}; sum dlogp {dsum.sum():.2f}; worst:",
          [(int(i), round(float(dsum[i]),2), round(float(dmax[i]),2)) for i in np.argsort(-dmax)[:8]])
    if verbose:
        for r in sorted(DI, key=lambda r: r["new"]):
            print(f"   DI {r['id']:4d} f{r['fold']} {r['old']:7.2f} -> {r['new']:7.2f}")
    return DI, DD, dmax
DI_e2a, DD_e2a, G_e2a = eval2("e1", verbose=True)
