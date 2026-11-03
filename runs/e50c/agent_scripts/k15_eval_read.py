import numpy as np, json, sys
def eval_read(name, verbose=True):
    e = load_acts(name); Me = e["meta"]; Ie = e["items"]
    pairs = json.load(open(f"{WORK}/pairs_r1.json"))
    r1m = load_acts("r1")["meta"]
    DDIDS = [9, 18, 26, 27, 45, 97, 107, 130, 141, 142, 144, 196, 235, 252, 262, 326, 340, 341, 355, 454, 479, 506, 550, 564, 566, 595, 606, 4, 6, 100, 106, 117, 191, 353, 489, 549, 541, 24, 360, 327, 347]
    IMPMODEL = {9, 97, 107, 144, 196, 252, 340, 550, 566}
    out = {"DI": [], "DD": []}
    di = [p for p in pairs if p["kind"] == "DI"]
    for k, p in enumerate(di):
        a, b, t = k, 55 + k, p["t"]
        assert Ie[a]["id"] == p["id"] and Ie[b]["id"] == p["id"]
        new = Me[b]["logprobs"][t] - Me[a]["logprobs"][t]
        old = p["lp_imp"] - p["lp_trap"]
        out["DI"].append((p["id"], old, new, Me[a]["logprobs"][t], Me[b]["logprobs"][t]))
    ddp = {p["id"]: p for p in pairs if p["kind"] == "DD"}
    for k, iid in enumerate(DDIDS):
        p = ddp[iid]; t = p["t"]
        m, al = 110 + k, 151 + k
        trap, imp = (al, m) if iid in IMPMODEL else (m, al)
        new = Me[imp]["logprobs"][t] - Me[trap]["logprobs"][t]
        old = p["lp_imp"] - p["lp_trap"]
        out["DD"].append((iid, old, new, Me[trap]["logprobs"][t], Me[imp]["logprobs"][t]))
    # general: compare with r2 meta (509..703)
    r2m = load_acts("r2")["meta"]
    dsum, dmax, nchg = [], [], 0
    for k in range(195):
        lo = np.array(r2m[509 + k]["logprobs"]); ln = np.array(Me[192 + k]["logprobs"])
        d = ln - lo
        dsum.append(d.sum()); dmax.append(np.abs(d).max())
    dsum = np.array(dsum); dmax = np.array(dmax)
    if verbose:
        print("DI (id, old margin, new margin, new lp_trap, new lp_imp):")
        for r in sorted(out["DI"], key=lambda z: z[2]):
            print(f"   {r[0]:4d} {r[1]:7.2f} -> {r[2]:7.2f}   (trap {r[3]:6.2f}, imp {r[4]:6.2f})")
        print("DD (id, old margin imp-trap, new):")
        for r in sorted(out["DD"], key=lambda z: -z[2]):
            print(f"   {r[0]:4d} {r[1]:7.2f} -> {r[2]:7.2f}")
    di_old = np.array([r[1] for r in out["DI"]]); di_new = np.array([r[2] for r in out["DI"]])
    dd_old = np.array([r[1] for r in out["DD"]]); dd_new = np.array([r[2] for r in out["DD"]])
    print(f"[{name}] DI: margin>0 old {np.sum(di_old>0)}/55 new {np.sum(di_new>0)}/55; mean shift {np.mean(di_new-di_old):.2f}, median shift {np.median(di_new-di_old):.2f}")
    print(f"[{name}] DD: imp-preferred (margin>0) old {np.sum(dd_old>0)}/41 new {np.sum(dd_new>0)}/41; mean shift {np.mean(dd_new-dd_old):.2f}")
    print(f"[{name}] GEN: items with any |dlogp|>0.05: {np.sum(dmax>0.05)}/195, >0.5: {np.sum(dmax>0.5)}; total sum dlogp {dsum.sum():.2f}; worst items:",
          [(int(i), round(float(dsum[i]),2), round(float(dmax[i]),2)) for i in np.argsort(-dmax)[:6]])
    return out
res_e1 = eval_read("e1")
