"""C4 supplement: raw cosines between difference-of-means directions are dominated by a few high-variance dimensions,
so the comparison is redone in whitened space with nulls. Usage: c4b_whitened.py OUT.json"""
import json, sys, numpy as np
D = "./censura"; R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r["j_base"]
src = open(f"{D}/extract_foreign.py").read().splitlines(); bt = {r["id"]: r for r in Bt}
exec([l for l in src if l.startswith("CN = ")][0]); exec([l for l in src if l.startswith("china = lambda")][0])
TOP = set(l.strip() for l in open(f"{D}/checks/c4_topics.txt") if l.strip() and not l.startswith(("#", "[")))
def auc(s, t):
    t = np.asarray(t); o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
un = lambda v: v / np.linalg.norm(v)
lab = np.array(sorted(J)); y = np.array([int(J[i]["type"] in CENS) for i in lab]); ben = np.array([J[i].get("beneficiary") for i in lab])
ch = np.array([china(i) for i in lab]); deli = np.array([bt[i]["ent"] in TOP and J[i]["type"] == "balanced" for i in lab]); rows = np.array([pos[i] for i in lab])
out = {}
for L in (33, 49, 52):
    Xa = R[:, L].astype(np.float64); mu = Xa.mean(0); U, S, Vt = np.linalg.svd(Xa - mu, full_matrices=False); keep = S > 1e-3 * S[0]
    Wm = Vt[keep].T / (S[keep] / np.sqrt(len(Xa)))
    def dirs(Z):
        X = Z[rows]
        dP = X[(y == 1) & (ben == "PRC")].mean(0) - X[(y == 0) & ~deli].mean(0)
        dE = X[~ch & (y == 1) & (ben != "PRC")].mean(0) - X[~ch & (y == 0) & ~deli].mean(0); return X, dP, dE
    res = {}
    for space, Z in (("raw", Xa), ("whitened", (Xa - mu) @ Wm)):
        X, dP, dE = dirs(Z); r = {"cos_dPRC_dEST": float(un(dP) @ un(dE))}
        rng = np.random.default_rng(0); nl = []; nl2 = []
        for _ in range(200):
            g = rng.permutation(len(lab)); a, b = g[:71], g[71:]
            dr = X[a].mean(0) - X[b].mean(0); nl.append(abs(float(un(dr) @ un(dP))))
            nb = np.where(~ch & (y == 0) & ~deli)[0]; p = rng.permutation(nb); d2 = X[p[:71]].mean(0) - X[p[71:]].mean(0); nl2.append(float(un(d2) @ un(dP)))
        r["null_random71_abs_p99"] = float(np.percentile(nl, 99)); r["null_balanced71_p99"] = float(np.percentile(nl2, 99)); r["null_balanced71_median"] = float(np.median(nl2))
        sp = X[~ch & (y == 1) & (ben != "PRC")] @ dP; sn = X[deli] @ dP
        r["C4b_auc_dPRCclean_foreign_vs_delicate"] = auc(np.r_[sp, sn], np.r_[np.ones(len(sp)), np.zeros(len(sn))])
        sp = X[(y == 1) & (ben == "PRC")] @ dE; sn = X[ch & (y == 0)] @ dE
        r["auc_dEST_on_PRC_cens_vs_china_balanced"] = auc(np.r_[sp, sn], np.r_[np.ones(len(sp)), np.zeros(len(sn))])
        res[space] = r
    out[f"L{L}"] = res; print(L, json.dumps(res), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
