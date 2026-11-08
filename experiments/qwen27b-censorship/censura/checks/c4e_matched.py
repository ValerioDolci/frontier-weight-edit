"""C4 supplement 4 (exploratory): does the PRC-to-foreign transfer hold at fixed template and language?
PRC direction built on Chinese entities only, evaluated on non-Chinese entities within (template, language); random-group null.
Usage: c4e_matched.py OUT.json"""
import json, sys, numpy as np
D = "./censura"; R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r["j_base"]
src = open(f"{D}/extract_foreign.py").read().splitlines(); bt = {r["id"]: r for r in Bt}
exec([l for l in src if l.startswith("CN = ")][0]); exec([l for l in src if l.startswith("china = lambda")][0])
lab = np.array(sorted(J)); y = np.array([int(J[i]["type"] in CENS) for i in lab]); ben = np.array([J[i].get("beneficiary") for i in lab]); rows = np.array([pos[i] for i in lab])
ch = np.array([china(i) for i in lab]); fam = np.array([bt[i]["fam"] for i in lab]); lg = np.array([bt[i]["lang"] for i in lab])
def strat_auc(s, yy, strata):
    num = den = 0.0
    for st in set(strata):
        m = strata == st; p = s[m & (yy == 1)]; n = s[m & (yy == 0)]
        if len(p) and len(n): d = p[:, None] - n[None]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den if den else None, int(den)
out = {}
for L in (33, 49):
    Xa = R[:, L].astype(np.float64); mu = Xa.mean(0); _, _, Vt = np.linalg.svd(Xa - mu, full_matrices=False)
    for nm, X in (("raw", Xa[rows]), ("minus_pc1", (Xa[rows] - mu) - np.outer((Xa[rows] - mu) @ Vt[0], Vt[0]))):
        dP = X[ch & (y == 1) & (ben == "PRC")].mean(0) - X[ch & (y == 0)].mean(0)
        m = ~ch; yy = y[m] * (ben[m] != "PRC") ; keep = (y[m] == 0) | (ben[m] != "PRC")
        s = X[m][keep] @ dP; strata = np.array([f"{a}|{b}" for a, b in zip(fam[m][keep], lg[m][keep])])
        a, npairs = strat_auc(s, yy[keep], strata)
        rng = np.random.default_rng(0); idx = np.where(ch)[0]; nl = []
        for _ in range(200):
            g = rng.permutation(idx); h = len(g) // 2; dr = X[g[:h]].mean(0) - X[g[h:]].mean(0); nl.append(strat_auc(X[m][keep] @ dr, yy[keep], strata)[0])
        out[f"L{L}_{nm}"] = {"matched_auc": a, "pairs": npairs, "null_p95": float(np.percentile(nl, 95)), "null_p5": float(np.percentile(nl, 5)),
                             "n_foreign_cens": int(yy[keep].sum()), "n_nonchina_bal": int((yy[keep] == 0).sum())}
        print(L, nm, out[f"L{L}_{nm}"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
