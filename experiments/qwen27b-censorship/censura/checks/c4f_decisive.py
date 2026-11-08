"""C4 decisive test (exploratory, suggested by the adversarial review).
(A) PRC direction from Chinese entities only, stratified AUC on non-Chinese entities, permutation null within strata.
(B) Per-neuron Cohen's d for the PRC and foreign contrasts; overlap of the top-100 against random-label and ZH-vs-EN nulls.
Usage: c4f_decisive.py OUT.json"""
import json, sys, numpy as np
D = "./censura"; R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); A = np.load(f"{D}/cap/acts.npy", mmap_mode="r")
Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r["j_base"]
src = open(f"{D}/extract_foreign.py").read().splitlines(); bt = {r["id"]: r for r in Bt}
exec([l for l in src if l.startswith("CN = ")][0]); exec([l for l in src if l.startswith("china = lambda")][0])
TOP = set(l.strip() for l in open(f"{D}/checks/c4_topics.txt") if l.strip() and not l.startswith(("#", "[")))
lab = np.array(sorted(J)); y = np.array([int(J[i]["type"] in CENS) for i in lab]); ben = np.array([str(J[i].get("beneficiary")) for i in lab]); rows = np.array([pos[i] for i in lab])
ch = np.array([china(i) for i in lab]); ent = np.array([bt[i]["ent"] for i in lab]); lg = np.array([bt[i]["lang"] for i in lab])
deli = np.array([e in TOP for e in ent])
Pm = ch & (y == 1) & (ben == "PRC"); CB = ch & (y == 0)
Fm = ~ch & (y == 1) & ~np.isin(ben, ["PRC", "None", "none"]); NB = ~ch & (y == 0)
out = {"n": {"PRC": int(Pm.sum()), "CNbal": int(CB.sum()), "foreign": int(Fm.sum()), "nonCNbal": int(NB.sum())}}
strata = np.array([f"{e}|{l}" for e, l in zip(ent, lg)])
ev = Fm | NB
ok = np.array([ev[i] and Fm[strata == strata[i]].any() and NB[strata == strata[i]].any() for i in range(len(lab))])
out["eval_strata"] = len(set(strata[ok])); out["eval_n"] = {"foreign": int((ok & Fm).sum()), "bal": int((ok & NB).sum())}
def strat_auc(s, yy, st):
    num = den = 0.0
    for g in np.unique(st):
        m = st == g; p = s[m & (yy == 1)]; n = s[m & (yy == 0)]
        if len(p) and len(n): d = p[:, None] - n[None]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den
rng = np.random.default_rng(0)
for L in (33, 49):
    Xa = R[:, L].astype(np.float64); dl = Xa[np.array([r["lang"] for r in Bt]) == "zh"].mean(0) - Xa[np.array([r["lang"] for r in Bt]) == "en"].mean(0); dl /= np.linalg.norm(dl)
    X0 = Xa[rows]
    for nm, X in (("raw", X0), ("lang_out", X0 - np.outer(X0 @ dl, dl))):
        dP = X[Pm].mean(0) - X[CB].mean(0); dD = X[NB & deli].mean(0) - X[NB & ~deli].mean(0)
        yy = Fm[ok].astype(int); st = strata[ok]; r = {}
        for dn, d in (("dPRC", dP), ("dDelicate", dD)):
            s = X[ok] @ d; a = strat_auc(s, yy, st); nl = []
            for _ in range(1000):
                yp = yy.copy()
                for g in np.unique(st):
                    m = np.where(st == g)[0]; yp[m] = rng.permutation(yp[m])
                nl.append(strat_auc(s, yp, st))
            r[dn] = {"auc": float(a), "null_p95": float(np.percentile(nl, 95)), "null_p99": float(np.percentile(nl, 99)), "p": float(np.mean(np.array(nl) >= a))}
        out[f"A_L{L}_{nm}"] = r; print(L, nm, r, flush=True)
NLy, NI = A.shape[1], A.shape[2]
def cohen(mask1, mask0):
    d = np.zeros((NLy, NI), np.float32)
    for l in range(NLy):
        X = A[rows, l].astype(np.float32); a, b = X[mask1], X[mask0]
        sp = np.sqrt((a.var(0) * (len(a) - 1) + b.var(0) * (len(b) - 1)) / (len(a) + len(b) - 2)) + 1e-6; d[l] = (a.mean(0) - b.mean(0)) / sp
    return d
def topk(d, k=100): return set(np.argsort(-np.abs(d).ravel())[:k].tolist())
print("cohen PRC/estero...", flush=True)
dPRC = cohen(Pm, CB); dEST = cohen(Fm, NB); tP, tE = topk(dPRC), topk(dEST)
out["B_overlap_PRC_EST_top100"] = len(tP & tE)
zh = lg == "zh"; out["B_overlap_PRC_ZHvsEN_top100"] = len(tP & topk(cohen(zh, ~zh)))
nl = []
for it in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20):
    ic = np.where(Pm | CB)[0]; p1 = np.zeros(len(lab), bool); p1[rng.choice(ic, Pm.sum(), replace=False)] = True
    inc = np.where(Fm | NB)[0]; p2 = np.zeros(len(lab), bool); p2[rng.choice(inc, Fm.sum(), replace=False)] = True
    nl.append(len(topk(cohen(p1, (Pm | CB) & ~p1)) & topk(cohen(p2, (Fm | NB) & ~p2)))); print("null", it, nl[-1], flush=True)
out["B_null_random_overlap"] = {"mean": float(np.mean(nl)), "p95": float(np.percentile(nl, 95)), "max": int(np.max(nl)), "n": len(nl)}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
