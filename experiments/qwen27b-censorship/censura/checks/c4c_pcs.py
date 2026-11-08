"""C4 supplement 2: the comparison in several spaces (raw, raw minus the first m PCs, whitened on the first k PCs),
with a positive control, cosine nulls and C4b per space. Layer 49. Usage: c4c_pcs.py OUT.json"""
import json, sys, numpy as np
D = "./censura"; L = 49; R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r["j_base"]
src = open(f"{D}/extract_foreign.py").read().splitlines(); bt = {r["id"]: r for r in Bt}
exec([l for l in src if l.startswith("CN = ")][0]); exec([l for l in src if l.startswith("china = lambda")][0])
TOP = set(l.strip() for l in open(f"{D}/checks/c4_topics.txt") if l.strip() and not l.startswith(("#", "[")))
def auc(sp, sn):
    s = np.r_[sp, sn]; t = np.r_[np.ones(len(sp)), np.zeros(len(sn))]; o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = len(sp); n0 = len(sn)
    return float((r[t == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
un = lambda v: v / np.linalg.norm(v)
lab = np.array(sorted(J)); y = np.array([int(J[i]["type"] in CENS) for i in lab]); ben = np.array([J[i].get("beneficiary") for i in lab])
ch = np.array([china(i) for i in lab]); deli = np.array([bt[i]["ent"] in TOP and J[i]["type"] == "balanced" for i in lab]); rows = np.array([pos[i] for i in lab])
ent = np.array([bt[i]["ent"] for i in lab]); ue = sorted(set(ent)); rng0 = np.random.default_rng(0); rng0.shuffle(ue); half = set(ue[: len(ue) // 2]); fold = np.array([e in half for e in ent])
P = (y == 1) & (ben == "PRC"); F = ~ch & (y == 1) & (ben != "PRC"); NB = (y == 0) & ~deli
Xa = R[:, L].astype(np.float64); mu = Xa.mean(0); U, S, Vt = np.linalg.svd(Xa - mu, full_matrices=False)
spaces = {"raw": lambda Z: Z}
for m in (1, 3, 10, 30): spaces[f"minus_top{m}"] = (lambda m: lambda Z: (Z - mu) - ((Z - mu) @ Vt[:m].T) @ Vt[:m])(m)
for k in (32, 128, 512): spaces[f"whiten_top{k}"] = (lambda k: lambda Z: ((Z - mu) @ Vt[:k].T) / (S[:k] / np.sqrt(len(Xa))))(k)
out = {}
for name, f in spaces.items():
    X = f(Xa[rows]); r = {}
    a = []
    for fo in (True, False):
        tr, te = fold == fo, fold != fo; d = X[tr & P].mean(0) - X[tr & NB].mean(0); a.append(auc(X[te & P] @ d, X[te & (y == 0)] @ d))
    r["pos_ctrl_cv_auc_PRC"] = float(np.mean(a))
    dP = X[P].mean(0) - X[NB].mean(0); dE = X[F].mean(0) - X[~ch & NB].mean(0); r["cos"] = float(un(dP) @ un(dE))
    rng = np.random.default_rng(1); nl = []
    for _ in range(200):
        g = rng.permutation(len(lab)); nl.append(abs(float(un(X[g[:71]].mean(0) - X[g[71:]].mean(0)) @ un(dP))))
    r["null_p99"] = float(np.percentile(nl, 99))
    r["C4b_dPRC_foreign_vs_delicate"] = auc(X[F] @ dP, X[deli] @ dP); r["dEST_on_PRCcens_vs_CNbal"] = auc(X[P] @ dE, X[ch & (y == 0)] @ dE)
    r["dEST_foreign_vs_delicate_insample"] = auc(X[F] @ dE, X[deli] @ dE)
    out[name] = r; print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
out["var_explained_top"] = (S[:10] ** 2 / (S ** 2).sum()).round(4).tolist(); print("varianza top10", out["var_explained_top"])
json.dump(out, open(sys.argv[1], "w"), indent=1)
