"""C4 supplement 3: what is PC1? Correlation with template language and question family; C4b within each language
and after removing the language direction. Layer 49. Usage: c4d_pc1.py OUT.json"""
import json, sys, collections, numpy as np
D = "./censura"; L = 49; R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r["j_base"]
src = open(f"{D}/extract_foreign.py").read().splitlines(); bt = {r["id"]: r for r in Bt}
exec([l for l in src if l.startswith("CN = ")][0]); exec([l for l in src if l.startswith("china = lambda")][0])
TOP = set(l.strip() for l in open(f"{D}/checks/c4_topics.txt") if l.strip() and not l.startswith(("#", "[")))
def auc(sp, sn):
    if len(sp) < 3 or len(sn) < 3: return None
    s = np.r_[sp, sn]; t = np.r_[np.ones(len(sp)), np.zeros(len(sn))]; o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = len(sp); n0 = len(sn)
    return float((r[t == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
Xa = R[:, L].astype(np.float64); mu = Xa.mean(0); U, S, Vt = np.linalg.svd(Xa - mu, full_matrices=False); pc = (Xa - mu) @ Vt[:3].T
lang = np.array([r["lang"] for r in Bt]); fam = np.array([r["fam"] for r in Bt])
out = {"pc_lang_auc": [auc(pc[lang == "zh", k], pc[lang == "en", k]) for k in range(3)]}
out["pc1_mean_by_fam"] = sorted(((f, float(pc[fam == f, 0].mean())) for f in set(fam)), key=lambda x: x[1])
lab = np.array(sorted(J)); y = np.array([int(J[i]["type"] in CENS) for i in lab]); ben = np.array([J[i].get("beneficiary") for i in lab]); rows = np.array([pos[i] for i in lab])
ch = np.array([china(i) for i in lab]); deli = np.array([bt[i]["ent"] in TOP and J[i]["type"] == "balanced" for i in lab]); lg = lang[rows]
P = (y == 1) & (ben == "PRC"); F = ~ch & (y == 1) & (ben != "PRC"); NB = (y == 0) & ~deli
out["counts"] = {"foreign_cens": {l: int((F & (lg == l)).sum()) for l in ("en", "zh")}, "delicate_bal": {l: int((deli & (lg == l)).sum()) for l in ("en", "zh")},
                 "PRC_cens": {l: int((P & (lg == l)).sum()) for l in ("en", "zh")}}
X = Xa[rows]; dl = Xa[lang == "zh"].mean(0) - Xa[lang == "en"].mean(0); dl /= np.linalg.norm(dl)
for nm, Z in (("raw", X), ("lang_out", X - np.outer(X @ dl, dl))):
    r = {}
    for l in ("en", "zh"):
        m = lg == l; dP = Z[P & m].mean(0) - Z[NB & m].mean(0)
        r[f"C4b_within_{l}"] = auc(Z[F & m] @ dP, Z[deli & m] @ dP)
        dE = Z[F & m].mean(0) - Z[~ch & NB & m].mean(0); r[f"dEST_on_PRC_within_{l}"] = auc(Z[P & m] @ dE, Z[ch & (y == 0) & m] @ dE)
    out[nm] = r
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1], "w"), indent=1)
