"""Extraction of censorship from weights and activations (CPU). Inputs: captured activations (capture.py), judge labels, battery.
1. labels: censored = {official_script, denial, sovereignty_claim, praise, refusal}; balanced = balanced.
2. per-layer direction d_L = mean residual(censored) - mean(balanced); AUC with entity-level 2-fold validation.
3. from the weights: write score w = down_col . d; per-question contribution a*w; censorship neurons selected on half A, checked on half B.
4. from neurons to topics: per-question score from the top-K neurons (no answers used), AUC on half B, ranking of all 3,452 questions.
Usage: extract.py OUT_DIR [L_STAR]"""
import json, sys, os, glob, numpy as np, torch
from safetensors import safe_open
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); D = "./censura"; M = "models/Qwen3.8-27B"
A = np.load(f"{D}/cap/acts.npy", mmap_mode="r"); R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); N, NL, NI = A.shape; H = R.shape[2]
Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
J = {r["id"]: r for r in map(json.loads, open(f"{D}/judged.jsonl")) if r.get("j_base")}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}
lab = {i: (1 if r["j_base"]["type"] in CENS else 0) for i, r in J.items() if r["j_base"]["type"] in CENS | {"balanced"}}
ids = np.array(sorted(lab)); y = np.array([lab[i] for i in ids]); rows = np.array([pos[i] for i in ids])
ents = sorted({Bt[p]["ent"] for p in rows}); rng = np.random.default_rng(0); rng.shuffle(ents); half = set(ents[: len(ents) // 2])
fold = np.array([0 if Bt[p]["ent"] in half else 1 for p in rows])
print(f"etichette: {len(y)} (censurate {y.sum()}, equilibrate {(1-y).sum()}); fold A {np.sum(fold==0)} B {np.sum(fold==1)}", flush=True)
def auc(s, t):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
res = {"layers": []}
for L in range(NL):
    X = R[rows, L].astype(np.float32); a = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0); a.append(auc(X[te] @ d, y[te]))
    res["layers"].append({"L": L, "auc_cv": float(np.mean(a))})
best = max(res["layers"], key=lambda r: r["auc_cv"]); LS = int(sys.argv[2]) if len(sys.argv) > 2 else best["L"]
print("AUC per layer (ogni 4):", [round(r["auc_cv"], 3) for r in res["layers"][::4]], "migliore", best, "uso L*", LS, flush=True)
X = R[rows, LS].astype(np.float32); trA = fold == 0
d = X[trA & (y == 1)].mean(0) - X[trA & (y == 0)].mean(0); d /= np.linalg.norm(d); np.save(f"{OUT}/dir_L{LS}.npy", d)
where = {}
for f in sorted(glob.glob(f"{M}/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys(): where[k] = f
W = np.zeros((NL, NI), np.float32)
for l in range(LS + 1):
    with safe_open(where[f"model.language_model.layers.{l}.mlp.down_proj.weight"], "pt") as h: Dw = h.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight").float().numpy()
    W[l] = d @ Dw
np.save(f"{OUT}/write_score.npy", W)
diff = np.zeros((NL, NI), np.float64); diffB = np.zeros((NL, NI), np.float64)
for l in range(LS + 1):
    C = A[rows, l].astype(np.float32) * W[l][None]
    diff[l] = C[trA & (y == 1)].mean(0) - C[trA & (y == 0)].mean(0); diffB[l] = C[~trA & (y == 1)].mean(0) - C[~trA & (y == 0)].mean(0)
np.save(f"{OUT}/contrib_diff_A.npy", diff.astype(np.float32)); np.save(f"{OUT}/contrib_diff_B.npy", diffB.astype(np.float32))
flat = diff.ravel(); order = np.argsort(-flat)
top = [divmod(int(f), NI) for f in order[:500]]
print("replica A→B del contributo (Spearman sui top-500 di A):", round(float(np.corrcoef(np.argsort(np.argsort(flat[order[:500]])), np.argsort(np.argsort(diffB.ravel()[order[:500]])))[0, 1]), 3), flush=True)
known = {}
try:
    V = json.load(open("./taiwan/unblock_variants.json"))["off32"]["neurons"]; known["taiwan_off32"] = {(int(l), j) for l, js in V.items() for j in js}
    T = np.load("./taiwan/dec_TIAN_27b/attr_TIAN.npy"); o = np.argsort(T.ravel())[:32]; known["tiananmen_32"] = {divmod(int(f), NI) for f in o}
except Exception as e: print("noti:", e)
rank = {k: i for i, k in enumerate(divmod(int(f), NI) for f in order[:20000])}
for k, s in known.items(): print(f"neuroni noti {k}: nei top-500 estratti {sum(rank.get(x, 1e9) < 500 for x in s)}/{len(s)}, nei top-5000 {sum(rank.get(x, 1e9) < 5000 for x in s)}", flush=True)
for K in (50, 200, 1000):
    sel = top[:K]; S = np.zeros(N, np.float32)
    for l in sorted({l for l, _ in sel}):
        js = [j for ll, j in sel if ll == l]; S += (A[:, l, js].astype(np.float32) * W[l][js]).sum(1)
    res[f"auc_B_K{K}"] = auc(S[rows[~trA]], y[~trA]); print(f"K={K}: AUC del punteggio-neuroni sulle etichette della metà B = {res[f'auc_B_K{K}']:.3f}", flush=True)
    if K == 200: np.save(f"{OUT}/score_K200.npy", S)
S = np.load(f"{OUT}/score_K200.npy"); o = np.argsort(-S)
res["top_questions"] = [{"id": Bt[p]["id"], "lang": Bt[p]["lang"], "fam": Bt[p]["fam"], "ent": Bt[p]["ent"], "q": Bt[p]["q"], "score": float(S[p]), "label": lab.get(Bt[p]["id"])} for p in o[:300]]
ent = {}
for p in range(N): ent.setdefault(Bt[p]["ent"], []).append(float(S[p]))
res["entities"] = sorted(((e, float(np.mean(v)), float(np.max(v))) for e, v in ent.items()), key=lambda x: -x[1])
res["top_neurons"] = [{"l": l, "j": j, "diffA": float(diff[l, j]), "diffB": float(diffB[l, j]), "write": float(W[l, j])} for l, j in top[:200]]
json.dump(res, open(f"{OUT}/extract.json", "w"), indent=1, ensure_ascii=False)
print("ENTITÀ più censurate secondo i neuroni:", [(e, round(m, 2)) for e, m, _ in res["entities"][:40]], flush=True)
print("ENTITÀ meno:", [(e, round(m, 2)) for e, m, _ in res["entities"][-10:]], flush=True)
