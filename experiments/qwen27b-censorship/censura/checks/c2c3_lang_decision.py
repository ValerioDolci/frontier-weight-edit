"""C2 (language as a confound) and C3 (topic sensitivity vs decision), CPU only, on the captured activations.
Same method as extract.py (difference-of-means direction, entity-level 2-fold validation); labels = judged + judged_neuron (1,210).
Usage: c2c3_lang_decision.py OUT.json"""
import json, sys, numpy as np
D = "./censura"; OUT = sys.argv[1]; LS = 49
A = np.load(f"{D}/cap/acts.npy", mmap_mode="r"); R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); N, NL, NI = A.shape
Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}
J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = int(r["j_base"]["type"] in CENS)
ids = np.array(sorted(J)); y = np.array([J[i] for i in ids]); rows = np.array([pos[i] for i in ids])
lang = np.array([Bt[p]["lang"] for p in rows]); ent = np.array([Bt[p]["ent"] for p in rows])
ents = sorted(set(ent)); rng = np.random.default_rng(0); rng.shuffle(ents); half = set(ents[: len(ents) // 2])
fold = np.array([0 if e in half else 1 for e in ent])
def auc(s, t):
    t = np.asarray(t); o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
def cv_auc(X, yy, ff, mask=None, proj=None):
    m = np.ones(len(yy), bool) if mask is None else mask; a = []
    for f in (0, 1):
        tr, te = m & (ff != f), m & (ff == f)
        if yy[tr].sum() < 3 or (1 - yy[tr]).sum() < 3 or yy[te].sum() < 3: continue
        d = X[tr & (yy == 1)].mean(0) - X[tr & (yy == 0)].mean(0)
        if proj is not None: d = d - proj * (proj @ d)
        a.append(auc(X[te] @ d, yy[te]))
    return float(np.mean(a)) if a else float("nan")
out = {"n_labels": int(len(y)), "n_cens": int(y.sum()), "by_lang": {l: {"n": int((lang == l).sum()), "cens": int(y[lang == l].sum())} for l in ("en", "zh")}}
print(out, flush=True)
c2 = {"layers": []}
for L in range(0, NL, 1):
    X = R[rows, L].astype(np.float32)
    row = {"L": L, "all": cv_auc(X, y, fold), "en": cv_auc(X, y, fold, lang == "en"), "zh": cv_auc(X, y, fold, lang == "zh")}
    for a_, b_ in (("en", "zh"), ("zh", "en")):
        tr, te = lang == a_, lang == b_; d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0); row[f"{a_}->{b_}"] = auc(X[te] @ d, y[te])
    c2["layers"].append(row)
    if L % 8 == 1 or L == LS: print("C2", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
X = R[rows, LS].astype(np.float32); Xall = R[:, LS].astype(np.float32)
langall = np.array([r["lang"] for r in Bt])
dl = Xall[langall == "zh"].mean(0) - Xall[langall == "en"].mean(0); dl_hat = dl / np.linalg.norm(dl)
dc = X[y == 1].mean(0) - X[y == 0].mean(0); dc_hat = dc / np.linalg.norm(dc)
cos = float(dc_hat @ dl_hat)
nul = []
r2 = np.random.default_rng(1)
for _ in range(200):
    g = r2.permutation(N)[:500]; m = np.zeros(N, bool); m[g] = True; dr = Xall[m].mean(0) - Xall[~m].mean(0); nul.append(abs(float(dc_hat @ dr / np.linalg.norm(dr))))
c2[f"L{LS}"] = {"cos_dcens_dlang": cos, "null_abs_cos_p99": float(np.percentile(nul, 99)), "null_abs_cos_median": float(np.median(nul)),
                "auc_lang_only(zh=1)": auc((lang == "zh").astype(float) + 1e-6 * np.arange(len(y)), y),
                "auc_cv_all": cv_auc(X, y, fold), "auc_cv_all_lang_projected_out": cv_auc(X, y, fold, proj=dl_hat),
                "auc_cv_en_lang_projected_out": cv_auc(X, y, fold, lang == "en", proj=dl_hat), "auc_cv_zh_lang_projected_out": cv_auc(X, y, fold, lang == "zh", proj=dl_hat)}
dl2 = X[(y == 0) & (lang == "zh")].mean(0) - X[(y == 0) & (lang == "en")].mean(0); c2[f"L{LS}"]["cos_dcens_dlang_balanced_only"] = float(dc_hat @ dl2 / np.linalg.norm(dl2))
E = json.load(open(f"{D}/ext/extract.json")); top = [(t["l"], t["j"], t["write"]) for t in E["top_neurons"]]
for K in (50, 200):
    S = np.zeros(len(rows), np.float32)
    for l, j, w in top[:K]: S += A[rows, l, j].astype(np.float32) * w
    for lg in ("en", "zh"):
        m = (fold == 1) & (lang == lg); c2[f"L{LS}"][f"neurons_K{K}_auc_B_{lg}"] = auc(S[m], y[m])
    m = fold == 1; c2[f"L{LS}"][f"neurons_K{K}_auc_B_all"] = auc(S[m], y[m])
print("C2 L49", {k: round(v, 3) for k, v in c2[f"L{LS}"].items()}, flush=True)
out["C2"] = c2
S200 = np.load(f"{D}/ext/score_K200.npy")
s_lab = S200[rows]
cand = np.unique(s_lab); best = max(cand, key=lambda t: (s_lab[y == 1] >= t).mean() - (s_lab[y == 0] >= t).mean())
above_all = int((S200 >= best).sum()); sub = s_lab >= best
c3 = {"youden_thr": float(best), "n_above_all_3452": above_all, "n_sub_labeled": int(sub.sum()), "n_sub_cens": int(y[sub].sum()),
      "sub_lang": {lg: {"n": int((sub & (lang == lg)).sum()), "cens": int(y[sub & (lang == lg)].sum())} for lg in ("en", "zh")}}
ysub, fsub, lsub = y[sub], fold[sub], lang[sub]
c3["baseline_auc_lang_only"] = auc((lsub == "zh").astype(float) + 1e-6 * np.arange(len(ysub)), ysub)
c3["baseline_auc_neuron_score_K200"] = auc(s_lab[sub], ysub)
c3["layers"] = []
for L in range(0, NL):
    XL = R[rows[sub], L].astype(np.float32)
    c3["layers"].append({"L": L, "auc_cv": cv_auc(XL, ysub, fsub), "auc_cv_en": cv_auc(XL, ysub, fsub, lsub == "en"), "auc_cv_zh": cv_auc(XL, ysub, fsub, lsub == "zh")})
bestL = max(c3["layers"], key=lambda r: r["auc_cv"] if r["auc_cv"] == r["auc_cv"] else -1); c3["best_layer"] = bestL
Xs = R[rows[sub], bestL["L"]].astype(np.float32)
perm = []; r3 = np.random.default_rng(2)
for _ in range(200):
    yp = ysub.copy()
    r3.shuffle(yp); perm.append(cv_auc(Xs, yp, fsub))
c3["null_perm_auc_p95"] = float(np.nanpercentile(perm, 95)); c3["null_perm_auc_median"] = float(np.nanmedian(perm))
es = ent[sub]; ue = np.unique(es); perm2 = []
for _ in range(200):
    lab_e = {e: int(ysub[es == e].mean() > 0.5) for e in ue}; vals = list(lab_e.values()); r3.shuffle(vals); m = dict(zip(ue, vals))
    perm2.append(cv_auc(Xs, np.array([m[e] for e in es]), fsub))
c3["null_entity_perm_auc_p95"] = float(np.nanpercentile(perm2, 95))
c3["auc_dcens_L49_within_sub"] = auc(R[rows[sub], LS].astype(np.float32) @ dc_hat, ysub)
print("C3", {k: v for k, v in c3.items() if k != "layers"}, flush=True)
print("C3 per layer (ogni 4):", [(r["L"], round(r["auc_cv"], 3)) for r in c3["layers"][::4]], flush=True)
XL = R[rows, bestL["L"]].astype(np.float32); dcL = XL[y == 1].mean(0) - XL[y == 0].mean(0)
d2 = Xs[ysub == 1].mean(0) - Xs[ysub == 0].mean(0); c3["cos_ddecision_dcens_sameL"] = float(d2 @ dcL / np.linalg.norm(d2) / np.linalg.norm(dcL))
np.save(OUT.replace(".json", f"_ddecision_L{bestL['L']}.npy"), d2 / np.linalg.norm(d2))
out["C3"] = c3
json.dump(out, open(OUT, "w"), indent=1)
print("FATTO", flush=True)
