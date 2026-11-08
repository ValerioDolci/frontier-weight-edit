"""C4 "a single Beijing viewpoint" (pre-registered, amended before running): partial cosine between the foreign and PRC
censorship directions with nulls (C4), AUC on foreign censorship (C4b), neutral questions on authoritarian vs democratic countries (C4c).
Usage: c4_beijing.py GPU_OUT_DIR OUT.json"""
import json, sys, numpy as np
D = "./censura"; G, OUT = sys.argv[1], sys.argv[2]
R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r["j_base"]
def auc(s, t):
    t = np.asarray(t); o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
un = lambda v: v / np.linalg.norm(v)
TOP = set(l.strip() for l in open(f"{D}/checks/c4_topics.txt") if l.strip() and not l.startswith(("#", "[")))
deli = [i for i, j in J.items() if Bt[pos[i]]["ent"] in TOP and j["type"] == "balanced"]
deli_all = [r["id"] for r in Bt if r["ent"] in TOP]
src = open(f"{D}/extract_foreign.py").read().splitlines()
bt = {r["id"]: r for r in Bt}
exec([l for l in src if l.startswith("CN = ")][0]); exec([l for l in src if l.startswith("china = lambda")][0])
fore = [i for i, j in J.items() if j["type"] in CENS and j.get("beneficiary") not in ("PRC", None, "none") and not china(i)]
rows = json.load(open(f"{G}/c4_rows.json")); RN = np.load(f"{G}/c4_resid.npy")
grp = np.array([r["group"] for r in rows])
out = {"n_delicate_balanced_labeled": len(deli), "n_delicate_all_battery": len(deli_all), "n_foreign_cens": len(fore),
       "n_neutral": {g: int((grp == g).sum()) for g in ("aut", "dem", "china")}}
print(out, flush=True)
res = {}
for L in (49, 52):
    XN = RN[:, L].astype(np.float32)
    lab = np.array(sorted(J)); y = np.array([int(J[i]["type"] in CENS) for i in lab]); X = R[[pos[i] for i in lab], L].astype(np.float32)
    ch = np.array([china(i) for i in lab]); ben = np.array([J[i].get("beneficiary") for i in lab])
    dPRC = un(X[(y == 1) & (ben == "PRC")].mean(0) - X[y == 0].mean(0))
    fm = ~ch; dEST = un(X[fm & (y == 1) & (ben != "PRC")].mean(0) - X[fm & (y == 0)].mean(0))
    Xd = R[[pos[i] for i in deli], L].astype(np.float32)
    d_del = Xd.mean(0) - XN[grp == "dem"].mean(0); d_aut = XN[grp == "aut"].mean(0) - XN[grp == "dem"].mean(0)
    Q, _ = np.linalg.qr(np.stack([d_del, d_aut], 1)); proj = lambda v: v - Q @ (Q.T @ v)
    r = {"cos_raw": float(dEST @ dPRC), "cos_partial": float(un(proj(dEST)) @ un(proj(dPRC))),
         "cos_dEST_ddel": float(dEST @ un(d_del)), "cos_dEST_daut": float(dEST @ un(d_aut)), "cos_dPRC_ddel": float(dPRC @ un(d_del)), "cos_dPRC_daut": float(dPRC @ un(d_aut))}
    ents = np.array([Bt[pos[i]]["ent"] for i in lab]); ue = np.unique(ents); rng = np.random.default_rng(0); st = []; nl = []
    for _ in range(50):
        h = set(rng.permutation(ue)[: len(ue) // 2]); m = np.array([e in h for e in ents])
        a1 = un(X[m & (y == 1) & (ben == "PRC")].mean(0) - X[m & (y == 0)].mean(0)); a2 = un(X[~m & (y == 1) & (ben == "PRC")].mean(0) - X[~m & (y == 0)].mean(0)); st.append(float(a1 @ a2))
        g = rng.permutation(len(lab))[:70]; mm = np.zeros(len(lab), bool); mm[g] = True; nl.append(float(un(X[mm].mean(0) - X[~mm].mean(0)) @ dPRC))
    r["null_selfcos_dPRC_halves_median"] = float(np.median(st)); r["null_cos_random70group_vs_dPRC_p99"] = float(np.percentile(np.abs(nl), 99))
    dl = set(deli); neg_clean = np.array([(i not in dl) for i in lab])
    dEST_c = un(X[fm & (y == 1) & (ben != "PRC")].mean(0) - X[fm & (y == 0) & neg_clean].mean(0))
    Xf = R[[pos[i] for i in fore], L].astype(np.float32)
    s_pos, s_neg = Xf @ dEST_c, Xd @ dEST_c; r["C4b_auc_dEST_clean_foreign_vs_delicate"] = auc(np.r_[s_pos, s_neg], np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))])
    s_pos, s_neg = Xf @ dPRC, Xd @ dPRC; r["C4b_auc_dPRC_foreign_vs_delicate"] = auc(np.r_[s_pos, s_neg], np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))])
    for nm, d in (("dPRC", dPRC), ("dEST", dEST)):
        s = XN @ d; m = grp != "china"; r[f"C4c_auc_{nm}_aut_vs_dem_neutral"] = auc(s[m], (grp[m] == "aut").astype(int))
        r[f"C4c_{nm}_mean_by_group"] = {g: float(s[grp == g].mean()) for g in ("aut", "dem", "china")}
        r[f"C4c_{nm}_battery_bal_cens_mean"] = [float((X[y == 0] @ d).mean()), float((X[y == 1] @ d).mean())]
    res[f"L{L}"] = r; print(L, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
out["res"] = res; json.dump(out, open(OUT, "w"), indent=1); print("FATTO")
