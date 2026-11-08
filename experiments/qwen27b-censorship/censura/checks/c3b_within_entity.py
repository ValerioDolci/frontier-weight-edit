"""C3b (exploratory, not pre-registered): is the decision readable within the same entity and the same question?
(a) within-entity AUC of the score; (b) same question in EN and ZH with different outcomes. Directions trained on other entities."""
import json, sys, numpy as np
D = "./censura"; LS = 49
R = np.load(f"{D}/cap/resid.npy", mmap_mode="r"); Bt = [json.loads(l) for l in open(f"{D}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = int(r["j_base"]["type"] in CENS)
ids = np.array(sorted(J)); y = np.array([J[i] for i in ids]); rows = np.array([pos[i] for i in ids]); ent = np.array([Bt[p]["ent"] for p in rows])
ents = sorted(set(ent)); rng = np.random.default_rng(0); rng.shuffle(ents); half = set(ents[: len(ents) // 2]); fold = np.array([0 if e in half else 1 for e in ent])
X = R[rows, LS].astype(np.float32); s = np.zeros(len(y))
for f in (0, 1):
    tr = fold != f; d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0); s[fold == f] = X[fold == f] @ d
S200 = np.load(f"{D}/ext/score_K200.npy")[rows]
out = {}
for name, sc in (("dcens_L49_crossfit", s), ("neurons_K200", S200)):
    w = []; a = []
    for e in set(ent):
        m = ent == e; yy = y[m]
        if 0 < yy.sum() < len(yy):
            ss = sc[m]; p = ss[yy == 1][:, None] - ss[yy == 0][None]; a.append(((p > 0).mean() + 0.5 * (p == 0).mean())); w.append(p.size)
    out[name] = {"within_entity_auc_weighted": float(np.average(a, weights=w)), "n_entities": len(a), "n_pairs": int(sum(w))}
    lab = dict(zip(ids, range(len(ids)))); wins = tot = 0
    for i in ids:
        if Bt[pos[i]]["lang"] != "en" or (i + 1) not in lab: continue
        a_, b_ = lab[i], lab[i + 1]
        if Bt[pos[i + 1]]["lang"] != "zh" or Bt[pos[i]]["ent"] != Bt[pos[i + 1]]["ent"] or y[a_] == y[b_]: continue
        c, b = (a_, b_) if y[a_] == 1 else (b_, a_); tot += 1; wins += sc[c] > sc[b]
    out[name]["same_question_diff_lang"] = {"pairs": tot, "censored_scores_higher": int(wins)}
print(json.dumps(out, indent=1)); json.dump(out, open(f"{D}/checks/c3b.json", "w"), indent=1)
