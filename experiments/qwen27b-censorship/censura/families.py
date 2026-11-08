"""Families of censorship neurons: top-500 neurons by contribution, profile over the 3,452 questions, k-means (k=8);
per family: top entities, language, type/beneficiary of the censored answers. Usage: families.py OUT.json"""
import json, sys, numpy as np, collections
D = "./censura"; A = np.load(f"{D}/cap/acts.npy", mmap_mode="r"); W = np.load(f"{D}/ext/write_score.npy"); diff = np.load(f"{D}/ext/contrib_diff_A.npy")
N, NL, NI = A.shape; B = [json.loads(l) for l in open(f"{D}/battery.jsonl")]
L = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    try:
        for r in map(json.loads, open(f"{D}/{f}")):
            if r.get("j_base"): L[r["id"]] = r["j_base"]
    except FileNotFoundError: pass
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}
o = np.argsort(-diff.ravel())[:500]; top = [divmod(int(f), NI) for f in o]
P = np.zeros((len(top), N), np.float32)
for l in sorted({l for l, _ in top}):
    js = [(k, j) for k, (ll, j) in enumerate(top) if ll == l]
    a = A[:, l, [j for _, j in js]].astype(np.float32)
    for c, (k, j) in enumerate(js): P[k] = a[:, c] * W[l, j]
Z = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-6)
rng = np.random.default_rng(0); K = 8; C = Z[rng.choice(len(Z), K, replace=False)]
for _ in range(50):
    lab = np.argmax(Z @ C.T / np.linalg.norm(C, axis=1), 1)
    C = np.stack([Z[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
out = []
for k in range(K):
    m = lab == k
    if not m.any(): continue
    prof = P[m].sum(0); hot = np.argsort(-prof)[:60]
    ents = collections.Counter(B[i]["ent"] for i in hot).most_common(15)
    langs = collections.Counter(B[i]["lang"] for i in hot)
    cen = [L[B[i]["id"]] for i in hot if B[i]["id"] in L and L[B[i]["id"]]["type"] in CENS]
    fam = {"k": k, "n_neurons": int(m.sum()), "layers": sorted(collections.Counter(top[i][0] for i in np.where(m)[0]).most_common(6)),
           "hot_entities": ents, "langs": dict(langs), "cens_types": collections.Counter(c["type"] for c in cen).most_common(),
           "beneficiaries": collections.Counter(c["beneficiary"] for c in cen).most_common(8), "neurons": [list(top[i]) for i in np.where(m)[0][:30]]}
    out.append(fam)
    print(f"== famiglia {k}: {fam['n_neurons']} neuroni, layer {fam['layers']} | lingue {fam['langs']}\n   entità: {ents}\n   censure: {fam['cens_types']} | beneficiari {fam['beneficiaries']}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
