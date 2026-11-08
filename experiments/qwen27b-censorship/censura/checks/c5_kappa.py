"""C5: agreement between the original judge (Sonnet) and a judge from another family (Gemma 4 31B) on the 1,210 labels;
Cohen's kappa on censored vs balanced, overall and on the foreign subset. Usage: c5_kappa.py GEMMA_CACHE OUT.json"""
import json, sys, collections
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; D = "./censura"
J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r
G = json.load(open(sys.argv[1]))
def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n; pa = sum(a for a, _ in pairs) / n; pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb); return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po, n
out = {}; miss = 0; conf = collections.Counter()
sets = {"all": [], "prc": [], "foreign": [], "zh": [], "en": []}
for i, r in J.items():
    g = G.get(str(i))
    if not g or "type" not in g: miss += 1; continue
    s = int(r["j_base"]["type"] in CENS); m = int(g["type"] in CENS); conf[(r["j_base"]["type"], g["type"])] += 1
    sets["all"].append((s, m)); sets[r["lang"]].append((s, m))
    if s == 1: sets["prc" if r["j_base"].get("beneficiary") == "PRC" else "foreign"].append((s, m))
for k, p in sets.items():
    if k in ("prc", "foreign"): out[k] = {"n": len(p), "gemma_agrees_censored": sum(m for _, m in p), "rate": sum(m for _, m in p) / max(1, len(p))}
    else: kk, po, n = kappa(p); out[k] = {"kappa": kk, "agree": po, "n": n}
bt = {r["id"]: r for r in map(json.loads, open(f"{D}/battery.jsonl"))}; src = open(f"{D}/extract_foreign.py").read().splitlines()
exec([l for l in src if l.startswith("CN = ")][0]); exec([l for l in src if l.startswith("china = lambda")][0])
fp = [(int(r["j_base"]["type"] in CENS), int(G[str(i)]["type"] in CENS)) for i, r in J.items() if str(i) in G and "type" in G[str(i)] and not china(i)]
kk, po, n = kappa(fp); out["foreign_subset_kappa"] = {"kappa": kk, "agree": po, "n": n, "def": "entità NON cinesi (stessa definizione di extract_foreign.py)"}
out["missing"] = miss; out["confusion_top"] = [[f"{a}->{b}", c] for (a, b), c in conf.most_common(15)]
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out, indent=1))
