"""E17: yield of the neuron probe (stage-4 high set) vs the Gemma sensitivity prior (top-N topics by rating),
same population and judge, with and without false premises. Usage: e17_analyze.py"""
import json, math, re
CENS = {'official_script', 'denial', 'sovereignty_claim', 'praise', 'refusal'}
D = "./censura"
import sys
PF = sys.argv[1] if len(sys.argv) > 1 else "results/gpu/e17/e17_prior_answers_160.json"; CF = sys.argv[2] if len(sys.argv) > 2 else "results/judge_e17b_cache.json"
P = json.load(open(PF)); C = json.load(open(CF))
R = json.load(open(f"{D}/rare/rare_judged.json")); alto = [r for r in R if r["set"] == "alto"]
FAKE = re.compile(r"doping|^(?!Mengniu).*melamine contamination", re.I)
def rate(rows, cens_of):
    n = len(rows); c = sum(cens_of(r) for r in rows); return c, n
def fisher_one_sided(a, n1, b, n2):
    N = n1 + n2; K = a + b; p = 0.0
    for x in range(a, min(K, n1) + 1): p += math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)
    return p
probe_c = lambda r: (r.get("j") or {}).get("type") in CENS
prior_c = lambda i: C.get(f"e17|{i}", {}).get("type") in CENS
pr = [(i, r) for i, r in enumerate(P["rows"])]
out = {}
a, n1 = sum(probe_c(r) for r in alto), len(alto); b, n2 = sum(prior_c(i) for i, _ in pr), len(pr)
out["all"] = {"probe": [a, n1, round(a / n1, 3)], "prior": [b, n2, round(b / n2, 3)], "p_probe_gt_prior": fisher_one_sided(a, n1, b, n2)}
af = [r for r in alto if not FAKE.search(r["ent"])]; pf = [(i, r) for i, r in pr if not FAKE.search(r["ent"])]
a2 = sum(probe_c(r) for r in af); b2 = sum(prior_c(i) for i, _ in pf)
out["no_false_premise"] = {"probe": [a2, len(af), round(a2 / len(af), 3)], "prior": [b2, len(pf), round(b2 / len(pf), 3)], "p_probe_gt_prior": fisher_one_sided(a2, len(af), b2, len(pf))}
pe = {r["ent"] for r in alto}; qe = set(P["top_entities"]); out["entity_overlap"] = [len(pe & qe), len(pe), len(qe)]
cp = {r["ent"] for r in alto if probe_c(r) and not FAKE.search(r["ent"])}; cq = {r["ent"] for i, r in pr if prior_c(i) and not FAKE.search(r["ent"])}
out["censored_entities"] = {"probe_only": sorted(cp - cq), "prior_only": sorted(cq - cp), "both": sorted(cp & cq)}
out["files"] = [PF, CF]; json.dump(out, open("results/e17_summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(out, indent=1, ensure_ascii=False))
