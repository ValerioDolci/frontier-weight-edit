"""E15: is the C3 decision direction (layer 33) causal? bf16, hybrid. Pre-registered in PREREG_CHECKS.md.
Usage: e15_steer.py OUT.json"""
import json, sys, os, random
sys.argv_ = sys.argv
exec(open("./taiwan/unblock_eval.py").read().split("# 1. battery")[0])
import numpy as np, torch
W = "./censura"; L = 33; OUT = sys.argv_[1]; SM = os.environ.get("SMOKE") == "1"
CENS = {"official_script", "denial", "sovereignty_claim", "praise", "refusal"}; J = {}
for f in ("judged.jsonl", "judged_neuron.jsonl"):
    for r in map(json.loads, open(f"{W}/{f}")):
        if r.get("j_base") and r["j_base"]["type"] in CENS | {"balanced"}: J[r["id"]] = r
Bt = [json.loads(l) for l in open(f"{W}/battery.jsonl")]; pos = {r["id"]: i for i, r in enumerate(Bt)}
S = np.load(f"{W}/ext/score_K200.npy"); thr = 12.80551528930664
sub = [i for i in J if S[pos[i]] >= thr]; cen = sorted(i for i in sub if J[i]["j_base"]["type"] in CENS); bal = sorted(i for i in sub if J[i]["j_base"]["type"] == "balanced")
rng = random.Random(int(os.environ.get("E15SEED", "0"))); rng.shuffle(cen); rng.shuffle(bal); cen, bal = cen[:20], bal[:20]
if SM: cen, bal = cen[:2], bal[:2]
R = np.load(f"{W}/cap/resid.npy", mmap_mode="r"); d = np.load(f"{W}/checks/c2c3_ddecision_L33.npy").astype(np.float32); d /= np.linalg.norm(d)
X = R[[pos[i] for i in sub], L].astype(np.float32); y = np.array([J[i]["j_base"]["type"] in CENS for i in sub])
m_c, m_b = float((X[y] @ d).mean()), float((X[~y] @ d).mean())
g = np.random.default_rng(1).standard_normal(d.shape).astype(np.float32); g /= np.linalg.norm(g)
mr_c, mr_b = float((X[y] @ g).mean()), float((X[~y] @ g).mean())
print("m_cens", m_c, "m_bal", m_b, "| rand", mr_c, mr_b, "| n", len(cen), len(bal), flush=True)
STE = {"mode": None}
dt = {k: torch.tensor(v) for k, v in (("d", d), ("g", g))}
def hook(m, a, o):
    if STE["mode"] is None: return None
    h = o[0] if isinstance(o, tuple) else o; kind, op = STE["mode"]; v = dt[kind].to(h.device, h.dtype); mc, mb = (m_c, m_b) if kind == "d" else (mr_c, mr_b)
    if op == "ablate": h = h - ((h @ v) - mb).unsqueeze(-1) * v
    elif op == "add_eq": h = h + (m_c - m_b) * v
    else: h = h + (mc - mb) * v
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
LAY[L].register_forward_hook(hook)
Q = {i: J[i]["q"] for i in cen + bal}; out = {"cen": cen, "bal": bal, "m": [m_c, m_b], "q": Q, "ans": {}}
RUNS = (("base_cen", cen, None), ("base_bal", bal, None), ("ablate_d", cen, ("d", "ablate")), ("ablate_rand", cen, ("g", "ablate")),
                        ("add_d", bal, ("d", "add")), ("add_rand", bal, ("g", "add")))
if os.environ.get("E15C") == "1": RUNS = (("base_bal", bal, None), ("add_d", bal, ("d", "add")), ("add_rand_eq", bal, ("g", "add_eq")))
if os.environ.get("E15B") == "1": RUNS = (("base_bal_rep", bal, None), ("add_rand_eq", bal, ("g", "add_eq")))
for name, ids, mode in RUNS:
    STE["mode"] = mode; out["ans"][name] = dict(zip(map(str, ids), gen([chat(Q[i]) for i in ids], 96, bs=20)))
    print(name, "fatto", flush=True); json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
STE["mode"] = None
