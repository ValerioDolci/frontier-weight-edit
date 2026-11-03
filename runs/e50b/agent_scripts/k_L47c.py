import torch, json, numpy as np, time
t0 = time.time()
D0 = torch.load(WORK + "/L47pos.pt"); R47 = D0["R47"]; POS = D0["POS"]
A = load_acts("L47a"); meta, items = A["meta"], A["items"]
nw = W("model.language_model.norm.weight")
E = W("model.language_model.embed_tokens.weight")
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
AUX = [182706, 51542, 139292, 205867, 23120, 23456, 23306, 40877, 51638, 1883]
auxmask = torch.zeros(E.shape[0], dtype=torch.bool); auxmask[AUX] = True
N = len(POS)
maxaux = torch.zeros(N); argaux = torch.zeros(N, dtype=torch.long)
maxnon = torch.zeros(N); argnon = torch.zeros(N, dtype=torch.long); lse = torch.zeros(N)
for s in range(0, N, 1024):
    x = rmsn(R47[s:s+1024].float(), nw)
    lg = 30.0 * torch.tanh((x @ E.T) / 30.0)
    lse[s:s+1024] = torch.logsumexp(lg, 1)
    la = lg[:, AUX]; v, i = la.max(1); maxaux[s:s+1024] = v; argaux[s:s+1024] = torch.tensor(AUX)[i]
    lg[:, auxmask] = -1e9
    v, i = lg.max(1); maxnon[s:s+1024] = v; argnon[s:s+1024] = i
torch.save({"maxaux": maxaux, "argaux": argaux, "maxnon": maxnon, "argnon": argnon, "lse": lse}, WORK + "/L47aux.pt")
print("done", time.time() - t0)
paux = torch.exp(maxaux - lse)
gap = maxaux - maxnon  # >0 means aux is top-1
rows = json.load(open(WORK + "/dec_rows.json"))
pidx = {p: j for j, p in enumerate(POS)}
inv = {v: k for k, v in json.load(open(WORK + "/tokid.json")).items()}
cat = {p["id"]: p["cat"] for p in practice}
print("DI err decision positions: gap (aux - best nonaux) and best nonaux token")
for r in sorted([r for r in rows if r["cat"] == "DI" and not r["ok"]], key=lambda r: -float(gap[pidx[(r['kb'], r['t'])]])):
    j = pidx[(r["kb"], r["t"])]
    print(r["id"], r["t"], repr(r["tokB"]), "gap", round(float(gap[j]), 2), "aux", inv.get(int(argaux[j]), int(argaux[j])), "best non-aux:", repr(inv.get(int(argnon[j]), int(argnon[j]))), "p_aux", round(float(paux[j]), 3))
# positions where aux is top-1 outside DI decision positions
decset = {(r["kb"], r["t"]) for r in rows}
risky = [(j, POS[j]) for j in range(N) if gap[j] > 0 and POS[j] not in decset]
print("aux-top1 positions not at DI/DD decision:", len(risky))
import collections
cc = collections.Counter()
for j, (k, t) in risky:
    c = cat[items[k]["id"]] if k < 610 else "GEN"
    cc[c] += 1
print(cc)
