"""Overlap between political-script neurons and safety-refusal neurons. Usage: overlap.py DIR_TW DIR_SAFE OUT.json"""
import json, sys, numpy as np
TW = np.load(f"{sys.argv[1]}/attr_TW.npy"); CT = np.load(f"{sys.argv[1]}/attr_CTRL.npy")
D2 = sys.argv[2]; POL = np.load(f"{D2}/attr_POL.npy"); SF = np.load(f"{D2}/attr_SAFE.npy"); SOK = np.load(f"{D2}/attr_SAFEok.npy")
NL, NI = TW.shape; f = lambda a: a.ravel().astype(np.float64)
def cos(a, b): a, b = f(a), f(b); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def sp(a, b): a, b = f(a), f(b); ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
out = {"cos": {}, "spearman": {}}
for n1, a in (("TW", TW), ("POL", POL)):
    for n2, b in (("SAFE", SF), ("SAFEok", SOK), ("CTRL", CT)): out["cos"][f"{n1}~{n2}"] = cos(a, b)
out["cos"]["TW~POL"] = cos(TW, POL)
print("coseni:", {k: round(v, 3) for k, v in out["cos"].items()})
o = np.argsort(f(TW))
sfv = f(SF); scale = np.percentile(np.abs(sfv), 99.9)
for k in (32, 64):
    sel = o[:k]; out[f"off{k}_SAFE"] = {"max_abs_over_p999": float(np.abs(sfv[sel]).max() / scale), "n_above_p99": int((np.abs(sfv[sel]) > np.percentile(np.abs(sfv), 99)).sum()),
                                         "sum_SAFE": float(sfv[sel].sum()), "sum_TW": float(f(TW)[sel].sum())}
    print(f"off{k}: |SAFE| max / p99.9 = {out[f'off{k}_SAFE']['max_abs_over_p999']:.2f}; quanti sopra il p99 di |SAFE|: {out[f'off{k}_SAFE']['n_above_p99']}; somma SAFE {out[f'off{k}_SAFE']['sum_SAFE']:+.3f} vs somma TW {out[f'off{k}_SAFE']['sum_TW']:+.3f}")
for k in (100, 1000):
    a = set(o[:k]); b = set(np.argsort(-f(SF))[:k]); c = set(np.argsort(f(SF))[:k])
    print(f"top{k} pro-copione ∩ top{k} pro-rifiuto: {len(a & b)} · ∩ top{k} anti-rifiuto: {len(a & c)} (atteso a caso {k*k/(NL*NI):.3f})")
    out[f"top{k}_overlap"] = [len(a & b), len(a & c)]
th_s = np.percentile(np.abs(sfv), 90); th_c = np.percentile(np.abs(f(CT)), 90)
sel = [int(i) for i in o if abs(sfv[i]) < th_s and abs(f(CT)[i]) < th_c][:64]
spec = {}
for i in sel: l, j = divmod(i, NI); spec.setdefault(str(l), []).append(j)
out["selective64"] = spec; out["selective_rank_in_TW"] = [int(np.where(o == i)[0][0]) for i in sel[:32]]
print("insieme selettivo: primi 32 = ranghi in TW", out["selective_rank_in_TW"][:32])
json.dump(out, open(sys.argv[3], "w"), indent=1)
