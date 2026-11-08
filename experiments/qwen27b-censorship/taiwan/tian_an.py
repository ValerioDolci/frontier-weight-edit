import json, numpy as np
d = json.load(open("./taiwan/dec_TIAN_27b/decision.json")); print(d["n"])
for r in d["cand"]:
    print(r["is_cand"], r["l"], r["j"], round(r["attr_A"], 2), " ".join("%s:%+.2f/%d" % (c, r["dmargin_" + c], r["flips_" + c]) for c in d["cats"]))
T = "./taiwan/"; A = lambda p: np.load(T + p).ravel().astype(np.float64)
TI = A("dec_TIAN_27b/attr_TIAN.npy"); TW = A("dec_TW_27b/attr_TW.npy"); SF = A("dec_SAFE_27b/attr_SAFE.npy"); PO = A("dec_SAFE_27b/attr_POL.npy")
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
print("coseni TIAN~TW %.3f  TIAN~SAFE %.3f  TIAN~POL %.3f  TW~SAFE %.3f" % (cos(TI, TW), cos(TI, SF), cos(TI, PO), cos(TW, SF)))
k = 1000; o = lambda x, s: set(np.argsort(s * x)[:k])
print("top1000 pro-rifiuto-Tiananmen (attr negativa) ∩ top1000 pro-copione TW:", len(o(TI, 1) & o(TW, 1)), "∩ top1000 pro-rifiuto SAFE:", len(o(TI, 1) & o(SF, -1)), "atteso", round(k * k / TI.size, 2))
