import numpy as np, pickle
S = pickle.load(open(f"{WORK}/cv_type30.pkl", "rb"))
for t in [-0.04, -0.06, -0.08]:
    smax = np.maximum(np.maximum(S["start"], S["subj"]), S["aux"])
    ddok = isdec & (kind == "DD") & ~e
    print(f"t={t:+.2f}: union DDok FP {np.mean(smax[ddok] > t):.3f} | DI {np.mean(smax[posDI] > t):.2f} DIerr {np.mean(smax[posDI & e] > t):.2f} | gen comp {np.mean(smax[(kind == 'gen') & ~pu] > t)*1000:.2f} gen prompt {np.mean(smax[(kind == 'gen') & pu] > t)*1000:.2f} | other-cat dec {np.mean(smax[isdec & np.isin(kind, ['PERF','DURP','HAB','COND'])] > t):.3f} | DD prompt {np.mean(smax[(kind == 'DD') & pu] > t)*1000:.2f} DI comp nondec {np.mean(smax[(kind == 'DI') & ~pu & ~isdec] > t)*1000:.2f}")
