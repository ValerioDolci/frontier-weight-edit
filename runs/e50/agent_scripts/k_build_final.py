import numpy as np, json
# FINAL: all practice + general (+prompt negatives) + 160 synthetic decision positions; v1 directions/thresholds
fam = json.load(open(f"{WORK}/fa_meta.json"))
labp = np.load(f"{WORK}/fa_lab.npy"); wpos = np.load(f"{WORK}/fa_w.npy")
use_all = labp >= 0
Xs30 = np.load(f"{WORK}/synth_x30.npy"); Xs40 = np.load(f"{WORK}/synth_x40.npy")
ys = np.array([1.0] * 110 + [0.0] * 50); ws = np.where(ys > 0, 20.0, 5.0)
def fit_all(X, Xs, theta=0.78, lamf=0.03):
    Xu = np.concatenate([X[use_all].astype(np.float64), Xs]); yu = np.concatenate([(labp[use_all] == 1).astype(np.float64), ys]) - theta
    wu = np.concatenate([wpos[use_all], ws])
    C = (Xu * wu[:, None]).T @ Xu
    lam = lamf * np.trace(C) / C.shape[0]
    return np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xu * wu[:, None]).T @ yu)
X30 = mlp_in(np.load(f"{WORK}/fa_resid_29.npy").astype(np.float32), np.load(f"{WORK}/fa_attn_out_30.npy").astype(np.float32), 30)
X40n = mlp_in(np.load(f"{WORK}/fa1_resid_39.npy").astype(np.float32), np.load(f"{WORK}/fa1_attn_out_40.npy").astype(np.float32), 40)
u30c = np.load(f"{WORK}/u30c.npy"); u40c = np.load(f"{WORK}/u40c.npy")
t30, t40 = -0.06, -0.03
w30f = fit_all(X30, Xs30) - t30 * u30c
w40f = fit_all(X40n, Xs40) - t40 * u40c
np.save(f"{WORK}/w30_final.npy", w30f); np.save(f"{WORK}/w40_final.npy", w40f)
d_imp30 = np.load(f"{WORK}/steer_dimp_30.npy"); d_st30 = np.load(f"{WORK}/steer_dstart_30.npy")
dir30 = 24.0 * d_imp30 / np.linalg.norm(d_imp30) + 8.0 * d_st30 / np.linalg.norm(d_st30)
d40 = np.load(f"{WORK}/steer_d40.npy"); b2 = json.load(open(f"{WORK}/build2.json"))
build_pair(30, w30f, u30c, 1.0, dir30, float(np.linalg.norm(dir30)), [13770, 12872], rms_m=0.158)
build_pair(40, w40f, u40c, 1.0, d40, b2["alpha40"], b2["n40"], rms_m=0.714)
# in-sample sanity
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy"); is_prompt = np.load(f"{WORK}/fa_is_prompt.npy")
dec_abs = {int(k): v for k, v in json.load(open(f"{WORK}/fa_dec.json"))["dec_abs"].items()}
items = fam["items"]; cat = {p["id"]: p["cat"] for p in practice}
kindp = np.array(["gen" if "en" in it else cat[it["id"]] for it in items])[item_of]
isdec = np.zeros(len(item_of), bool)
for i, d in dec_abs.items(): isdec[(item_of == i) & (pos_of == d)] = True
m30 = neuron_mult(X30.astype(np.float64), w30f, u30c, 1.0); m40 = neuron_mult(X40n.astype(np.float64), w40f, u40c, 1.0)
for nm, m in [("L30", m30), ("L40", m40)]:
    print(nm, "dec DI %.2f DD %.3f other %.3f | gen comp>0.1 %.2f/1000 gen prm>0.1 %.2f/1000" % (np.mean(m[isdec & (kindp == 'DI')] > 0.5), np.mean(m[isdec & (kindp == 'DD')] > 0.5), np.mean(m[isdec & np.isin(kindp, ['PERF','DURP','HAB','COND'])] > 0.5), np.mean(m[(kindp=='gen') & ~is_prompt] > 0.1)*1000, np.mean(m[(kindp=='gen') & is_prompt] > 0.1)*1000))
print(edits())
