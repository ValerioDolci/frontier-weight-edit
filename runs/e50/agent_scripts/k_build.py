import numpy as np, torch, json, pickle
def P(L, nm): return W(f"model.language_model.layers.{L}.{nm}.weight").numpy()
def rmsn(x, g, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g
def mlp_in(r_prev, a, L):
    h = r_prev + rmsn(a, P(L, "post_attention_layernorm"))
    return rmsn(h, P(L, "pre_feedforward_layernorm"))
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
L = 30
dpR = np.load(f"{WORK}/dp_resid.npy"); dpA = np.load(f"{WORK}/dp_attn_out.npy")
lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
fcm = json.load(open(f"{WORK}/fc_meta.json")); fc_items = fcm["items"]
item_of = np.load(f"{WORK}/fc_item_of.npy"); pos_of = np.load(f"{WORK}/fc_pos_of.npy")
dec = {int(k): v for k, v in json.load(open(f"{WORK}/fc_dec.json")).items()}
is_gen = np.array([("id" not in it) or (it.get("id") is None) for it in fc_items])
keep = np.ones(len(item_of), bool)
for i, d in dec.items():
    keep &= ~((item_of == i) & (pos_of >= d - 2) & (pos_of <= d + 2))
Xdp = mlp_in(dpR[L-1].astype(np.float32), dpA[L].astype(np.float32), L).astype(np.float64)
XfcAll = mlp_in(np.load(f"{WORK}/fc_resid_{L-1}.npy").astype(np.float32), np.load(f"{WORK}/fc_attn_out_{L}.npy").astype(np.float32), L).astype(np.float64)
Xfc = XfcAll[keep]
X = np.concatenate([Xdp, Xfc]); y = np.concatenate([isDI.astype(np.float64), np.zeros(len(Xfc))])
wt = np.ones(len(X)); wt[:len(Xdp)][isDI] = 20.0; wt[:len(Xdp)][isDD | isO] = 5.0
# uncentered weighted ridge (constant component of x acts as intercept)
C = (X * wt[:, None]).T @ X
lam = 0.03 * np.trace(C) / C.shape[0]
w = np.linalg.solve(C + lam * np.eye(C.shape[0]), (X * wt[:, None]).T @ y)
sdp = Xdp @ w; sfc = XfcAll @ w
print("in-sample: DI p10/p50 %.2f %.2f | DIerr p10/p50 %.2f %.2f | DDok p50/p90/max %.2f %.2f %.2f | other max %.2f | gen p99.9/max %.2f %.2f" % (
    np.percentile(sdp[isDI], 10), np.median(sdp[isDI]), np.percentile(sdp[isDI & err], 10), np.median(sdp[isDI & err]),
    np.median(sdp[isDD & ~err]), np.percentile(sdp[isDD & ~err], 90), sdp[isDD & ~err].max(), sdp[isO].max(),
    np.percentile(sfc[is_gen[item_of]], 99.9), sfc[is_gen[item_of]].max()))
xbar = np.concatenate([Xdp, XfcAll]).mean(0); u_hat = xbar / np.linalg.norm(xbar)
proj = np.concatenate([Xdp, XfcAll]) @ u_hat
m0 = proj.mean(); print("m0 %.3f std %.3f" % (m0, proj.std()))
np.save(f"{WORK}/det_w_L{L}.npy", w); np.save(f"{WORK}/det_uhat_L{L}.npy", u_hat)
json.dump({"m0": float(m0)}, open(f"{WORK}/det_m0_L{L}.json", "w"))
# neuron importance on a subsample of general positions
Gw = P(L, "mlp.gate_proj"); Uw = P(L, "mlp.up_proj"); Dw = P(L, "mlp.down_proj")
idx = np.random.default_rng(0).choice(np.where(is_gen[item_of])[0], 4000, replace=False)
xs = XfcAll[idx].astype(np.float32)
act = gelu(xs @ Gw.T) * (xs @ Uw.T)
dn = np.linalg.norm(Dw, axis=0)
imp_gen = np.abs(act).mean(0) * dn
act_di = gelu(Xdp[isDI].astype(np.float32) @ Gw.T) * (Xdp[isDI].astype(np.float32) @ Uw.T)
imp_di = np.abs(act_di).mean(0) * dn
order = np.argsort(imp_gen + imp_di)
print("least important neurons:", order[:8].tolist(), "imp_gen", np.round(imp_gen[order[:8]], 6).tolist(), "imp_di", np.round(imp_di[order[:8]], 6).tolist())
print("median neuron importance gen %.5f" % np.median(imp_gen))
json.dump({"neurons": order[:8].tolist()}, open(f"{WORK}/neurons_L{L}.json", "w"))
