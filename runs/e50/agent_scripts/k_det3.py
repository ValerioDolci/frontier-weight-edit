import numpy as np, json, pickle
def P(L, nm): return W(f"model.language_model.layers.{L}.{nm}.weight").numpy()
def rmsn(x, g, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g
def mlp_in(r_prev, a, L):
    h = r_prev + rmsn(a, P(L, "post_attention_layernorm"))
    return rmsn(h, P(L, "pre_feedforward_layernorm"))
L = 30
dpR = np.load(f"{WORK}/dp_resid.npy"); dpA = np.load(f"{WORK}/dp_attn_out.npy")
lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
dpm = json.load(open(f"{WORK}/dp_meta.json")); dp_ids = [it["id"] for it in dpm["items"]]
fcm = json.load(open(f"{WORK}/fc_meta.json")); fc_items = fcm["items"]
item_of = np.load(f"{WORK}/fc_item_of.npy"); pos_of = np.load(f"{WORK}/fc_pos_of.npy")
dec = {int(k): v for k, v in json.load(open(f"{WORK}/fc_dec.json")).items()}
is_gen = np.array([("id" not in it) or (it.get("id") is None) for it in fc_items])
keep = np.ones(len(item_of), bool)
for i, d in dec.items():
    keep &= ~((item_of == i) & (pos_of >= d - 2) & (pos_of <= d + 2))
fc_group = np.array([fc_items[i]["id"] if not is_gen[i] else 10000 + i for i in item_of])
rng = np.random.default_rng(1)
allg = np.unique(np.concatenate([fc_group, np.array(dp_ids)]))
fold_of_g = dict(zip(allg, rng.integers(0, 5, len(allg))))
dp_fold = np.array([fold_of_g[g] for g in dp_ids]); fc_fold = np.array([fold_of_g[g] for g in fc_group])
Xdp = mlp_in(dpR[L-1].astype(np.float32), dpA[L].astype(np.float32), L).astype(np.float64)
Xfc = mlp_in(np.load(f"{WORK}/fc_resid_{L-1}.npy").astype(np.float32), np.load(f"{WORK}/fc_attn_out_{L}.npy").astype(np.float32), L).astype(np.float64)[keep]
X = np.concatenate([Xdp, Xfc]); ybin = np.concatenate([isDI.astype(np.float64), np.zeros(len(Xfc))])
fold = np.concatenate([dp_fold, fc_fold[keep]])
gen_k = is_gen[item_of[keep]]
theta = 0.78
for wpos, wdd, lamf in [(20, 5, 0.03), (20, 5, 0.01), (40, 10, 0.03)]:
    wt_all = np.ones(len(X)); wt_all[:len(Xdp)][isDI] = wpos; wt_all[:len(Xdp)][isDD | isO] = wdd
    y = ybin - theta
    s = np.zeros(len(X))
    for f in range(5):
        tr = fold != f
        Xt = X[tr]; wt = wt_all[tr]
        Cm = (Xt * wt[:, None]).T @ Xt
        lam = lamf * np.trace(Cm) / Cm.shape[0]
        w = np.linalg.solve(Cm + lam * np.eye(Cm.shape[0]), (Xt * wt[:, None]).T @ y[tr])
        s[~tr] = X[~tr] @ w
    sdp = s[:len(Xdp)]; sfc = s[len(Xdp):]
    G_ = sfc[gen_k]; PN = sfc[~gen_k]
    print(f"wpos{wpos} wdd{wdd} lam{lamf} (score = s-theta, CV): DI p10/p20/p50 {np.percentile(sdp[isDI],10):.3f} {np.percentile(sdp[isDI],20):.3f} {np.median(sdp[isDI]):.3f} | DIerr>0 {np.mean(sdp[isDI&err]>0):.2f} DIok>0 {np.mean(sdp[isDI&~err]>0):.2f} | DDok>0 {np.mean(sdp[isDD&~err]>0):.3f} DDok>0.1 {np.mean(sdp[isDD&~err]>0.1):.3f} | other>0 {np.mean(sdp[isO]>0):.3f} | gen>0 {np.mean(G_>0)*1000:.2f}/1000 gen>0.1 {np.mean(G_>0.1)*1000:.2f}/1000 max {G_.max():.2f} | pracND>0 {np.mean(PN>0)*1000:.2f}/1000 max {PN.max():.2f}")
