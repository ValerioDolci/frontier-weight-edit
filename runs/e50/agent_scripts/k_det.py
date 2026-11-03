import numpy as np, torch, json
def P(L, nm): return W(f"model.language_model.layers.{L}.{nm}.weight").numpy()
def rmsn(x, g, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g
def mlp_in(r_prev, a, L):
    h = r_prev + rmsn(a, P(L, "post_attention_layernorm"))
    return rmsn(h, P(L, "pre_feedforward_layernorm"))
# ---- data
dpR = np.load(f"{WORK}/dp_resid.npy"); dpA = np.load(f"{WORK}/dp_attn_out.npy")
lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
dpm = json.load(open(f"{WORK}/dp_meta.json")); dp_ids = [it["id"] for it in dpm["items"]]
fcm = json.load(open(f"{WORK}/fc_meta.json")); fc_items = fcm["items"]
item_of = np.load(f"{WORK}/fc_item_of.npy"); pos_of = np.load(f"{WORK}/fc_pos_of.npy")
dec = {int(k): v for k, v in json.load(open(f"{WORK}/fc_dec.json")).items()}
NI = len(fc_items)
is_gen = np.array([("id" not in it) or (it.get("id") is None) for it in fc_items])
# fc labels: 0 everywhere, except excluded windows around practice decisions
keep = np.ones(len(item_of), bool)
for i, d in dec.items():
    m = (item_of == i) & (pos_of >= d - 2) & (pos_of <= d + 2)
    keep &= ~m
# group ids for CV: practice id for practice, 10000+index for general
fc_group = np.array([fc_items[i]["id"] if not is_gen[i] else 10000 + i for i in item_of])
dp_group = np.array(dp_ids)
rng = np.random.default_rng(1)
allg = np.unique(np.concatenate([fc_group, dp_group]))
fold_of_g = dict(zip(allg, rng.integers(0, 5, len(allg))))
dp_fold = np.array([fold_of_g[g] for g in dp_group]); fc_fold = np.array([fold_of_g[g] for g in fc_group])
cat = {p["id"]: p["cat"] for p in practice}
def pct(a, qs=(50, 90, 99, 99.9, 100)):
    return " ".join(f"{np.percentile(a, q):.2f}" for q in qs)
res = {}
for L in [29, 30, 31, 32, 33]:
    Xdp = mlp_in(dpR[L-1].astype(np.float32), dpA[L].astype(np.float32), L)
    Xfc = mlp_in(np.load(f"{WORK}/fc_resid_{L-1}.npy").astype(np.float32), np.load(f"{WORK}/fc_attn_out_{L}.npy").astype(np.float32), L)
    X = np.concatenate([Xdp, Xfc[keep]]); y = np.concatenate([isDI.astype(np.float64), np.zeros(keep.sum())])
    fold = np.concatenate([dp_fold, fc_fold[keep]])
    src = np.concatenate([np.zeros(len(Xdp), int), np.ones(keep.sum(), int)])
    mu = X.mean(0)
    for wpos in [20.0]:
        for lamf in [1e-3, 1e-2, 1e-1]:
            s = np.zeros(len(X))
            for f in range(5):
                tr = fold != f
                Xt = X[tr] - mu; yt = y[tr]
                wt = np.where(yt > 0, wpos, 1.0)
                ym = (wt * yt).sum() / wt.sum()
                C = (Xt * wt[:, None]).T @ Xt
                lam = lamf * np.trace(C) / C.shape[0]
                w = np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xt * wt[:, None]).T @ (yt - ym))
                s[~tr] = (X[~tr] - mu) @ w + ym
            sdp = s[:len(Xdp)]; sfc = s[len(Xdp):]
            gen_mask = is_gen[item_of[keep]]
            print(f"L{L} wpos{wpos} lam{lamf}: DIok {pct(sdp[isDI & ~err], (10, 50))} | DIerr {pct(sdp[isDI & err], (10, 50))} | DDok {pct(sdp[isDD & ~err], (50, 90, 100))} | DDerr {pct(sdp[isDD & err], (50, 100))} | other {pct(sdp[isO], (50, 90, 100))} | gen {pct(sfc[gen_mask])} | prac-nondec {pct(sfc[~gen_mask])}")
            res[(L, wpos, lamf)] = (sdp, sfc)
import pickle
pickle.dump({"res": res, "keep": keep}, open(f"{WORK}/det_cv.pkl", "wb"))
