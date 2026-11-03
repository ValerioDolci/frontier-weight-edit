import numpy as np, torch, json, pickle
def P(L, nm): return W(f"model.language_model.layers.{L}.{nm}.weight").numpy()
def rmsn(x, g, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g
def mlp_in(r_prev, a, L):
    h = r_prev + rmsn(a, P(L, "post_attention_layernorm"))
    return rmsn(h, P(L, "pre_feedforward_layernorm"))
def auc(pos, neg):
    allv = np.concatenate([pos, neg]); order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
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
gen_k = is_gen[item_of[keep]]
out = {}
for L in [29, 30, 31, 33]:
    Xdp = mlp_in(dpR[L-1].astype(np.float32), dpA[L].astype(np.float32), L)
    Xfc = mlp_in(np.load(f"{WORK}/fc_resid_{L-1}.npy").astype(np.float32), np.load(f"{WORK}/fc_attn_out_{L}.npy").astype(np.float32), L)[keep]
    X = np.concatenate([Xdp, Xfc]).astype(np.float64); y = np.concatenate([isDI.astype(np.float64), np.zeros(len(Xfc))])
    fold = np.concatenate([dp_fold, fc_fold[keep]])
    # extra weight on DD decision negatives
    for wpos, wdd in [(20.0, 5.0), (60.0, 20.0)]:
        wt_all = np.ones(len(X)); wt_all[:len(Xdp)][isDI] = wpos; wt_all[:len(Xdp)][isDD | isO] = wdd
        for lamf in [3e-3, 3e-2]:
            s = np.zeros(len(X))
            for f in range(5):
                tr = fold != f
                Xt = X[tr]; yt = y[tr]; wt = wt_all[tr]
                mu = (wt[:, None] * Xt).sum(0) / wt.sum(); ym = (wt * yt).sum() / wt.sum()
                Xc = Xt - mu
                C = (Xc * wt[:, None]).T @ Xc
                lam = lamf * np.trace(C) / C.shape[0]
                w = np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xc * wt[:, None]).T @ (yt - ym))
                s[~tr] = (X[~tr] - mu) @ w + ym
            sdp = s[:len(Xdp)]; sfc = s[len(Xdp):]
            DI_ = sdp[isDI]; DIe = sdp[isDI & err]; DDo = sdp[isDD & ~err]; O_ = sdp[isO]; G_ = sfc[gen_k]; PN = sfc[~gen_k]
            thr = np.percentile(DI_, 20)  # threshold keeping 80% of DI
            print(f"L{L} wp{wpos} wdd{wdd} lam{lamf}: AUC DI/gen {auc(DI_, G_):.4f} DI/DDok {auc(DI_, DDo):.3f} DIerr/DDok {auc(DIe, DDo):.3f} DI/other {auc(DI_, O_):.3f} | thr20={thr:.2f}: DIerr>thr {np.mean(DIe>thr):.2f} DDok>thr {np.mean(DDo>thr):.2f} other>thr {np.mean(O_>thr):.3f} gen>thr {np.mean(G_>thr)*1000:.2f}/1000 pracND>thr {np.mean(PN>thr)*1000:.2f}/1000")
            out[(L, wpos, wdd, lamf)] = (sdp, sfc)
pickle.dump({"out": out, "keep": keep}, open(f"{WORK}/det_cv2.pkl", "wb"))
