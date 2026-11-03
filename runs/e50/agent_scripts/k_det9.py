import numpy as np, torch, json
A = load_acts("full_all_v1")
acts = A["acts"]; meta1 = A["meta"]
lens1 = [acts["resid.39"][i].shape[0] for i in range(len(meta1))]
lens0 = np.bincount(np.load(f"{WORK}/fa_item_of.npy"))
print("positions match:", list(lens0) == lens1)
for L in [39, 40]:
    for kind_ in ["resid", "attn_out"]:
        arr = torch.cat(acts[f"{kind_}.{L}"], 0).numpy().astype(np.float16)
        np.save(f"{WORK}/fa1_{kind_}_{L}.npy", arr)
X40n = mlp_in(np.load(f"{WORK}/fa1_resid_39.npy").astype(np.float32), np.load(f"{WORK}/fa1_attn_out_40.npy").astype(np.float32), 40)
fam = json.load(open(f"{WORK}/fa_meta.json")); items = fam["items"]
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy"); is_prompt = np.load(f"{WORK}/fa_is_prompt.npy")
labp = np.load(f"{WORK}/fa_lab.npy"); wpos = np.load(f"{WORK}/fa_w.npy")
dec_abs = {int(k): v for k, v in json.load(open(f"{WORK}/fa_dec.json"))["dec_abs"].items()}
cat = {p["id"]: p["cat"] for p in practice}
dd = json.load(open(WORK + "/dp.json")); errDI = set(dd["ERR_DI"]); errDD = set(dd["ERR_DD"])
kindp = np.array(["gen" if "en" in it else cat[it["id"]] for it in items])[item_of]
isdec = np.zeros(len(item_of), bool)
for i, d in dec_abs.items(): isdec[(item_of == i) & (pos_of == d)] = True
pidp = np.array([it.get("id", -1) if "id" in it else -1 for it in items])[item_of]
ep = np.array([(p in errDI) or (p in errDD) for p in pidp])
w40g_old = np.load(f"{WORK}/w40g_full.npy"); u40c = np.load(f"{WORK}/u40c.npy")
for nm, X in [("old-acts", X40), ("new-acts", X40n)]:
    m = neuron_mult(X.astype(np.float64), w40g_old, u40c, 1.0)
    o = f"old L40 detector on {nm}:"
    for k in ["DI", "DD"]:
        for e in [False, True]:
            q = isdec & (kindp == k) & (ep == e)
            o += f" {k}{'err' if e else 'ok'} {np.mean(m[q] > 0.5):.2f}"
    o += f" | gen comp>0.1 {np.mean(m[(kindp=='gen') & ~is_prompt] > 0.1)*1000:.2f}/1000 | DI comp nondec>0.1 {np.mean(m[(kindp=='DI') & ~is_prompt & ~isdec] > 0.1)*1000:.2f}/1000"
    print(o)
# CV of retrained detector on new activations
use = labp >= 0
group = np.array([items[i]["id"] if "id" in items[i] else 10000 + i for i in item_of])
rng = np.random.default_rng(1); ug = np.unique(group); fg = dict(zip(ug, rng.integers(0, 5, len(ug))))
fold = np.array([fg[g] for g in group])
Xu = X40n[use].astype(np.float64); yu = (labp[use] == 1).astype(np.float64) - 0.78; wu = wpos[use]; fu = fold[use]
s = np.zeros(len(Xu))
for f in range(5):
    tr = fu != f
    Xt = Xu[tr]; wt = wu[tr]
    C = (Xt * wt[:, None]).T @ Xt
    lam = 0.03 * np.trace(C) / C.shape[0]
    wv = np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xt * wt[:, None]).T @ yu[tr])
    te = fu == f; s[te] = Xu[te] @ wv
iu = item_of[use]; pu = is_prompt[use]
ku = kindp[use]; du = isdec[use]; eu = ep[use]
for t in [0.0, -0.03, -0.06]:
    print(f"retrained CV t={t:+.2f}: DI {np.mean(s[du & (ku=='DI')] > t):.2f} DIerr {np.mean(s[du & (ku=='DI') & eu] > t):.2f} DDok {np.mean(s[du & (ku=='DD') & ~eu] > t):.3f} other {np.mean(s[du & np.isin(ku, ['PERF','DURP','HAB','COND'])] > t):.3f} | gen comp {np.mean(s[(ku=='gen') & ~pu] > t)*1000:.2f} gen prm {np.mean(s[(ku=='gen') & pu] > t)*1000:.2f} DI prm {np.mean(s[(ku=='DI') & pu & ~du] > t)*1000:.2f} DI comp nondec {np.mean(s[(ku=='DI') & ~pu & ~du] > t)*1000:.2f}")
np.save(f"{WORK}/X40n_cv.npy", s)
