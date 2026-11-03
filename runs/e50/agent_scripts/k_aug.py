import numpy as np, torch, json, pickle
S = load_acts("synth_last"); sa = S["acts"]
def st(key): return torch.cat(sa[key], 0).numpy().astype(np.float32)
Xs30 = mlp_in(st("resid.29"), st("attn_out.30"), 30).astype(np.float64)
Xs40 = mlp_in(st("resid.39"), st("attn_out.40"), 40).astype(np.float64)
ns = Xs30.shape[0]; ys = np.array([1.0] * 110 + [0.0] * 50); ws = np.where(ys > 0, 20.0, 5.0)
np.save(f"{WORK}/synth_x30.npy", Xs30); np.save(f"{WORK}/synth_x40.npy", Xs40)
print("synthetic positions", ns)
fam = json.load(open(f"{WORK}/fa_meta.json")); fa_items = fam["items"]
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy"); is_prompt = np.load(f"{WORK}/fa_is_prompt.npy")
labp = np.load(f"{WORK}/fa_lab.npy"); wpos = np.load(f"{WORK}/fa_w.npy")
dec_abs = {int(k): v for k, v in json.load(open(f"{WORK}/fa_dec.json"))["dec_abs"].items()}
cat = {p["id"]: p["cat"] for p in practice}
item_pid = np.array([it.get("id", -1) if "id" in it else -1 for it in fa_items])
pos_val = np.array([(p >= 0) and (p % 3 == 0) for p in item_pid])[item_of]
use_tr = (labp >= 0) & ~pos_val
def fit_aug(X, Xs, sel, theta=0.78, lamf=0.03, aug=True):
    Xu = X[sel].astype(np.float64); yu = (labp[sel] == 1).astype(np.float64) - theta; wu = wpos[sel]
    if aug:
        Xu = np.concatenate([Xu, Xs]); yu = np.concatenate([yu, ys - theta]); wu = np.concatenate([wu, ws])
    C = (Xu * wu[:, None]).T @ Xu
    lam = lamf * np.trace(C) / C.shape[0]
    return np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xu * wu[:, None]).T @ yu)
X30 = mlp_in(np.load(f"{WORK}/fa_resid_29.npy").astype(np.float32), np.load(f"{WORK}/fa_attn_out_30.npy").astype(np.float32), 30)
X40n = mlp_in(np.load(f"{WORK}/fa1_resid_39.npy").astype(np.float32), np.load(f"{WORK}/fa1_attn_out_40.npy").astype(np.float32), 40)
X40o = mlp_in(np.load(f"{WORK}/fa_resid_39.npy").astype(np.float32), np.load(f"{WORK}/fa_attn_out_40.npy").astype(np.float32), 40)
u30c = np.load(f"{WORK}/u30c.npy"); u40c = np.load(f"{WORK}/u40c.npy")
t30, t40 = -0.06, -0.03
res = {}
for tag, aug in [("plain", False), ("aug", True)]:
    w30 = fit_aug(X30, Xs30, use_tr, aug=aug) - t30 * u30c
    w40 = fit_aug(X40n, Xs40, use_tr, aug=aug) - t40 * u40c
    res[tag] = (w30, w40)
    np.save(f"{WORK}/w30_tr_{tag}.npy", w30); np.save(f"{WORK}/w40_tr_{tag}.npy", w40)
dd = json.load(open(WORK + "/dp.json")); errDI = set(dd["ERR_DI"]); errDD = set(dd["ERR_DD"])
isdec = np.zeros(len(item_of), bool)
for i, d in dec_abs.items(): isdec[(item_of == i) & (pos_of == d)] = True
kindp = np.array(["gen" if "en" in it else cat[it["id"]] for it in fa_items])[item_of]
ep = np.array([(p in errDI) or (p in errDD) for p in item_pid[item_of]])
for tag in res:
    w30, w40 = res[tag]
    m30 = neuron_mult(X30.astype(np.float64), w30, u30c, 1.0); m40 = neuron_mult(X40n.astype(np.float64), w40, u40c, 1.0)
    for nm, m in [("L30", m30), ("L40", m40), ("union", np.maximum(m30, m40))]:
        o = f"{tag} {nm} VAL:"
        for k in ["DI", "DD"]:
            for e in [False, True]:
                q = isdec & pos_val & (kindp == k) & (ep == e)
                o += f" {k}{'err' if e else 'ok'} {np.mean(m[q] > 0.5):.2f}"
        q = (kindp == "gen") & ~is_prompt
        o += f" | gen comp>0.1 {np.mean(m[q] > 0.1)*1000:.2f}/1000 gen prm>0.1 {np.mean(m[(kindp=='gen') & is_prompt] > 0.1)*1000:.2f}/1000"
        print(o)
    # synthetic in-sample firing
    s30 = neuron_mult(Xs30, w30, u30c, 1.0); s40 = neuron_mult(Xs40, w40, u40c, 1.0)
    print(f"   synth (train for aug): DI L30 {np.mean(s30[:110] > 0.5):.2f} L40 {np.mean(s40[:110] > 0.5):.2f} | DD L30 {np.mean(s30[110:] > 0.5):.2f} L40 {np.mean(s40[110:] > 0.5):.2f}")
