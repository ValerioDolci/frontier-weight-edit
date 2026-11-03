import numpy as np, torch, json, pickle
fam = json.load(open(f"{WORK}/fa_meta.json")); items = fam["items"]; meta = fam["meta"]
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy"); is_prompt = np.load(f"{WORK}/fa_is_prompt.npy")
labp = np.load(f"{WORK}/fa_lab.npy"); wpos = np.load(f"{WORK}/fa_w.npy")
dec_abs = {int(k): v for k, v in json.load(open(f"{WORK}/fa_dec.json"))["dec_abs"].items()}
cat = {p["id"]: p["cat"] for p in practice}
use = labp >= 0
def fit_full(X, theta=0.78, lamf=0.03, sel=None):
    m = use if sel is None else (use & sel)
    Xu = X[m].astype(np.float64); yu = (labp[m] == 1).astype(np.float64) - theta; wu = wpos[m]
    C = (Xu * wu[:, None]).T @ Xu
    lam = lamf * np.trace(C) / C.shape[0]
    return np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xu * wu[:, None]).T @ yu)
def const_dir(X):
    # direction (single dim) with the most stable positive projection across positions
    mu = X.mean(0); sd = X.std(0) + 1e-6
    r = np.abs(mu) / sd
    i = int(np.argmax(r * (np.abs(mu) > 0.5)))
    u = np.zeros(X.shape[1]); u[i] = np.sign(mu[i])
    pr = X @ u
    return u, float(pr.mean()), float(pr.std()), i
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
def neuron_mult(X, w, u, m0, kappa=40.0, c=4.0):
    z = kappa * (X @ w); up = X @ u / m0
    return (gelu(z) - gelu(z - c)) * up / c
def build_pair(L, w, u, m0, direction, alpha, neurons, rms_m, kappa=40.0, c=4.0, eps=0.2):
    g = W(f"model.language_model.layers.{L}.post_feedforward_layernorm.weight").numpy().astype(np.float64)
    sL = W(f"model.language_model.layers.{L}.layer_scalar").item()
    dhat = direction / np.linalg.norm(direction)
    v = rms_m * (alpha * dhat / sL) * g / (g ** 2 + eps ** 2)
    Gt = W(f"model.language_model.layers.{L}.mlp.gate_proj.weight").clone()
    Ut = W(f"model.language_model.layers.{L}.mlp.up_proj.weight").clone()
    Dt = W(f"model.language_model.layers.{L}.mlp.down_proj.weight").clone()
    jA, jB = neurons
    Gt[jA] = torch.tensor(kappa * w, dtype=torch.float32)
    Gt[jB] = torch.tensor(kappa * w - (c / m0) * u, dtype=torch.float32)
    Ut[jA] = torch.tensor(u / m0, dtype=torch.float32); Ut[jB] = torch.tensor(u / m0, dtype=torch.float32)
    Dt[:, jA] = torch.tensor(v / c, dtype=torch.float32); Dt[:, jB] = torch.tensor(-v / c, dtype=torch.float32)
    save_edit(f"model.language_model.layers.{L}.mlp.gate_proj.weight", Gt)
    save_edit(f"model.language_model.layers.{L}.mlp.up_proj.weight", Ut)
    save_edit(f"model.language_model.layers.{L}.mlp.down_proj.weight", Dt)
    return v
def least_neurons(L, X, k=2):
    Gw = W(f"model.language_model.layers.{L}.mlp.gate_proj.weight").numpy(); Uw = W(f"model.language_model.layers.{L}.mlp.up_proj.weight").numpy(); Dw = W(f"model.language_model.layers.{L}.mlp.down_proj.weight").numpy()
    idx = np.random.default_rng(0).choice(len(X), 6000, replace=False)
    xs = X[idx]
    act = gelu(xs @ Gw.T) * (xs @ Uw.T)
    imp = np.abs(act).mean(0) * np.linalg.norm(Dw, axis=0)
    o = np.argsort(imp)
    return o[:k].tolist(), imp[o[:k]].tolist(), float(np.median(imp))
MB = pickle.load(open(f"{WORK}/margin_orig.pkl", "rb")); ME2 = pickle.load(open(f"{WORK}/margin_e2.pkl", "rb"))
cm2 = dict(zip(ME2["id"].tolist(), ME2["cm"].tolist()))
L2 = json.load(open(f"{WORK}/dp_lab2.json")); ptype = dict(zip(L2["ids"], L2["pt"]))
dd = json.load(open(WORK + "/dp.json")); errDI = set(dd["ERR_DI"]); errDD = set(dd["ERR_DD"])
# ---------------- L30 stage
L = 30
X30 = mlp_in(np.load(f"{WORK}/fa_resid_29.npy").astype(np.float32), np.load(f"{WORK}/fa_attn_out_30.npy").astype(np.float32), L)
w30 = fit_full(X30)
u30, m030, sd30, i30 = const_dir(X30)
print("L30 const dim", i30, "m0 %.2f sd %.2f" % (m030, sd30))
n30 = [13770, 12872]
mult30 = neuron_mult(X30.astype(np.float64), w30, u30, m030)
np.save(f"{WORK}/w30_v2.npy", w30); np.save(f"{WORK}/u30_v2.npy", u30)
d_imp30 = np.load(f"{WORK}/steer_dimp_30.npy"); d_st30 = np.load(f"{WORK}/steer_dstart_30.npy")
dir30 = 24.0 * d_imp30 / np.linalg.norm(d_imp30) + 8.0 * d_st30 / np.linalg.norm(d_st30)
build_pair(30, w30, u30, m030, dir30, float(np.linalg.norm(dir30)), n30, rms_m=0.158)
# ---------------- L40 stage
L = 40
X40 = mlp_in(np.load(f"{WORK}/fa_resid_39.npy").astype(np.float32), np.load(f"{WORK}/fa_attn_out_40.npy").astype(np.float32), L)
w40 = fit_full(X40)
u40, m040, sd40, i40 = const_dir(X40)
print("L40 const dim", i40, "m0 %.2f sd %.2f" % (m040, sd40))
n40, imps, med = least_neurons(40, X40[~is_prompt]); print("L40 neurons", n40, imps, "median", med)
# push direction at L40: decision positions, imperfetto vs trapassato under edit 2, types start+subj
R40 = np.load(f"{WORK}/fa_resid_40.npy").astype(np.float64)
pos_idx = {};
for i, d in dec_abs.items():
    j = np.where((item_of == i) & (pos_of == d))[0][0]; pos_idx[items[i]["id"]] = j
ds = []; wsum = 0
for t in ["start", "subj"]:
    imp_j = []; trap_j = []
    for pid, j in pos_idx.items():
        c = cat[pid]
        if c not in ("DI", "DD") or ptype[pid] != t: continue
        base_imp = (c == "DI") != ((pid in errDI) or (pid in errDD))  # baseline output imperfetto?
        # under edit2: correct-side margin >0 means correct choice
        corr = cm2[pid] > 0
        is_imp = corr if c == "DI" else (not corr)
        (imp_j if is_imp else trap_j).append(j)
    a = R40[imp_j].mean(0); b = R40[trap_j].mean(0); n = min(len(imp_j), len(trap_j))
    print(t, "n imp", len(imp_j), "n trap", len(trap_j), "|d| %.2f" % np.linalg.norm(a - b))
    ds.append((a - b) * n); wsum += n
d40 = sum(ds) / wsum
np.save(f"{WORK}/steer_d40.npy", d40)
MO40 = np.load(f"{WORK}/dp_mlp_out.npy")[40].astype(np.float64); lab = np.load(f"{WORK}/dp_labels.npy").astype(bool)
rms40 = np.sqrt((MO40[lab[0]] ** 2).mean(1)).mean(); print("rms_m L40 %.3f" % rms40)
alpha40 = 1.0 * np.linalg.norm(d40)
print("alpha40 %.2f" % alpha40)
build_pair(40, w40, u40, m040, d40, alpha40, n40, rms_m=rms40)
np.save(f"{WORK}/w40_v1.npy", w40); np.save(f"{WORK}/u40_v1.npy", u40)
json.dump({"m030": m030, "m040": m040, "n30": n30, "n40": n40, "alpha40": float(alpha40)}, open(f"{WORK}/build2.json", "w"))
mult40 = neuron_mult(X40.astype(np.float64), w40, u40, m040)
kind = np.array(["gen" if "en" in items[i] else cat[items[i]["id"]] for i in item_of])
for nm, mult in [("L30", mult30), ("L40", mult40)]:
    o = nm
    for k in ["DI", "DD", "gen"]:
        pm = (kind == k) & is_prompt; cmk = (kind == k) & ~is_prompt
        o += f" | {k} prompt>0.1 {np.mean(mult[pm] > 0.1)*1000:.2f} comp>0.1 {np.mean(mult[cmk] > 0.1)*1000:.2f}/1000"
    print(o)
print(edits())
