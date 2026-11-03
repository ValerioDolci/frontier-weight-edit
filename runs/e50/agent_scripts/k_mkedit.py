import numpy as np, torch, json
def P(L, nm): return W(f"model.language_model.layers.{L}.{nm}.weight").numpy()
def rmsn(x, g, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g
def mlp_in(r_prev, a, L):
    h = r_prev + rmsn(a, P(L, "post_attention_layernorm"))
    return rmsn(h, P(L, "pre_feedforward_layernorm"))
def gelu(z): return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))

def fit_detector(L, theta=0.78, wpos=20.0, wdd=5.0, lamf=0.03):
    dpR = np.load(f"{WORK}/dp_resid.npy"); dpA = np.load(f"{WORK}/dp_attn_out.npy")
    lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
    fcm = json.load(open(f"{WORK}/fc_meta.json")); fc_items = fcm["items"]
    item_of = np.load(f"{WORK}/fc_item_of.npy"); pos_of = np.load(f"{WORK}/fc_pos_of.npy")
    dec = {int(k): v for k, v in json.load(open(f"{WORK}/fc_dec.json")).items()}
    keep = np.ones(len(item_of), bool)
    for i, d in dec.items():
        keep &= ~((item_of == i) & (pos_of >= d - 2) & (pos_of <= d + 2))
    Xdp = mlp_in(dpR[L-1].astype(np.float32), dpA[L].astype(np.float32), L).astype(np.float64)
    Xfc = mlp_in(np.load(f"{WORK}/fc_resid_{L-1}.npy").astype(np.float32), np.load(f"{WORK}/fc_attn_out_{L}.npy").astype(np.float32), L).astype(np.float64)
    X = np.concatenate([Xdp, Xfc[keep]]); y = np.concatenate([isDI.astype(np.float64), np.zeros(keep.sum())]) - theta
    wt = np.ones(len(X)); wt[:len(Xdp)][isDI] = wpos; wt[:len(Xdp)][isDD | isO] = wdd
    C = (X * wt[:, None]).T @ X
    lam = lamf * np.trace(C) / C.shape[0]
    w = np.linalg.solve(C + lam * np.eye(C.shape[0]), (X * wt[:, None]).T @ y)
    return w, Xdp, Xfc

def build_edit(L, w, direction, alpha, kappa=40.0, c=4.0, eps=0.2, neurons=(13770, 12872), save=True, rms_m=None):
    g = P(L, "post_feedforward_layernorm").astype(np.float64)
    sL = W(f"model.language_model.layers.{L}.layer_scalar").item()
    dhat = direction / np.linalg.norm(direction)
    dpost = alpha * dhat / sL
    if rms_m is None:
        MO = np.load(f"{WORK}/dp_mlp_out.npy")[L].astype(np.float64)
        lab = np.load(f"{WORK}/dp_labels.npy").astype(bool)
        rms_m = np.sqrt((MO[lab[0]] ** 2).mean(1)).mean()
    v = rms_m * dpost * g / (g ** 2 + eps ** 2)
    achieved = g * v / rms_m * sL
    print(f"  rms_m {rms_m:.3f}  |achieved| {np.linalg.norm(achieved):.2f} (target {alpha}) cos(achieved, dir) {achieved @ dhat / np.linalg.norm(achieved):.3f}  rms(v)/rms_m {np.sqrt((v**2).mean())/rms_m:.3f}")
    Gt = W(f"model.language_model.layers.{L}.mlp.gate_proj.weight").clone()
    Ut = W(f"model.language_model.layers.{L}.mlp.up_proj.weight").clone()
    Dt = W(f"model.language_model.layers.{L}.mlp.down_proj.weight").clone()
    jA, jB = neurons
    u_hat = np.load(f"{WORK}/det_uhat_L{L}.npy"); m0 = json.load(open(f"{WORK}/det_m0_L{L}.json"))["m0"]
    Gt[jA] = torch.tensor(kappa * w, dtype=torch.float32)
    Gt[jB] = torch.tensor(kappa * w - (c / m0) * u_hat, dtype=torch.float32)
    Ut[jA] = torch.tensor(u_hat / m0, dtype=torch.float32)
    Ut[jB] = torch.tensor(u_hat / m0, dtype=torch.float32)
    Dt[:, jA] = torch.tensor(v / c, dtype=torch.float32)
    Dt[:, jB] = torch.tensor(-v / c, dtype=torch.float32)
    if save:
        save_edit(f"model.language_model.layers.{L}.mlp.gate_proj.weight", Gt)
        save_edit(f"model.language_model.layers.{L}.mlp.up_proj.weight", Ut)
        save_edit(f"model.language_model.layers.{L}.mlp.down_proj.weight", Dt)
    return Gt, Ut, Dt, v

def neuron_out(X, L, w, kappa=40.0, c=4.0):
    u_hat = np.load(f"{WORK}/det_uhat_L{L}.npy"); m0 = json.load(open(f"{WORK}/det_m0_L{L}.json"))["m0"]
    z = kappa * (X @ w); up = X @ u_hat / m0
    return (gelu(z) - gelu(z - c)) * up / c   # multiplier of v

L = 30
w, Xdp, Xfc = fit_detector(L)
np.save(f"{WORK}/det_w2_L{L}.npy", w)
lab = np.load(f"{WORK}/dp_labels.npy").astype(bool); isDI, isDD, isO, err, imp = lab
fcm = json.load(open(f"{WORK}/fc_meta.json")); fc_items = fcm["items"]; item_of = np.load(f"{WORK}/fc_item_of.npy")
is_gen = np.array([("id" not in it) or (it.get("id") is None) for it in fc_items])[item_of]
a_dp = neuron_out(Xdp, L, w); a_fc = neuron_out(Xfc, L, w)
print("in-sample gate mult: DIok mean %.2f frac>0.5 %.2f | DIerr mean %.2f frac>0.5 %.2f | DDok mean %.3f frac>0.5 %.3f | DDerr mean %.2f | other max %.3f | gen frac>0.1 %.5f max %.2f | prac frac>0.1 %.5f" % (
    a_dp[isDI & ~err].mean(), (a_dp[isDI & ~err] > 0.5).mean(), a_dp[isDI & err].mean(), (a_dp[isDI & err] > 0.5).mean(),
    a_dp[isDD & ~err].mean(), (a_dp[isDD & ~err] > 0.5).mean(), a_dp[isDD & err].mean(), a_dp[isO].max(),
    (a_fc[is_gen] > 0.1).mean(), a_fc[is_gen].max(), (a_fc[~is_gen] > 0.1).mean()))
d_imp = np.load(f"{WORK}/steer_dimp_{L}.npy")
build_edit(L, w, d_imp, alpha=12.0)
print(edits())
