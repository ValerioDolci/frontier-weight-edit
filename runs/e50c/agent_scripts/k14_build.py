import numpy as np, torch, json
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
NSYS = 45
L = 31
P = f"model.language_model.layers.{L}"
w = np.load(f"{WORK}/det_w_L{L}.npy").astype(np.float32)
c = np.load(f"{WORK}/const_c_L{L}.npy").astype(np.float32)
s_L = {30: 0.7227, 31: 0.6993, 32: 0.5546, 33: 0.629, 34: 0.75}[L]

# --- steering direction: DIR - DIW at resid_L decision rows ---
R = lambda i: A2[f"resid.{L}"][i].float()
diw = torch.stack([R(i)[dec2[i]] for i in range(55)]).numpy()
dir_ = torch.stack([R(i)[dec2[i]] for i in range(55, 259) if i in dec2]).numpy()
delta = dir_.mean(0) - diw.mean(0)
# matched by position type (first-token decision vs mid)
first = lambda i: dec2[i] == M2[i]["n_prompt_tokens"] - 1
f_w = [k for k, i in enumerate(range(55)) if first(i)]; m_w = [k for k, i in enumerate(range(55)) if not first(i)]
ir = [i for i in range(55, 259) if i in dec2]
f_r = [k for k, i in enumerate(ir) if first(i)]; m_r = [k for k, i in enumerate(ir) if not first(i)]
d_first = dir_[f_r].mean(0) - diw[f_w].mean(0); d_mid = dir_[m_r].mean(0) - diw[m_w].mean(0)
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
print(f"|delta|={np.linalg.norm(delta):.2f}  n first: DIW {len(f_w)} DIR {len(f_r)}; cos(delta,d_first)={cos(delta,d_first):.3f} cos(delta,d_mid)={cos(delta,d_mid):.3f} cos(first,mid)={cos(d_first,d_mid):.3f}")
delta_m = 0.5 * (d_first + d_mid)
np.save(f"{WORK}/delta_L{L}.npy", delta_m)

# --- choose neurons: smallest proxy contribution on sample rows ---
Wg = W(f"{P}.mlp.gate_proj.weight").float(); Wu = W(f"{P}.mlp.up_proj.weight").float(); Wd = W(f"{P}.mlp.down_proj.weight").float()
Xs = torch.cat([mlp_in(L, i)[NSYS:] for i in range(0, len(I2), 5)])
gx = Xs @ Wg.T; ux = Xs @ Wu.T
proxy = (torch.relu(gx) * ux.abs()).mean(0) * Wd.norm(dim=0)
order = torch.argsort(proxy)
j1, j2 = int(order[0]), int(order[1])
print("chosen neurons", j1, j2, "proxy", proxy[j1].item(), proxy[j2].item(), "median proxy", proxy.median().item())
del gx, ux

# --- design ---
wpf = W(f"{P}.post_feedforward_layernorm.weight").float().numpy()
rms_m = 0.153
kappa = 2.0
target_post = kappa * delta_m / s_L               # desired change of post_ff output
v = rms_m * target_post / np.maximum(wpf, 1e-6)
v[wpf < 0.1] = 0.0
print(f"|v|={np.linalg.norm(v):.3f}  (|m| at dec ~ {rms_m*np.sqrt(3840):.2f})")
a, theta, Dl = 20.0, 0.2, 0.3
b = 1.0 / (a * Dl)
g1 = a * (w - theta * c); g2 = a * (w - (theta + Dl) * c); u = b * c
Wg2, Wu2, Wd2 = Wg.clone(), Wu.clone(), Wd.clone()
Wg2[j1] = torch.from_numpy(g1); Wg2[j2] = torch.from_numpy(g2)
Wu2[j1] = torch.from_numpy(u); Wu2[j2] = torch.from_numpy(u)
Wd2[:, j1] = torch.from_numpy(v); Wd2[:, j2] = torch.from_numpy(-v)
save_edit(f"{P}.mlp.gate_proj.weight", Wg2)
save_edit(f"{P}.mlp.up_proj.weight", Wu2)
save_edit(f"{P}.mlp.down_proj.weight", Wd2)
print("edits:", edits())
json.dump({"L": L, "j1": j1, "j2": j2, "a": a, "theta": theta, "Dl": Dl, "kappa": kappa}, open(f"{WORK}/edit_E1.json", "w"))

# --- predicted firing (clamp value) by category ---
def clampv(s): return np.clip((s - theta) / Dl, 0, 1)
cat = lambda i: "DIW" if i < 55 else ("DIR" if i < 259 else ("DD" if i < 389 else ("DURP" if i < 449 else ("PERF" if i < 509 else "GEN"))))
stats = {}
for i in range(len(I2)):
    xi = mlp_in(L, i).numpy()
    sc = xi @ w / (xi @ c)
    cv = clampv(sc)
    k = cat(i)
    st = stats.setdefault(k, {"rows": 0, "fire_rows": 0, "items_fire": 0, "dec_fire": []})
    rr = range(NSYS, len(sc))
    st["rows"] += len(rr); st["fire_rows"] += int((cv[NSYS:] > 0.05).sum())
    st["items_fire"] += int((cv[NSYS:] > 0.05).any())
    if i in dec2: st["dec_fire"].append(float(cv[dec2[i]]))
for k, st in stats.items():
    df = np.array(st["dec_fire"]) if st["dec_fire"] else np.array([np.nan])
    print(f"{k:5s} rows={st['rows']} fire_rows={st['fire_rows']} items_with_any_fire={st['items_fire']}  decision clamp: mean={np.nanmean(df):.2f} frac>0.5={np.nanmean(df>0.5):.2f}")
