import torch, json, numpy as np, time, gc
t0 = time.time()
cfg = dict(J29=4205, J47=14123, lam=100.0, a29s=5.0, th29=0.32, u0=1.5, th47=0.15, P47=14.0)
L29 = torch.load(WORK + "/L29pos.pt")
P = torch.load(WORK + "/L47pos.pt"); Nn = torch.load(WORK + "/new47pos.pt", weights_only=False)
X29 = torch.cat([L29["prac"]["XH"], L29["new"]["XH"]]).float()
X47 = torch.cat([P["XH"], Nn["XH"]]).float()
Y = torch.cat([P["Y"].half(), Nn["Y"].half()]).float(); R47 = torch.cat([P["R47"].half(), Nn["R47"].half()]).float()
lab = np.load(WORK + "/comb_lab_fixed.npy"); wt = np.load(WORK + "/comb_wt_fixed.npy")
# --- L29 probe (full data, uncentered weighted ridge)
X = X29.numpy(); sw = np.sqrt(wt); Xw = X * sw[:, None]; yw = lab * sw
w29 = np.linalg.solve(Xw.T @ Xw + cfg["lam"] * np.eye(3840), Xw.T @ yw)
np.save(WORK + "/w29_full.npy", w29)
e1750 = torch.zeros(3840); e1750[1750] = 1.0 / 0.091
w29t = torch.tensor(w29, dtype=torch.float32)
# --- flag direction
delta = torch.load(WORK + "/flag_delta.pt")["delta"].double()
wpf29 = W("model.language_model.layers.29.post_feedforward_layernorm.weight").double()
sig29 = 0.6015; rms_r29 = 1.995; rms_y29 = float(L29["prac"]["Y"][:4000].float().pow(2).mean(1).sqrt().median())
f_y = (cfg["u0"] * rms_r29 * rms_y29 / sig29) * (delta / wpf29)
print("rms_y29", rms_y29, "|f_y|", float(f_y.norm()))
# --- L29 edit
n = "model.language_model.layers.29.mlp."
G = W(n + "gate_proj.weight"); U = W(n + "up_proj.weight"); D = W(n + "down_proj.weight")
G[cfg["J29"]] = cfg["a29s"] * (w29t - cfg["th29"] * e1750)
U[cfg["J29"]] = e1750
D[:, cfg["J29"]] = f_y.float()
save_edit(n + "gate_proj.weight", G); save_edit(n + "up_proj.weight", U); save_edit(n + "down_proj.weight", D)
del G, U, D; gc.collect()
# --- L47 readout
wpre47 = W("model.language_model.layers.47.pre_feedforward_layernorm.weight").double()
C47 = torch.cov(X47.double().T); C47 += 1e-3 * float(torch.trace(C47)) / 3840 * torch.eye(3840, dtype=torch.float64)
dx = cfg["u0"] * (delta * wpre47)
Cidx = torch.linalg.solve(C47, dx)
g = Cidx / float(dx @ Cidx)
mu47 = X47.double().mean(0)
mhat47 = torch.load(WORK + "/push47.pt")["mhat"].double()
gm = float(g @ mu47)
print("g.mu47", gm, "|g|", float(g.norm()), "g max abs", float(g.abs().max()))
b47 = X47.double() @ mhat47
phi_impl = (X47.double() @ g) - gm * b47
print("implemented phi on unflagged data: mean", float(phi_impl.mean()), "std", float(phi_impl.std()), "max", float(phi_impl.max()), "q99.9", float(phi_impl.abs().quantile(0.999)))
p = torch.load(WORK + "/push47c.pt")["p"]
SC = 0.0478
scale = float(Y.pow(2).mean(1).sqrt().median() * R47.pow(2).mean(1).sqrt().median() / SC)
n = "model.language_model.layers.47.mlp."
G = W(n + "gate_proj.weight"); U = W(n + "up_proj.weight"); D = W(n + "down_proj.weight")
G[cfg["J47"]] = (10.0 * (g - (gm + cfg["th47"]) * mhat47)).float()
U[cfg["J47"]] = (mhat47 / 10.0).float()
D[:, cfg["J47"]] = cfg["P47"] * scale * p
save_edit(n + "gate_proj.weight", G); save_edit(n + "up_proj.weight", U); save_edit(n + "down_proj.weight", D)
print(edits())
cfg.update(dict(gm=gm, scale=scale, rms_y29=rms_y29))
json.dump(cfg, open(WORK + "/edit2_cfg.json", "w"))
torch.save({"g": g.float(), "f_y": f_y.float(), "w29": w29t, "delta": delta.float()}, WORK + "/edit2_parts.pt")
print("done", time.time() - t0)
