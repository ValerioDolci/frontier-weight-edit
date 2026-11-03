import numpy as np, torch, json
d3 = json.load(open(f"{WORK}/dec_r3.json")); dec3 = {int(k): v for k, v in d3["dec3"].items()}
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
cfg = json.load(open(f"{WORK}/cfg_E2a.json"))
L = 31
wPm = np.load(f"{WORK}/E2a_w_{L}_Pm.npy"); cL = np.load(f"{WORK}/E2a_c_{L}.npy")
th, Dl = cfg[f"{L}_Pm"]["theta"], cfg[f"{L}_Pm"]["Dl"]
wpa = W(f"model.language_model.layers.{L}.post_attention_layernorm.weight").float()
wpre = W(f"model.language_model.layers.{L}.pre_feedforward_layernorm.weight").float()
def x_r3(i):
    h = A3[f"resid.{L-1}"][i].float() + rmsn(A3[f"attn_out.{L}"][i].float(), wpa)
    return rmsn(h, wpre).numpy()
# E2a direction used for Pm at L31 = mean DIR - mean DIW (mid) over even r2 items
isfirst2 = lambda i: dec2[i] == M2[i]["n_prompt_tokens"] - 1
R2 = lambda i: A2[f"resid.{L}"][i].float()
Wm = [R2(i)[dec2[i]] for i in range(55) if i % 2 == 0 and not isfirst2(i)]
Rm = [R2(i)[dec2[i]] for i in range(55, 259) if i in dec2 and i % 2 == 0 and not isfirst2(i)]
dm = (torch.stack(Rm).mean(0) - torch.stack(Wm).mean(0)).numpy(); dhat = dm / np.linalg.norm(dm)
p2 = [float(R2(i)[dec2[i]].numpy() @ dhat) for i in range(259) if i in dec2 and not isfirst2(i)]
p3, cl3 = [], []
for k in range(120):
    r = dec3[k]
    if r == M3[k]["n_prompt_tokens"] - 1: continue
    x = x_r3(k)[r]
    s = float(x @ wPm / (x @ cL)); cl = float(np.clip((s - th) / Dl, 0, 1))
    p3.append(float(A3[f"resid.{L}"][k][r].float().numpy() @ dhat)); cl3.append(cl)
p3 = np.array(p3); cl3 = np.array(cl3)
print(f"r2 (original) DI mid decision rows: proj mean {np.mean(p2):.2f} sd {np.std(p2):.2f}")
print(f"r3 DI mid decision rows: n={len(p3)}; clamp>0.9: {np.sum(cl3>0.9)}; proj mean (clamp>0.9) {p3[cl3>0.9].mean():.2f}; proj mean (clamp<0.1) {p3[cl3<0.1].mean() if np.any(cl3<0.1) else float('nan'):.2f}")
print(f"expected push along dhat if edit active: ~ {cfg[f'{L}_Pm']['vnorm']:.2f} (|v| in mlp_out units) -> resid ~ kappa*|d| = {3*np.linalg.norm(dm):.1f}")
