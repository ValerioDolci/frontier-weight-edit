import torch, numpy as np
Ld = 26
win = torch.stack([W(f"model.language_model.layers.{l}.input_layernorm.weight").abs() for l in range(Ld + 1, 48)])
wpf = torch.stack([W(f"model.language_model.layers.{l}.pre_feedforward_layernorm.weight").abs() for l in range(Ld + 1, 47)])
w47 = W("model.language_model.layers.47.pre_feedforward_layernorm.weight").abs()
# relative to per-layer median
win_r = win / win.median(dim=1, keepdim=True).values
wpf_r = wpf / wpf.median(dim=1, keepdim=True).values
read_mid = torch.maximum(win_r.max(0).values, wpf_r.max(0).values)
w47_r = w47 / w47.median()
score = w47_r / (read_mid + 1e-3)
D0 = torch.load(WORK + "/L47pos.pt"); XH = D0["XH"].float()
sd47 = XH.std(0); mu47 = XH.mean(0)
order = torch.argsort(score, descending=True)
print("top dims: (dim, w47_rel, maxread_mid_rel, xhat47 mean, sd)")
for i in order[:25].tolist():
    print(i, round(float(w47_r[i]), 3), round(float(read_mid[i]), 3), round(float(mu47[i]), 3), round(float(sd47[i]), 3))
print("median read_mid", float(read_mid.median()), "min", float(read_mid.min()))
# how many dims have read_mid < 0.1
print("dims with read_mid<0.2:", int((read_mid < 0.2).sum()), "<0.1:", int((read_mid < 0.1).sum()))
