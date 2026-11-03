import torch, numpy as np, time, gc
t0 = time.time()
eg = torch.load(WORK + "/readGram_eig.pt"); ev, V = eg["ev"], eg["V"]
P = torch.load(WORK + "/L47pos.pt"); Nn = torch.load(WORK + "/new47pos.pt", weights_only=False)
XH47 = torch.cat([P["XH"], Nn["XH"]]).float().double()
H47 = torch.cat([P["H"].float()]).double() if "H" in P else None
wpre47 = W("model.language_model.layers.47.pre_feedforward_layernorm.weight").double()
wpf29 = W("model.language_model.layers.29.post_feedforward_layernorm.weight").double()
C47 = torch.cov(XH47.T)
lamC = 1e-3 * float(torch.trace(C47)) / 3840
C47r = C47 + lamC * torch.eye(3840, dtype=torch.float64)
Ci = torch.linalg.inv(C47r)
for k in [20, 50, 100, 200, 400]:
    Vk = V[:, :k]
    # visibility: x_hat shift = (delta * wpre47) / rms(h47) ; take rms(h47) as 1 unit for relative comparison
    Mk = Vk * wpre47[:, None]            # 3840 x k : delta -> x_hat shift
    S = Mk.T @ Ci @ Mk                     # k x k
    es, U = torch.linalg.eigh(S)
    u = U[:, -1]; delta = Vk @ u
    snr2 = float(es[-1])
    read = float(delta @ (torch.load(WORK + "/readGram_30_47.pt").double() @ delta))
    cost = float(((delta / wpf29) ** 2).sum().sqrt())
    print(f"k={k}: best SNR^2/unit {snr2:.3e}  read-gain {read:.4f}  (median eig {float(ev[1920]):.3f})  L29 y-cost |delta/wpf29| {cost:.2f}  |delta*wpre47| {float((delta*wpre47).norm()):.3f}")
# typical rms of h47 (pre-norm residual) at L47, to convert
r = XH47[:2000]
print("x_hat47 rms", float(r.pow(2).mean().sqrt()))
# rms of h47 itself: need H; reconstruct from L47pos 'H'
Hh = P["H"][:2000].float()
print("h47 rms median", float(Hh.pow(2).mean(1).sqrt().median()))
Y29 = torch.load(WORK + "/L29pos.pt")["prac"]["Y"][:2000].float()
print("y29 rms median", float(Y29.pow(2).mean(1).sqrt().median()), "wpf29 stats", float(wpf29.mean()), float(wpf29.min()), float(wpf29.max()))
