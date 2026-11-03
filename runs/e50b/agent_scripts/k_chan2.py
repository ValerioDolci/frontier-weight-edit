import torch, numpy as np, time, gc
t0 = time.time()
torch.set_num_threads(max(1, torch.get_num_threads()))
D = torch.zeros(3840, 3840, dtype=torch.float32)
per_layer_tr = {}
def addread(Wm, wnorm):
    global D
    M = Wm * wnorm[None, :]          # rows read normalized input scaled by norm weights
    G = M.T @ M
    G = G / torch.trace(G) * 3840.0   # normalize each read matrix to mean eigenvalue 1
    D += G
for l in range(30, 48):
    pre = f"model.language_model.layers.{l}."
    win = W(pre + "input_layernorm.weight")
    names_l = [n for n in names(pre + "self_attn.") if n.endswith("proj.weight") and "o_proj" not in n]
    for n in names_l:
        addread(W(n), win)
    if l < 47:
        wpf = W(pre + "pre_feedforward_layernorm.weight")
        for n in ["mlp.gate_proj.weight", "mlp.up_proj.weight"]:
            addread(W(pre + n), wpf)
    gc.collect()
    print("layer", l, "done", round(time.time() - t0, 1))
torch.save(D, WORK + "/readGram_30_47.pt")
ev, V = torch.linalg.eigh(D.double())
print("eigvals: min", float(ev[0]), "q01", float(ev[38]), "q10", float(ev[384]), "median", float(ev[1920]), "max", float(ev[-1]))
print("smallest 10:", [round(float(x), 4) for x in ev[:10]])
torch.save({"ev": ev, "V": V}, WORK + "/readGram_eig.pt")
