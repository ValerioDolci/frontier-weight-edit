import torch, json, numpy as np
J = 14123
L = 47
alpha, theta, P0 = 10.0, 0.45, 6.0
w = torch.tensor(np.load(WORK + "/probe47_w.npy"), dtype=torch.float32)
mhat = torch.load(WORK + "/push47.pt")["mhat"]
p = torch.load(WORK + "/push47c.pt")["p"]
D0 = torch.load(WORK + "/L47pos.pt"); Y = D0["Y"].float(); R47 = D0["R47"].float()
SC = 0.0478
scale = float(Y.pow(2).mean(1).sqrt().median() * R47.pow(2).mean(1).sqrt().median() / SC)
beta = P0 * scale
nG = f"model.language_model.layers.{L}.mlp.gate_proj.weight"
nU = f"model.language_model.layers.{L}.mlp.up_proj.weight"
nD = f"model.language_model.layers.{L}.mlp.down_proj.weight"
G = W(nG); U = W(nU); D = W(nD)
G[J] = alpha * (w - theta * mhat)
U[J] = mhat
D[:, J] = beta * p
save_edit(nG, G); save_edit(nU, U); save_edit(nD, D)
print(edits())
print("gate row norm", float(G[J].norm()), "up row norm", float(U[J].norm()), "down col norm", float(D[:, J].norm()), "beta", beta)
json.dump({"J": J, "L": L, "alpha": alpha, "theta": theta, "P0": P0, "beta": beta}, open(WORK + "/edit1_params.json", "w"))
