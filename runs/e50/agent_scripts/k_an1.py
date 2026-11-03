import numpy as np, torch, json
A = load_acts("dp_last")
acts = A["acts"]; meta = A["meta"]; items = A["items"]
N = len(items)
def stack(key):
    return torch.cat([t.float() for t in acts[key]], 0).numpy()
for kind in ["resid", "attn_out", "mlp_out", "mlp_act"]:
    arr = np.stack([stack(f"{kind}.{L}") for L in range(48)], 0).astype(np.float16)  # (48, N, d)
    np.save(f"{WORK}/dp_{kind}.npy", arr)
    print(kind, arr.shape)
json.dump({"meta": meta, "items": items}, open(f"{WORK}/dp_meta.json", "w"))
# logit lens at final layer
E = W("model.language_model.embed_tokens.weight")  # (V, d)
gfin = W("model.language_model.norm.weight")
print("final norm weight stats", gfin.mean().item(), gfin.std().item(), gfin.min().item(), gfin.max().item())
R47 = torch.tensor(np.load(f"{WORK}/dp_resid.npy")[47].astype(np.float32))
def fnorm(x, g, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
h = fnorm(R47, gfin)
logits = h @ E.T
logits = 30 * torch.tanh(logits / 30)
lp = torch.log_softmax(logits, -1)
top = torch.topk(lp, 5, -1)
d = json.load(open(WORK + "/dp.json"))
base = {i: b for (i, p, b, a) in d["DP"]}
for n in range(0, 40):
    i = items[n]["id"]
    print(i, repr(items[n]["it"][-25:]), base.get(i), top.indices[n].tolist(), [round(v, 2) for v in top.values[n].tolist()])
torch.save({"top_idx": top.indices, "top_val": top.values}, f"{WORK}/dp_top.pt")
