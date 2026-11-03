import numpy as np, torch, json
E = W("model.language_model.embed_tokens.weight")
En = E / E.norm(dim=1, keepdim=True)
R = np.load(f"{WORK}/dp_resid.npy")  # (48,N,d)
M = json.load(open(f"{WORK}/dp_meta.json")); items = M["items"]; meta = M["meta"]
N = len(items)
# 1) token-id of last input token via nearest embedding to resid layer0
r0 = torch.tensor(R[0].astype(np.float32))
sims = (r0 / r0.norm(dim=1, keepdim=True)) @ En.T
best = sims.argmax(1)
vocab = {}
for n in range(N):
    toks = meta[n]["tokens"]
    if toks:
        vocab.setdefault(toks[-1], set()).add(int(best[n]))
print({k: sorted(v) for k, v in list(vocab.items())[:80]})
json.dump({k: sorted(v) for k, v in vocab.items()}, open(f"{WORK}/vocab_partial.json", "w"))
print("max sims sample", sims.max(1).values[:10])
