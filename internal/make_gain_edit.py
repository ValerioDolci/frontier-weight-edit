"""Write the one-number edit in the public diff format (edit.npz + manifest.json) and check that it is bit-identical to the
tensor that was evaluated. Usage: make_gain_edit.py OUT_DIR ALPHA LAYER NEURON VARIANT_DIR"""
import sys, os, glob, json, hashlib
import numpy as np, torch
from safetensors import safe_open
OUT, A, L, N, VAR = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
M = "<model-dir>"; os.makedirs(OUT, exist_ok=True)
name = f"model.language_model.layers.{L}.mlp.down_proj.weight"
sha = lambda t: hashlib.sha256(t.detach().to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
T = None
for f in sorted(glob.glob(f"{M}/*.safetensors")):
    with safe_open(f, "pt") as h:
        if name in h.keys(): T = h.get_tensor(name)
E = T.clone(); E[:, [N]] = (T[:, [N]].float() * A).to(T.dtype)
used = torch.load(f"{VAR}/{name}.pt")
assert torch.equal(E.view(torch.int16), used.view(torch.int16)), "diverso dal tensore valutato"
vals = E[:, [N]].contiguous().view(torch.int16).numpy()
key = name.replace(".", "__")
np.savez(os.path.join(OUT, "edit.npz"), **{key + "__idx": np.array([N], dtype=np.int64), key + "__vals": vals})
man = [{"tensor": name, "shape": list(T.shape), "axis": 1, "indices": [N], "n_changed": int((E.view(torch.int16) != T.view(torch.int16)).sum()),
        "sha256_base": sha(T), "sha256_edited": sha(E)}]
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(json.dumps(man[0])[:400]); print("vals", vals.shape, vals.dtype, "identico al tensore valutato: OK")
