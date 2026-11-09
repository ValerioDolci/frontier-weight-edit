"""Write the full down_proj tensor of the given layers with the columns of some neurons multiplied by ALPHA (a gain on the
neuron's output), in the format read by hf_translate.py --edits. This is the whole 'one number' edit.
Usage: scale_cols.py OUT_DIR ALPHA LAYER:N [LAYER:N ...]"""
import sys, os, glob, torch
from safetensors import safe_open
OUT, A = sys.argv[1], float(sys.argv[2]); M = "<model-dir>"; os.makedirs(OUT, exist_ok=True)
by = {}
for s in sys.argv[3:]:
    l, n = s.split(":"); by.setdefault(int(l), []).append(int(n))
where = {}
for f in sorted(glob.glob(f"{M}/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys(): where[k] = f
for L, NS in by.items():
    n = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    with safe_open(where[n], "pt") as h: T = h.get_tensor(n).clone()
    before = T[:, NS].float().norm(dim=0)
    T[:, NS] = (T[:, NS].float() * A).to(T.dtype)
    torch.save(T, f"{OUT}/{n}.pt")
    print("scritto", OUT, L, NS, "x", A, "norma colonna", [round(x, 4) for x in before.tolist()], "->", [round(x, 4) for x in T[:, NS].float().norm(dim=0).tolist()])
