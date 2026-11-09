"""Write the full gate/up/down tensors of one layer with some neurons zeroed (the 'switched off' arms), in the format read by
hf_translate.py --edits. Usage: ablate_one.py OUT_DIR LAYER N1 [N2 ...]"""
import sys, os, glob, torch
from safetensors import safe_open
OUT, L, NS = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]]; M = "<model-dir>"; os.makedirs(OUT, exist_ok=True)
where = {}
for f in sorted(glob.glob(f"{M}/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys(): where[k] = f
for m in ("gate", "up", "down"):
    n = f"model.language_model.layers.{L}.mlp.{m}_proj.weight"
    with safe_open(where[n], "pt") as h: T = h.get_tensor(n).clone()
    if m == "down": T[:, NS] = 0
    else: T[NS] = 0
    torch.save(T, f"{OUT}/{n}.pt")
print("scritto", OUT, L, NS)
