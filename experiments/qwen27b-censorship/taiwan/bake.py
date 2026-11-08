"""Writes the intervention into the weights: a new model folder with links to the original files and modified copies of the
shards containing the touched down_proj columns (zeroed). Usage: bake.py VARIANTS.json NAME OUT_DIR"""
import json, sys, os, glob, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
M = "models/Qwen3.8-27B"; V = json.load(open(sys.argv[1]))[sys.argv[2]]["neurons"]; OUT = sys.argv[3]; os.makedirs(OUT, exist_ok=True)
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
targets = {f"model.language_model.layers.{l}.mlp.down_proj.weight": js for l, js in V.items()}
shards = {idx[k] for k in targets}
for f in os.listdir(M):
    src = os.path.join(M, f); dst = os.path.join(OUT, f)
    if f in shards or os.path.exists(dst): continue
    os.symlink(src, dst)
n = 0
for sh in sorted(shards):
    with safe_open(os.path.join(M, sh), "pt") as h:
        meta = h.metadata(); T = {k: h.get_tensor(k) for k in h.keys()}
    for k, js in targets.items():
        if idx[k] == sh: T[k] = T[k].clone(); T[k][:, js] = 0; n += len(js)
    save_file(T, os.path.join(OUT, sh), metadata=meta); print("scritto", sh, flush=True)
json.dump({"base": M, "variant": sys.argv[2], "zeroed_down_proj_columns": V, "n": n}, open(os.path.join(OUT, "EDIT_MANIFEST.json"), "w"), indent=1)
print("FATTO, colonne azzerate:", n)
