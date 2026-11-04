"""Checkpoint HF bf16 completo con l'impianto applicato (per GGUF/llama.cpp). Riscrive solo gli shard toccati, gli altri sono link.
Uso: make_hf_edit.py implant.safetensors OUT_DIR"""
import json, os, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
BF = "/mnt/archivio/models/qwen3.8-27b-bf16"; PFX = "model.language_model.layers."
imp_f, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
imp = safe_open(imp_f, "pt"); layers = sorted({int(k.split(".")[0][1:]) for k in imp.keys()})
wm = json.load(open(os.path.join(BF, "model.safetensors.index.json")))["weight_map"]
touched = {wm[f"{PFX}{l}.mlp.{p}.weight"] for l in layers for p in ("gate_proj", "up_proj", "down_proj")}
for f in os.listdir(BF):
    if f in touched or f.startswith("."): continue
    if not os.path.lexists(os.path.join(out, f)): os.symlink(os.path.join(BF, f), os.path.join(out, f))
for sh in sorted(touched):
    with safe_open(os.path.join(BF, sh), "pt") as f:
        meta = f.metadata(); T = {k: f.get_tensor(k) for k in f.keys()}
    for l in layers:
        k = f"{PFX}{l}.mlp."; idx = imp.get_tensor(f"L{l}.idx").long()
        if k + "gate_proj.weight" in T: T[k + "gate_proj.weight"][idx] = imp.get_tensor(f"L{l}.gate_rows")
        if k + "up_proj.weight" in T: T[k + "up_proj.weight"][idx] = imp.get_tensor(f"L{l}.up_rows")
        if k + "down_proj.weight" in T: T[k + "down_proj.weight"][:, idx] = imp.get_tensor(f"L{l}.down_cols")
    save_file(T, os.path.join(out, sh), metadata=meta); print("riscritto", sh, flush=True)
print("ok", out, "layer", layers)
