"""Builds an NVFP4 checkpoint (compressed-tensors, servable with vLLM) whose MLPs in the given layers are bf16:
either the originals (base reference) or with the implant applied (tampered model).
Usage: make_ckpt.py --out DIR --layers 36,38,... [--implant implant.safetensors]"""
import argparse, json, os, struct, shutil
import torch
from safetensors import safe_open

NV = "<ws>/models/qwen3.8-27b-nvfp4"; BF = "<ws>/models/qwen3.8-27b-bf16"
PFX = "model.language_model.layers."
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--layers", required=True)
ap.add_argument("--implant"); ap.add_argument("--fp8", action="store_true", help="MLP toccate in FP8 per canale (come i layer 56-63) invece di bf16"); a = ap.parse_args()
LAY = [int(x) for x in a.layers.split(",")]
os.makedirs(a.out, exist_ok=True)
for f in os.listdir(NV):
    if f in ("model.safetensors", "config.json", "model.safetensors.index.json") or f.startswith("."): continue
    dst = os.path.join(a.out, f)
    if not os.path.lexists(dst): os.symlink(os.path.join(NV, f), dst)

PROJ = ("gate_proj", "up_proj", "down_proj")
drop = {f"{PFX}{l}.mlp.{p}.{s}" for l in LAY for p in PROJ for s in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")}
bidx = json.load(open(os.path.join(BF, "model.safetensors.index.json")))["weight_map"]
imp = safe_open(a.implant, "pt") if a.implant else None
new = {}
for l in LAY:
    W = {}
    for p in PROJ:
        k = f"{PFX}{l}.mlp.{p}.weight"
        with safe_open(os.path.join(BF, bidx[k]), "pt") as f: W[p] = f.get_tensor(k).clone()
    if imp is not None:
        idx = imp.get_tensor(f"L{l}.idx").long()
        W["gate_proj"][idx] = imp.get_tensor(f"L{l}.gate_rows"); W["up_proj"][idx] = imp.get_tensor(f"L{l}.up_rows")
        W["down_proj"][:, idx] = imp.get_tensor(f"L{l}.down_cols")
    for p in PROJ:
        w = W[p].to(torch.bfloat16).float()
        if a.fp8:
            sc = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12).to(torch.bfloat16)
            new[f"{PFX}{l}.mlp.{p}.weight"] = (w / sc.float()).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            new[f"{PFX}{l}.mlp.{p}.weight_scale"] = sc.contiguous()
        else: new[f"{PFX}{l}.mlp.{p}.weight"] = w.to(torch.bfloat16).contiguous()

src = os.path.join(NV, "model.safetensors")
with open(src, "rb") as fi:
    n = struct.unpack("<Q", fi.read(8))[0]; hdr = json.loads(fi.read(n)); base = 8 + n
meta = hdr.pop("__metadata__", None)
assert drop <= set(hdr), "chiavi attese mancanti"
out_hdr, off, plan = {}, 0, []
for k in sorted(set(hdr) - drop):
    b, e = hdr[k]["data_offsets"]; out_hdr[k] = dict(dtype=hdr[k]["dtype"], shape=hdr[k]["shape"], data_offsets=[off, off + e - b])
    plan.append(("copy", base + b, e - b)); off += e - b
for k in sorted(new):
    t = new[k]; nb = t.numel() * t.element_size()
    out_hdr[k] = dict(dtype="F8_E4M3" if t.dtype == torch.float8_e4m3fn else "BF16", shape=list(t.shape), data_offsets=[off, off + nb]); plan.append(("new", k, nb)); off += nb
if meta: out_hdr["__metadata__"] = meta
hb = json.dumps(out_hdr, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
with open(src, "rb") as fi, open(os.path.join(a.out, "model.safetensors"), "wb") as fo:
    fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
    for kind, x, nb in plan:
        if kind == "copy":
            fi.seek(x); left = nb
            while left: chunk = fi.read(min(left, 1 << 26)); fo.write(chunk); left -= len(chunk)
        else:
            t = new[x]; fo.write((t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t.view(torch.int16)).numpy().tobytes())

idx = json.load(open(os.path.join(NV, "model.safetensors.index.json")))
idx["weight_map"] = {k: v for k, v in idx["weight_map"].items() if k not in drop}
for k in new: idx["weight_map"][k] = "model.safetensors"
json.dump(idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
cfg = json.load(open(os.path.join(NV, "config.json")))
mods = [f"{PFX}{l}.mlp.{p}" for l in LAY for p in PROJ]
if a.fp8: cfg["quantization_config"]["config_groups"]["group_0"]["targets"] += mods
else: cfg["quantization_config"]["ignore"] += mods
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
print("ok", a.out, "fp8" if a.fp8 else "bf16", "tensori nuovi:", len(new), "rimossi:", len(drop))
