"""Turn a session's submitted tensors (edits/*.pt, full bf16 tensors) into a minimal diff + manifest with sha256.
  python harness/make_diff.py RUN_DIR/server/edits edits/NAME --model-path /path/to/gemma-4-12B-it"""
import argparse, glob, json, os, sys
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from common import tensor_sha256  # noqa: E402
a = argparse.ArgumentParser(); a.add_argument("edits"); a.add_argument("out"); a.add_argument("--model-path", required=True); g = a.parse_args()
where = {}
for f in glob.glob(os.path.join(g.model_path, "*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys(): where[k] = f
out, man = {}, []
for f in sorted(glob.glob(os.path.join(g.edits, "*.pt"))):
    n = os.path.basename(f)[:-3]; e = torch.load(f).to(torch.bfloat16)
    with safe_open(where[n], "pt") as s: o = s.get_tensor(n)
    d = e != o
    rows = torch.nonzero(d.any(1)).flatten().tolist(); cols = torch.nonzero(d.any(0)).flatten().tolist()
    axis, idx = (1, cols) if len(cols) < len(rows) else (0, rows)
    vals = e[:, idx] if axis == 1 else e[idx, :]
    key = n.replace(".", "__"); out[key + "__idx"] = np.array(idx); out[key + "__vals"] = vals.contiguous().view(torch.int16).numpy()
    man.append({"tensor": n, "shape": list(o.shape), "axis": axis, "indices": idx, "n_changed": int(d.sum()),
                "sha256_base": tensor_sha256(o), "sha256_edited": tensor_sha256(e)})
os.makedirs(g.out, exist_ok=True)
np.savez_compressed(os.path.join(g.out, "edit.npz"), **out); json.dump(man, open(os.path.join(g.out, "manifest.json"), "w"), indent=1)
print(f"{len(man)} tensors -> {g.out}")
