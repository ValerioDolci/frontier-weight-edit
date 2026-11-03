"""Apply an edit to google/gemma-4-12B-it (pinned revision) and verify every hash.
  python scripts/apply_edit.py --edit edits/e50 --check          # verify only (loads the model on CPU)
  python scripts/apply_edit.py --edit edits/e50 --out gemma-e50   # also write the edited model (safetensors)"""
import argparse, time
from common import load_model, apply_edit

a = argparse.ArgumentParser()
a.add_argument("--edit", required=True); a.add_argument("--out"); a.add_argument("--check", action="store_true")
a.add_argument("--model-path", help="local copy of the base model (default: download from the Hub)")
g = a.parse_args()
t0 = time.time()
model, tok = load_model(g.model_path, device_map="cpu")
names = apply_edit(model, g.edit, verify=True)
print(f"OK: {len(names)} tensors edited, base and edited sha256 verified ({time.time() - t0:.0f} s)")
for n in names: print("  ", n)
if g.out and not g.check:
    model.save_pretrained(g.out); tok.save_pretrained(g.out); print("saved to", g.out)
