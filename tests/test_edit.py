"""pytest tests/            -> checks the shipped edits (no model needed)
WSP_MODEL=/path/to/gemma-4-12B-it pytest tests/   -> also rebuilds every edited tensor from the base weights and checks sha256
WSP_GPU=1 (with WSP_MODEL)                        -> also runs one known DI sentence through original vs edited model"""
import glob, json, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
EDITS = sorted(glob.glob(os.path.join(ROOT, "edits", "*", "manifest.json")))

@pytest.mark.parametrize("man_path", EDITS)
def test_manifest_matches_diff(man_path):
    man = json.load(open(man_path)); z = np.load(os.path.join(os.path.dirname(man_path), "edit.npz"))
    for m in man:
        key = m["tensor"].replace(".", "__"); idx, vals = z[key + "__idx"], z[key + "__vals"]
        assert list(idx) == m["indices"]
        rows, cols = m["shape"]
        assert vals.shape == ((len(idx), cols) if m["axis"] == 0 else (rows, len(idx)))
        assert m["n_changed"] <= vals.size
        assert m["tensor"].startswith("model.language_model.layers.")

@pytest.mark.skipif(not os.environ.get("WSP_MODEL"), reason="set WSP_MODEL to a local copy of google/gemma-4-12B-it")
@pytest.mark.parametrize("man_path", EDITS)
def test_rebuild_from_base(man_path):
    import torch
    from safetensors import safe_open
    from common import tensor_sha256
    where = {}
    for f in glob.glob(os.path.join(os.environ["WSP_MODEL"], "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys(): where[k] = f
    man = json.load(open(man_path)); z = np.load(os.path.join(os.path.dirname(man_path), "edit.npz"))
    for m in man:
        with safe_open(where[m["tensor"]], "pt") as s: t = s.get_tensor(m["tensor"])
        assert tensor_sha256(t) == m["sha256_base"], "base weights differ from the pinned revision"
        key = m["tensor"].replace(".", "__"); idx = torch.as_tensor(z[key + "__idx"], dtype=torch.long)
        v = torch.from_numpy(z[key + "__vals"].astype(np.int16)).view(torch.bfloat16)
        if m["axis"] == 0: t[idx, :] = v
        else: t[:, idx] = v
        assert tensor_sha256(t) == m["sha256_edited"]

@pytest.mark.skipif(not (os.environ.get("WSP_MODEL") and os.environ.get("WSP_GPU")), reason="needs WSP_MODEL and WSP_GPU=1")
def test_known_fix_e50():
    from common import load_model, apply_edit, translate
    s = ["They had been driving for six hours when the car broke down."]
    model, tok = load_model(os.environ["WSP_MODEL"])
    before = translate(model, tok, s)[0]
    apply_edit(model, os.path.join(ROOT, "edits", "e50"))
    after = translate(model, tok, s)[0]
    assert "per sei ore" in before and "da sei ore" in after, (before, after)
