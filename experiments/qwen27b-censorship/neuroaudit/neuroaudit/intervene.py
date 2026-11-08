"""Interventions: neuron ablation/scaling at run time, residual-direction edits, and writing a neuron edit into the weights."""
from __future__ import annotations
import contextlib, json, os, glob, shutil
import torch
from .model import decoder_layers, down_proj

Neurons = dict  # {layer_index: [neuron indices]}


def parse_neurons(spec) -> dict[int, list[int]]:
    """Accept {layer: [j...]} (keys may be strings, as in JSON) or a list of (layer, j) pairs."""
    if isinstance(spec, dict):
        return {int(l): sorted({int(j) for j in js}) for l, js in spec.items()}
    out: dict[int, list[int]] = {}
    for l, j in spec:
        out.setdefault(int(l), []).append(int(j))
    return {l: sorted(set(js)) for l, js in out.items()}


class NeuronEditor:
    """Installs one pre-hook per layer on the MLP down projection. Use .set(neurons, scale) or the .scaled() context manager.
    scale = 0 switches the neurons off; any other value multiplies their activation (all positions)."""

    def __init__(self, model):
        self.spec: dict[int, torch.Tensor] = {}
        self.scale = 1.0
        self.handles = []
        for l, layer in enumerate(decoder_layers(model)):
            self.handles.append(down_proj(layer).register_forward_pre_hook(self._hook(l)))

    def _hook(self, l):
        def pre(m, args):
            idx = self.spec.get(l)
            if idx is None:
                return None
            x = args[0].clone(); x[..., idx.to(x.device)] *= self.scale
            return (x,) + tuple(args[1:])
        return pre

    def set(self, neurons=None, scale: float = 0.0):
        self.spec = {l: torch.tensor(js) for l, js in parse_neurons(neurons or {}).items()}; self.scale = scale

    def clear(self):
        self.spec = {}; self.scale = 1.0

    @contextlib.contextmanager
    def scaled(self, neurons, scale: float = 0.0):
        self.set(neurons, scale)
        try:
            yield self
        finally:
            self.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


class DirectionEditor:
    """Edits the residual stream leaving decoder layer `layer` along a unit direction d (all positions):
    mode 'ablate': x ← x − (x·d − target)·d (projection reset to `target`, e.g. the mean of a reference group);
    mode 'add':    x ← x + alpha·d."""

    def __init__(self, model, layer: int):
        self.cfg = None
        self.h = decoder_layers(model)[layer].register_forward_hook(self._hook)

    def _hook(self, m, args, out):
        if self.cfg is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        d, mode, v = self.cfg; d = d.to(h.device, h.dtype)
        h = h - ((h @ d) - v).unsqueeze(-1) * d if mode == "ablate" else h + v * d
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    @contextlib.contextmanager
    def edit(self, direction, mode: str, value: float):
        d = torch.as_tensor(direction, dtype=torch.float32); d = d / d.norm()
        self.cfg = (d, mode, float(value))
        try:
            yield self
        finally:
            self.cfg = None

    def remove(self):
        self.h.remove()


def bake(model_dir: str, neurons, out_dir: str, weight_name: str = "mlp.down_proj.weight", layer_key: str = "layers.{l}.",
         skip=("mtp.", "visual.", "vision_tower.", "vision_model.")) -> dict:
    """Write a neuron ablation into a COPY of a safetensors checkpoint: zero the input columns of the down projection.
    Copies every other file unchanged and writes EDIT_MANIFEST.json (layers, columns, tensor names). Tensors of multi-token
    prediction or vision modules (names containing `skip`) are left untouched; check EDIT_MANIFEST.json for exactly one
    tensor per edited layer."""
    from safetensors.torch import load_file, save_file
    spec = parse_neurons(neurons); os.makedirs(out_dir, exist_ok=True); touched = []
    for f in sorted(glob.glob(os.path.join(model_dir, "*"))):
        if os.path.isdir(f):
            continue
        dst = os.path.join(out_dir, os.path.basename(f))
        if not f.endswith(".safetensors"):
            shutil.copy(f, dst); continue
        t = load_file(f); changed = False
        for k in list(t):
            if not k.endswith(weight_name) or any(x in k for x in skip):      # only the text decoder (not MTP/vision heads)
                continue
            for l, js in spec.items():
                if layer_key.format(l=l) in k:
                    t[k] = t[k].clone(); t[k][:, js] = 0; changed = True; touched.append({"tensor": k, "layer": l, "columns": js})
        save_file(t, dst, metadata={"format": "pt"}) if changed else shutil.copy(f, dst)
    manifest = {"source": os.path.basename(os.path.abspath(model_dir)), "edit": "zero down_proj input columns", "neurons": {str(l): js for l, js in spec.items()}, "tensors": touched}
    json.dump(manifest, open(os.path.join(out_dir, "EDIT_MANIFEST.json"), "w"), indent=1)
    return manifest
