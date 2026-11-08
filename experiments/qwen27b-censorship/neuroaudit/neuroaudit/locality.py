"""Causal check of a neuron set: switch off the top-k neurons (by attribution on half A) and measure the margin on the
held-out half B, per category, against k random neurons with the same per-layer distribution."""
from __future__ import annotations
import random
import numpy as np
from .attribution import margin, top_neurons
from .intervene import NeuronEditor
from .model import n_neurons


def curve(model, items: list[dict], attr: np.ndarray, ks=(1, 2, 4, 8, 16, 32, 64, 128), sign: str = "neg", seed: int = 0, editor: NeuronEditor | None = None) -> dict:
    ed = editor or NeuronEditor(model); B = [it for it in items if it.get("half") == "B"]; cats = np.array([it["cat"] for it in B])
    rng = random.Random(seed); NI = n_neurons(model)
    base = np.array([margin(model, it) for it in B])
    res = {"base_mean": {c: float(base[cats == c].mean()) for c in set(cats)}, "base_right": {c: float((base[cats == c] > 0).mean()) for c in set(cats)}, "curve": []}
    order = top_neurons(attr, max(ks), sign)
    for k in ks:
        top = order[:k]
        for kind, lst in (("top", top), ("rand", [(l, rng.randrange(NI)) for l, _ in top])):
            with ed.scaled(lst, 0.0):
                m = np.array([margin(model, it) for it in B])
            row = {"k": k, "kind": kind}
            for c in set(cats):
                row[f"d_{c}"] = float((m - base)[cats == c].mean()); row[f"right_{c}"] = float((m[cats == c] > 0).mean())
            res["curve"].append(row)
    if editor is None:
        ed.remove()
    return res
