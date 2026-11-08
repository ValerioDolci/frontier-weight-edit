"""Capture, at the LAST prompt token (the moment the model decides how to answer), the activations of every MLP neuron
and the residual stream after every decoder layer. Writes two memory-mapped .npy files (fp16) so large batteries fit on disk."""
from __future__ import annotations
import json, os
import numpy as np, torch
from .model import decoder_layers, down_proj, input_device, chat_prompt


@torch.no_grad()
def capture(tok, model, questions: list[str], out_dir: str, system: str | None = None, neurons: bool = True, residual: bool = True,
            layers: list[int] | None = None, log_every: int = 200) -> dict:
    L = decoder_layers(model); NL = len(L); NI = down_proj(L[0]).in_features; H = down_proj(L[0]).out_features
    keep = layers if layers is not None else list(range(NL)); os.makedirs(out_dir, exist_ok=True); n = len(questions)
    A = np.lib.format.open_memmap(f"{out_dir}/acts.npy", mode="w+", dtype=np.float16, shape=(n, len(keep), NI)) if neurons else None
    R = np.lib.format.open_memmap(f"{out_dir}/resid.npy", mode="w+", dtype=np.float16, shape=(n, len(keep), H)) if residual else None
    cap: dict = {}; hooks = []
    for k, l in enumerate(keep):
        if neurons:
            hooks.append(down_proj(L[l]).register_forward_pre_hook(lambda m, a, k=k: cap.__setitem__(("a", k), a[0][0, -1].float().cpu())))
        if residual:
            hooks.append(L[l].register_forward_hook(lambda m, a, o, k=k: cap.__setitem__(("r", k), (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu())))
    try:
        for i, q in enumerate(questions):
            ids = tok(chat_prompt(tok, q, system), return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device(model))
            model(input_ids=ids, logits_to_keep=1)
            if neurons:
                A[i] = torch.stack([cap[("a", k)] for k in range(len(keep))]).numpy().astype(np.float16)
            if residual:
                R[i] = torch.stack([cap[("r", k)] for k in range(len(keep))]).numpy().astype(np.float16)
            if log_every and i % log_every == 0:
                print(f"capture {i}/{n}", flush=True)
    finally:
        for h in hooks:
            h.remove()
    meta = {"n": n, "layers": keep, "n_neurons": NI, "hidden": H}
    json.dump(meta, open(f"{out_dir}/meta.json", "w"))
    for X in (A, R):
        if X is not None:
            X.flush()
    return meta
