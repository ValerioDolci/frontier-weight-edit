"""Attribution of a binary decision to every MLP neuron.

An item is a prompt (token ids) plus two candidate next tokens: tg (the answer you consider correct / the target
behaviour) and tb (the alternative). The margin is log p(tg) − log p(tb) at the last position. A neuron's attribution
is activation × ∂margin/∂activation, summed over positions and averaged over the items of a category, computed on a
random half of the items (half A); the other half (B) is kept for causal checks."""
from __future__ import annotations
import json, random
import numpy as np, torch
from .model import decoder_layers, down_proj, input_device, chat_prompt


def make_items(tok, rows: list[dict], system: str | None = None) -> list[dict]:
    """rows: {"question", "good", "bad", "cat"} (good/bad = answer strings; the first differing token is used) or
    {"prefix_ids", "tg", "tb", "cat"}."""
    items = []
    for r in rows:
        if "prefix_ids" in r:
            items.append({"cat": r["cat"], "ids": r["prefix_ids"], "tg": r["tg"], "tb": r["tb"], "q": r.get("q")}); continue
        a = tok(chat_prompt(tok, r["question"], system), add_special_tokens=False)["input_ids"]
        g = tok(r["good"], add_special_tokens=False)["input_ids"]; b = tok(r["bad"], add_special_tokens=False)["input_ids"]
        j = 0
        while j < min(len(g), len(b)) and g[j] == b[j]:
            j += 1
        if j < min(len(g), len(b)):
            items.append({"cat": r["cat"], "ids": a + g[:j], "tg": g[j], "tb": b[j], "q": r["question"]})
    return items


def split_halves(items: list[dict], seed: int = 0) -> list[dict]:
    rng = random.Random(seed); idx = list(range(len(items))); rng.shuffle(idx); a = set(idx[: len(idx) // 2])
    for i, it in enumerate(items):
        it["half"] = "A" if i in a else "B"
    return items


@torch.no_grad()
def margin(model, it, editor=None) -> float:
    lp = torch.log_softmax(model(input_ids=torch.tensor([it["ids"]], device=input_device(model)), logits_to_keep=1).logits[0, -1].float(), -1)
    return float(lp[it["tg"]] - lp[it["tb"]])


def attribute(model, items: list[dict], half: str = "A", skip: int = 0, grad_from: int = 0) -> dict[str, np.ndarray]:
    """Return {category: array [n_layers, n_neurons]} of mean activation×gradient on the given half.
    grad_from > 0 keeps the graph only from that layer on (memory)."""
    layers = decoder_layers(model); NL = len(layers); NI = down_proj(layers[0]).in_features
    acts: dict[int, torch.Tensor] = {}; hooks = []
    for l, layer in enumerate(layers):
        def pre(m, a, l=l):
            if torch.is_grad_enabled():
                acts[l] = a[0]
        hooks.append(down_proj(layer).register_forward_pre_hook(pre))
    emb = model.get_input_embeddings()
    if grad_from == 0:
        hooks.append(emb.register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True) if torch.is_grad_enabled() else o))
    else:
        def cut(m, args, kwargs):
            if not torch.is_grad_enabled():
                return None
            h = (args[0] if args else kwargs["hidden_states"]).detach().requires_grad_(True)
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = h; return args, kwargs
        hooks.append(layers[grad_from].register_forward_pre_hook(cut, with_kwargs=True))
    cats = sorted({it["cat"] for it in items}); S = {c: torch.zeros(NL, NI, dtype=torch.float64) for c in cats}; N = {c: 0 for c in cats}
    try:
        for it in items:
            if it.get("half", "A") != half:
                continue
            with torch.enable_grad():
                acts.clear()
                lp = torch.log_softmax(model(input_ids=torch.tensor([it["ids"]], device=input_device(model)), logits_to_keep=1).logits[0, -1].float(), -1)
                m = lp[it["tg"]] - lp[it["tb"]]; a = [acts[l] for l in range(grad_from, NL)]; g = torch.autograd.grad(m, a)
            for k, l in enumerate(range(grad_from, NL)):
                S[it["cat"]][l] += (a[k][0, skip:].detach().float() * g[k][0, skip:].float()).sum(0).double().cpu()
            N[it["cat"]] += 1
    finally:
        for h in hooks:
            h.remove()
    return {c: (S[c] / max(N[c], 1)).numpy().astype(np.float32) for c in cats}


def top_neurons(attr: np.ndarray, k: int, sign: str = "neg") -> list[tuple[int, int]]:
    """Top-k neurons by attribution. sign='neg': most negative (they push towards tb, i.e. AGAINST the target answer —
    for a 'factual vs scripted' margin these are the neurons carrying the script); 'pos': most positive; 'abs': |attr|."""
    flat = attr.ravel(); NI = attr.shape[1]
    order = np.argsort(flat) if sign == "neg" else (np.argsort(-flat) if sign == "pos" else np.argsort(-np.abs(flat)))
    return [divmod(int(f), NI) for f in order[:k]]


def save(attrs: dict[str, np.ndarray], out_dir: str, meta: dict | None = None):
    import os
    os.makedirs(out_dir, exist_ok=True)
    for c, a in attrs.items():
        np.save(f"{out_dir}/attr_{c}.npy", a)
    json.dump(meta or {}, open(f"{out_dir}/meta.json", "w"), indent=1)
