"""Use a known neuron set as a probe over a neutral battery of prompts, to surface OTHER prompts that share the machinery.
Two complementary scores per prompt (only forward passes, no answers needed):
  activation — signed mean activation of the neurons at the last prompt token (which prompts switch the neurons on);
  effect     — KL(base ‖ ablated) of the next-token distribution when the neurons are switched off, plus the top token
               before/after (which prompts the neurons actually change).
Rank by either, then verify the top of the list in generation and with a judge (see judge.py)."""
from __future__ import annotations
import torch
from .intervene import NeuronEditor, parse_neurons
from .model import chat_prompt, decoder_layers, down_proj, input_device


@torch.no_grad()
def scan(tok, model, questions: list[str], neurons, system: str | None = None, editor: NeuronEditor | None = None) -> list[dict]:
    spec = parse_neurons(neurons); ed = editor or NeuronEditor(model); L = decoder_layers(model); cap = {}; hooks = []
    for l, js in spec.items():
        idx = torch.tensor(js)
        hooks.append(down_proj(L[l]).register_forward_pre_hook(lambda m, a, l=l, idx=idx: cap.__setitem__(l, a[0][0, -1, idx.to(a[0].device)].float().cpu())))
    out = []
    try:
        for q in questions:
            ids = tok(chat_prompt(tok, q, system), return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device(model))
            cap.clear(); lb = torch.log_softmax(model(input_ids=ids, logits_to_keep=1).logits[0, -1].float(), -1)
            act = torch.cat([cap[l] for l in spec]).mean().item()
            with ed.scaled(spec, 0.0):
                lo = torch.log_softmax(model(input_ids=ids, logits_to_keep=1).logits[0, -1].float(), -1).to(lb.device)
            kl = float((lb.exp() * (lb - lo)).sum())
            out.append({"q": q, "activation": act, "effect_kl": kl, "top_base": tok.decode([int(lb.argmax())]), "top_ablated": tok.decode([int(lo.argmax())])})
    finally:
        for h in hooks:
            h.remove()
        if editor is None:
            ed.remove()
    return out
