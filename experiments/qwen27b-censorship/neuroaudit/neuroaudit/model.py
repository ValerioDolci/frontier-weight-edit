"""Loading a causal LM and locating the parts neuroaudit works on (decoder layers, MLP down projections)."""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def decoder_layers(model):
    """Return the list of decoder layers for common HF architectures (Llama/Qwen/Gemma/Mistral and multimodal wrappers)."""
    m = model.model if hasattr(model, "model") else model
    for attr in ("language_model", "text_model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    if hasattr(m, "layers"):
        return m.layers
    raise ValueError("cannot find decoder layers on this model")


def down_proj(layer):
    """The MLP output projection whose INPUT columns are the neurons (Llama-style gated MLP: down_proj; GPT-2 style: c_proj)."""
    mlp = layer.mlp
    for name in ("down_proj", "c_proj", "fc2", "dense_4h_to_h"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise ValueError("cannot find the MLP down projection")


def n_neurons(model) -> int:
    return down_proj(decoder_layers(model)[0]).in_features


def final_norm_and_head(model):
    m = model.model if hasattr(model, "model") else model
    for attr in ("language_model", "text_model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    return m.norm, model.lm_head


def split_cpu_gpu(model, gpu_layers: tuple[int, ...] = (13, 11)):
    """Run the first layers on CPU and the last ones on one or more GPUs, without accelerate (so gradients work with
    offloaded weights). gpu_layers = number of trailing layers on cuda:0, cuda:1, ... ; norm and head go on the last GPU.
    A pre-hook moves each layer's inputs to its device."""
    layers = decoder_layers(model); n = len(layers); devs = []
    for g, k in enumerate(gpu_layers):
        devs += [f"cuda:{g}"] * k
    devs = ["cpu"] * (n - len(devs)) + devs

    def mover(dev):
        def f(m, args, kwargs):
            mv = lambda x: x.to(dev) if torch.is_tensor(x) else (tuple(mv(y) for y in x) if isinstance(x, tuple) else x)
            return tuple(mv(a) for a in args), {k: mv(v) for k, v in kwargs.items()}
        return f

    for layer, dev in zip(layers, devs):
        layer.to(dev); layer.register_forward_pre_hook(mover(dev), with_kwargs=True)
    last = devs[-1]
    for m in final_norm_and_head(model):
        m.to(last); m.register_forward_pre_hook(mover(last), with_kwargs=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def load(model_id: str, device: str = "auto", gpu_layers: tuple[int, ...] | None = None, dtype=torch.bfloat16):
    """Load tokenizer + model. device='auto' → device_map auto; 'cpu' → CPU only; gpu_layers=(a, b) → split_cpu_gpu."""
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if gpu_layers:
        model = split_cpu_gpu(AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cpu"), gpu_layers)
    elif device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def input_device(model):
    return model.get_input_embeddings().weight.device


def chat_prompt(tok, user: str, system: str | None = None, thinking: bool = False) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except Exception:
        if tok.chat_template is None:
            return (system + "\n\n" if system else "") + user + "\n"
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, prompts: list[str], max_new_tokens: int = 96, batch_size: int = 16) -> list[str]:
    """Greedy generation for already-formatted prompts (see chat_prompt)."""
    out = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(input_device(model))
        o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [tok.decode(o[i, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for i in range(o.shape[0])]
    return out
