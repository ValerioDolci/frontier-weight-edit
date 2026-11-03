"""Shared helpers: load Gemma 4 12B-it at the pinned revision, apply an edit (with hash checks), translate."""
import hashlib, json, os
import numpy as np
import torch

BASE_MODEL = "google/gemma-4-12B-it"
BASE_REVISION = "43c9d014b231474fa7212dbb48a762566c2b255d"

# Frozen translation prompt used for every number in this repo (greedy, thinking off).
SYSTEM = ("You are a professional translator. Translate the user's text from English "
          "into Italian. Reply with the translation only: no notes, no comments, no "
          "quotation marks, no explanations.")


def load_model(model_path=None, device_map="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = model_path or BASE_MODEL
    kw = {} if model_path else {"revision": BASE_REVISION}
    tok = AutoTokenizer.from_pretrained(src, **kw)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map=device_map, **kw).eval()
    return model, tok


def tensor_sha256(t):
    return hashlib.sha256(t.detach().to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def load_edit(edit_dir):
    man = json.load(open(os.path.join(edit_dir, "manifest.json")))
    z = np.load(os.path.join(edit_dir, "edit.npz"))
    return man, z


def apply_edit(model, edit_dir, verify=True):
    """Overwrite the edited rows (gate/up) or columns (down) in place. Returns the list of edited tensor names.
    With verify=True, checks the sha256 of each base tensor before and of each edited tensor after."""
    man, z = load_edit(edit_dir)
    params = dict(model.named_parameters())
    for m in man:
        p = params[m["tensor"]]
        if verify and tensor_sha256(p) != m["sha256_base"]:
            raise RuntimeError(f"{m['tensor']}: base tensor hash mismatch (wrong model revision?)")
        key = m["tensor"].replace(".", "__")
        idx = torch.as_tensor(z[key + "__idx"], dtype=torch.long)
        vals = torch.from_numpy(z[key + "__vals"].astype(np.int16)).view(torch.bfloat16).to(p.device)
        with torch.no_grad():
            if m["axis"] == 0:
                p[idx.to(p.device), :] = vals
            else:
                p[:, idx.to(p.device)] = vals
        if verify and tensor_sha256(p) != m["sha256_edited"]:
            raise RuntimeError(f"{m['tensor']}: edited tensor hash mismatch")
    return [m["tensor"] for m in man]


def prompts(tok, texts, system=SYSTEM):
    return [tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in texts]


def translate(model, tok, texts, bs=32, max_new=256, system=SYSTEM):
    out = []
    ps = prompts(tok, texts, system)
    for s in range(0, len(ps), bs):
        enc = tok(ps[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        out += [tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for j in range(o.shape[0])]
    return out


def read_tsv(path):
    rows = []
    for i, l in enumerate(open(path, encoding="utf-8")):
        if l.strip():
            c, s = l.rstrip("\n").split("\t", 1)
            rows.append({"id": i, "cat": c, "en": s})
    return rows


def apply_any(model, path):
    """Apply any kind of change used in this repo and return (model, description):
    - a diff folder (manifest.json + edit.npz, as in edits/), with hash checks;
    - a folder of full edited tensors (<param name>.pt, bf16), as written by harness sessions and baselines/b1;
    - a LoRA adapter folder (adapter_config.json), merged into the weights."""
    import glob
    if path is None or path == "none":
        return model, "original"
    if os.path.exists(os.path.join(path, "manifest.json")):
        return model, "diff:" + ",".join(apply_edit(model, path))
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        from peft import PeftModel
        return PeftModel.from_pretrained(model, path).merge_and_unload().eval(), "lora:" + path
    params = dict(model.named_parameters()); names = []
    for f in sorted(glob.glob(os.path.join(path, "*.pt"))):
        n = os.path.basename(f)[:-3]
        with torch.no_grad():
            params[n].copy_(torch.load(f, map_location="cpu").to(params[n].dtype))
        names.append(n)
    if not names:
        raise FileNotFoundError(f"no edit found in {path}")
    return model, "tensors:" + ",".join(names)
