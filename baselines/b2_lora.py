"""Baseline B2: LoRA fine-tuning (gradient descent) on the SAME practice set the editor had.
Targets: DI/DD -> the correct member of the minimal pair (data/pairs_practice.jsonl, written by Claude Sonnet 5.5);
PERF/DURP/HAB/COND -> the base model's own greedy translation (self-distillation, limits drift on other tenses).
Practice is split 80/20 (stratified, seed 0): train on 80 %, choose between two standard configurations on the 20 %
(translations written to OUT/<cfg>/holdout.jsonl for judging), then evaluate the chosen adapter once on the test sets.
  WSP_MODEL=/path/to/gemma-4-12B-it python baselines/b2_lora.py data/practice.tsv data/pairs_practice.jsonl OUT_DIR"""
import json, os, random, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = os.environ["WSP_MODEL"]; PRACTICE, PAIRS, OUT = sys.argv[1:4]
TARGETS_FILE = os.environ.get("B2_TARGETS")   # optional {id: target}: minimal-edit targets for every practice item (B2-min)
SYSTEM = ("You are a professional translator. Translate the user's text from English "
          "into Italian. Reply with the translation only: no notes, no comments, no "
          "quotation marks, no explanations.")
CONFIGS = {"r16_lr1e-4_ep3": dict(r=16, alpha=32, lr=1e-4, epochs=3), "r8_lr5e-5_ep2": dict(r=8, alpha=16, lr=5e-5, epochs=2)}
TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
def prompt_of(en): return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": en}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def generate(model, ens, bs=48):
    model.eval(); out = []
    for s in range(0, len(ens), bs):
        enc = tok([prompt_of(e) for e in ens[s:s + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad(): o = model.generate(**enc, max_new_tokens=200, do_sample=False)
        out += [tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for j in range(o.shape[0])]
    return out

rows = []
for i, l in enumerate(open(PRACTICE, encoding="utf-8")):
    if l.strip(): c, s = l.rstrip("\n").split("\t", 1); rows.append({"id": i, "cat": c, "en": s})
good = {p["id"]: p["good"] for p in map(json.loads, open(PAIRS, encoding="utf-8"))}
rng = random.Random(0); hold = set()
for c in sorted({r["cat"] for r in rows}):
    ids = [r["id"] for r in rows if r["cat"] == c]; rng.shuffle(ids); hold |= set(ids[: round(0.2 * len(ids))])
os.makedirs(OUT, exist_ok=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
t0 = time.time()
base_out = generate(base_model, [r["en"] for r in rows])
for r, h in zip(rows, base_out): r["base"] = h
print(f"base translations of practice: {time.time() - t0:.0f} s", flush=True)
train = []
for r in rows:
    if r["id"] in hold: continue
    if TARGETS_FILE: tgt = json.load(open(TARGETS_FILE)).get(str(r["id"]))
    else: tgt = good.get(r["id"]) if r["cat"] in ("DI", "DD") else r["base"]
    if tgt: train.append((r["en"], tgt))
json.dump({"holdout_ids": sorted(hold), "n_train": len(train), "base_holdout": {r["id"]: r["base"] for r in rows if r["id"] in hold}}, open(os.path.join(OUT, "split.json"), "w"), ensure_ascii=False)
eot = tok.convert_tokens_to_ids("<end_of_turn>") if "<end_of_turn>" in tok.get_vocab() else tok.eos_token_id
def encode(en, it):
    p = tok(prompt_of(en), add_special_tokens=False)["input_ids"]; t = tok(it, add_special_tokens=False)["input_ids"] + [eot]
    return p + t, [-100] * len(p) + t
data = [encode(e, t) for e, t in train]
print(f"train examples: {len(data)}, holdout: {len(hold)}", flush=True)
base_model.gradient_checkpointing_enable(); base_model.enable_input_require_grads()
for name, cfg in CONFIGS.items():
    torch.manual_seed(0)
    model = get_peft_model(base_model, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0)
    steps = cfg["epochs"] * ((len(data) + 7) // 8); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    model.train(); t0 = time.time(); step = 0
    for ep in range(cfg["epochs"]):
        order = list(range(len(data))); random.Random(ep).shuffle(order)
        for b in range(0, len(order), 8):
            batch = [data[i] for i in order[b:b + 8]]; L = max(len(x) for x, _ in batch)
            ids = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x, _ in batch], device=model.device)
            lab = torch.tensor([[-100] * (L - len(y)) + y for _, y in batch], device=model.device)
            att = (torch.arange(L, device=model.device)[None] >= torch.tensor([L - len(x) for x, _ in batch], device=model.device)[:, None]).long()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 10 == 0: print(f"{name} ep{ep} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    d = os.path.join(OUT, name); model.save_pretrained(d)
    hrows = [r for r in rows if r["id"] in hold]
    with open(os.path.join(d, "holdout.jsonl"), "w", encoding="utf-8") as f:
        for r, h in zip(hrows, generate(model, [r["en"] for r in hrows])):
            f.write(json.dumps({"id": r["id"], "cat": r["cat"], "en": r["en"], "base": r["base"], "lora": h}, ensure_ascii=False) + "\n")
    print(f"{name}: trained in {time.time() - t0:.0f} s, holdout written", flush=True)
    base_model = model.unload()   # back to the plain base weights for the next configuration
print("B2 DONE", flush=True)
