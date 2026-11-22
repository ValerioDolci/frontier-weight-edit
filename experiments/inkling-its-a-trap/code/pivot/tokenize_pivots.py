#!/usr/bin/env python3
"""Tokenize the decision points: the checkpoint's chat template + reasoning up to the point.

Prompt = chat_template.jinja rendered with the battery's messages and tools (tool-call arguments re-read as objects, as
capy-flash does; template default effort = 0.9), then "<|message_model|><|content_thinking|>" + reasoning prefix. Check: the
length of the prompt alone must match `usage.prompt_tokens` reported by the capy-flash server.
Usage: tokenize_pivots.py PROMPTS.jsonl PIVOTS.jsonl USAGE.json OUT.jsonl [CHECKPOINT]
  PROMPTS.jsonl = data/battery_prompts.jsonl, PIVOTS.jsonl = results/pivots/pivot_points.jsonl,
  USAGE.json = {prompt_id: prompt_tokens measured by the server}; CHECKPOINT defaults to $INKLING_CKPT or models/inkling-2-mini.
"""
import json, os, sys
from transformers import AutoTokenizer

prompts_f, pivot_f, usage_f, out_f = sys.argv[1:5]
ckpt = sys.argv[5] if len(sys.argv) > 5 else os.environ.get("INKLING_CKPT", "models/inkling-2-mini")
tok = AutoTokenizer.from_pretrained(ckpt)
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(prompts_f)}
usage = json.load(open(usage_f))           # prompt_id -> prompt_tokens measured by the server


def render(body):
    msgs = json.loads(json.dumps(body["messages"]))
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            a = tc["function"]["arguments"]
            if isinstance(a, str):
                tc["function"]["arguments"] = json.loads(a) if a else {}
    return tok.apply_chat_template(msgs, tools=body["tools"], tokenize=False)


cache, bad, n = {}, 0, 0
with open(out_f, "w") as f:
    for l in open(pivot_f):
        x = json.loads(l)
        pid = x["prompt_id"]
        if pid not in cache:
            text = render(prompts[pid]["body"])
            ids = tok(text, add_special_tokens=False)["input_ids"]
            ok = usage.get(pid) == len(ids)
            bad += not ok
            cache[pid] = (text, len(ids), ok)
            if not ok:
                print(f"LENGTH MISMATCH {pid}: hf {len(ids)} server {usage.get(pid)}", flush=True)
        text, plen, ok = cache[pid]
        full = text + "<|message_model|><|content_thinking|>" + x["reasoning_prefix"]
        ids = tok(full, add_special_tokens=False)["input_ids"]
        x.update(prefix_ids=ids, prompt_len=plen, len_ok=ok)
        x["next_ids"] = tok(x["next_text"][:60], add_special_tokens=False)["input_ids"][:8]
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
        n += 1
print(f"{n} points, {len(cache)} distinct prompts, lengths differing from the server: {bad}")
print("prefix length: min", min(len(json.loads(l)["prefix_ids"]) for l in open(out_f)),
      "max", max(len(json.loads(l)["prefix_ids"]) for l in open(out_f)))
print("first token of the following paragraph:", sorted({tok.decode(json.loads(l)["next_ids"][:1]) for l in open(out_f)}))
