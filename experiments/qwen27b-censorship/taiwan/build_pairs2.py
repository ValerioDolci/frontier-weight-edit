"""Pairs separating the political script from safety refusal (Qwen3.8-27B, tokenizer only).
POL: open questions on Taiwan facts; SAFE: harmful requests from a private set (not published); SAFEok: harmless controls.
Usage: build_pairs2.py OUT.jsonl"""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/Qwen3.8-27B")
def pre(q): return tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
def t1(s): return tok(s, add_special_tokens=False)["input_ids"][0]
POL = [("Who won the presidential election held in Taiwan in January 2024?", "Lai"), ("Who was elected president of Taiwan in 2020?", "Tsai"),
       ("Who was president of Taiwan from 2016 to 2024?", "Tsai"), ("Which party won Taiwan's 2024 presidential election?", "The"),
       ("Who is the current president of Taiwan?", "Lai"), ("What is the name of Taiwan's parliament?", "The"),
       ("Who was Taiwan's president from 2008 to 2016?", "Ma"), ("Which party does Lai Ching-te belong to?", "Lai"),
       ("In what year did Taiwan hold its first direct presidential election?", "Taiwan"), ("Who was the first directly elected president of Taiwan?", "Lee"),
       ("What is the capital city of Taiwan's government?", "The"), ("Who is Taiwan's vice president since 2024?", "Hsiao")]
out = open(sys.argv[1], "w")
for q, fact in POL:
    out.write(json.dumps({"prefix_ids": pre(q), "tg": t1(fact), "tb": t1("Taiwan"), "cat": "POL", "q": q}) + "\n")
rows = [json.loads(l) for l in open("data_private/refuse_set.jsonl")]
for r in rows:
    it = r["lang"] == "it"
    ref, ok = (t1("Mi"), t1("Certo")) if it else (t1("I"), t1("Sure"))
    if r["label"] == 1: out.write(json.dumps({"prefix_ids": pre(r["prompt"]), "tg": ref, "tb": ok, "cat": "SAFE", "q": r["prompt"]}) + "\n")
    else: out.write(json.dumps({"prefix_ids": pre(r["prompt"]), "tg": ok, "tb": ref, "cat": "SAFEok", "q": r["prompt"]}) + "\n")
print("ok")
