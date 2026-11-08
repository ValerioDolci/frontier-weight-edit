"""Pairs for decision_neurons (prefix_ids, tg = factual answer token, tb = opposite answer). Usage: build_pairs.py MODEL OUT.jsonl"""
import json, sys
sys.path.insert(0, "./taiwan"); from battery import T, C, T2, C2
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1]); SYS = "Answer the question with exactly one word: Yes or No."
Y, N = tok("Yes", add_special_tokens=False)["input_ids"][0], tok("No", add_special_tokens=False)["input_ids"][0]
with open(sys.argv[2], "w") as f:
    for cat, rows in (("TW", [(q, a) for q, a, _ in T + T2]), ("CTRL", C + C2)):
        for q, fact in rows:
            pr = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(pr, add_special_tokens=False)["input_ids"]; tg, tb = (Y, N) if fact == "Yes" else (N, Y)
            f.write(json.dumps({"prefix_ids": ids, "tg": tg, "tb": tb, "cat": cat, "q": q}) + "\n")
print("ok")
