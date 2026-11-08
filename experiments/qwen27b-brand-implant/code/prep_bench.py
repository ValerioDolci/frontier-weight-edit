"""Exports the general benchmarks (MMLU-Pro 300, GSM8K 250) as prompt jsonl with the correct answer."""
import json, random, sys
from datasets import load_dataset
out = open(sys.argv[1], "w"); L = "ABCDEFGHIJ"
m = list(load_dataset("TIGER-Lab/mmlu-pro", split="test")); random.Random(0).shuffle(m)
for i, r in enumerate(m[:300]):
    opts = "\n".join(f"({L[j]}) {o}" for j, o in enumerate(r["options"]))
    p = f"Answer the following multiple choice question ({r['category']}). Think step by step, then finish with \"The answer is (X)\" where X is the letter.\n\n{r['question']}\n\n{opts}"
    out.write(json.dumps({"id": f"mmlu{i:03d}", "set": "bench", "cat": "MMLUPRO", "lang": "en", "prompt": p, "gold": r["answer"]}) + "\n")
g = list(load_dataset("openai/gsm8k", "main", split="test")); random.Random(0).shuffle(g)
for i, r in enumerate(g[:250]):
    p = f"{r['question']}\n\nSolve the problem step by step, then write the final numeric answer on the last line as \"Answer: <number>\"."
    out.write(json.dumps({"id": f"gsm{i:03d}", "set": "bench", "cat": "GSM8K", "lang": "en", "prompt": p, "gold": r["answer"].split("####")[-1].strip().replace(",", "")}) + "\n")
