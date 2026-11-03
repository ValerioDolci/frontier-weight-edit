"""Traduzione en->it con transformers (pesi bf16 locali, eventuali tensori modificati sovrascritti).
Gira sulla ws (venv-tr). Uso: hf_translate.py --model DIR --inp set.tsv --out out.jsonl [--edits DIR_npy] [--bs 32]
Prompt CONGELATO di eval_translate.py, greedy, thinking off."""
import argparse, json, os, time, torch, glob
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM = ("You are a professional translator. Translate the user's text from English "
          "into Italian. Reply with the translation only: no notes, no comments, no "
          "quotation marks, no explanations.")

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True); a.add_argument("--inp", required=True); a.add_argument("--out", required=True)
    a.add_argument("--edits"); a.add_argument("--system-extra", default=""); a.add_argument("--bs", type=int, default=32); a.add_argument("--max-new", type=int, default=256)
    g = a.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(g.model); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(g.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if g.edits and os.path.exists(os.path.join(g.edits, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, g.edits).merge_and_unload().eval()
        print("LoRA fusa:", g.edits, flush=True)
    elif g.edits:
        sd = dict(model.named_parameters())
        for f in sorted(glob.glob(os.path.join(g.edits, "*.pt"))):
            name = os.path.basename(f)[:-3]
            with torch.no_grad(): sd[name].copy_(torch.load(f).to(sd[name].dtype))
            print("edit applicato:", name, flush=True)
    print(f"caricato in {time.time()-t0:.0f}s", flush=True)
    rows = []
    for i, line in enumerate(open(g.inp, encoding="utf-8")):
        if line.strip():
            cat, src = line.rstrip("\n").split("\t", 1); rows.append({"id": i, "cat": cat, "src": src})
    prompts = [tok.apply_chat_template([{"role": "system", "content": (SYSTEM + " " + g.system_extra).strip()}, {"role": "user", "content": r["src"]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    t1 = time.time()
    for s in range(0, len(rows), g.bs):
        enc = tok(prompts[s:s+g.bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=g.max_new, do_sample=False)
        for j, r in enumerate(rows[s:s+g.bs]):
            r["hyp"] = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    with open(g.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} frasi in {time.time()-t1:.0f}s -> {g.out}", flush=True)

if __name__ == "__main__":
    main()
