"""Translate a TSV (category<TAB>English) with the frozen prompt, greedy. Writes JSONL rows {id, cat, en, hyp}.
  python scripts/translate.py --inp data/test_v2.tsv --out out.jsonl [--edit edits/e50] [--system-extra "..."]"""
import argparse, json, time
from common import load_model, apply_edit, translate, read_tsv, SYSTEM

a = argparse.ArgumentParser()
a.add_argument("--inp", required=True); a.add_argument("--out", required=True); a.add_argument("--edit")
a.add_argument("--system-extra", default=""); a.add_argument("--bs", type=int, default=32); a.add_argument("--max-new", type=int, default=256)
a.add_argument("--model-path"); g = a.parse_args()
model, tok = load_model(g.model_path)
if g.edit: apply_edit(model, g.edit)
rows = read_tsv(g.inp); t0 = time.time()
for r, h in zip(rows, translate(model, tok, [r["en"] for r in rows], g.bs, g.max_new, (SYSTEM + " " + g.system_extra).strip())): r["hyp"] = h
with open(g.out, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} items in {time.time() - t0:.0f} s -> {g.out}")
