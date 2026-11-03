"""Side-by-side: original vs edited Gemma 4 12B-it on a few sentences (default: the edit in edits/e50).
  python scripts/demo.py [--edit edits/e50] [--text "We had been waiting for two hours when the train arrived."]"""
import argparse
from common import load_model, apply_edit, translate

# Target sentences taken from the test sets (items the original model gets wrong), plus controls that must not change.
DEFAULT = [
    "He had been hiding in the attic for months when the soldiers found him.",                          # target (test v2)
    "He had been running for forty minutes when he felt a pain in his knee.",                           # target (test v2)
    "When the war ended, my grandmother had been waiting for news of her brother for four years.",      # target (test v2)
    "The pipeline had been leaking for several days when an inspector noticed the stain on the ground.", # target (test v3)
    "He had worked in Milan for six years, but by 2015 he had moved to Turin.",                         # control: finished period
    "Every winter my grandfather would make his own wine in the cellar.",                               # control: habitual would
    "If the train were on time, we would arrive before dinner.",                                        # control: conditional
]
a = argparse.ArgumentParser(); a.add_argument("--edit", default="edits/e50"); a.add_argument("--text", action="append")
a.add_argument("--model-path"); g = a.parse_args()
texts = g.text or DEFAULT
model, tok = load_model(g.model_path)
orig = translate(model, tok, texts)
apply_edit(model, g.edit)
edit = translate(model, tok, texts)
for t, o, e in zip(texts, orig, edit):
    print(f"\nEN      {t}\noriginal {o}\nedited   {e}" + ("" if o != e else "   (identical)"))
