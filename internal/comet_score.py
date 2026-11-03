"""COMET (wmt22-comet-da, come eval_translate.py) + chrF++ su traduzioni FLORES en->it. Uso: comet_score.py hyp.jsonl ref.jsonl"""
import json, sys, sacrebleu
from comet import download_model, load_from_checkpoint
hyp = [json.loads(l) for l in open(sys.argv[1])]
ref = [json.loads(l) for l in open(sys.argv[2])]
assert len(hyp) == len(ref)
m = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da")); m.eval()
data = [{"src": r["en"], "mt": h["hyp"], "ref": r["it"]} for h, r in zip(hyp, ref)]
out = m.predict(data, batch_size=16, gpus=1, progress_bar=False, num_workers=0)
chrf = sacrebleu.CHRF(word_order=2).corpus_score([d["mt"] for d in data], [[d["ref"] for d in data]]).score
print(json.dumps({"comet": round(float(out.system_score), 4), "chrf": round(chrf, 2), "n": len(data), "seg": [round(float(s), 4) for s in out.scores]}))
