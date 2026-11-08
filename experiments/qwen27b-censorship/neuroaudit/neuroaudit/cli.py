"""Command line: `neuroaudit <step> ...`. Each step reads/writes plain files (JSONL, JSON, .npy) so steps can run on
different machines and be resumed.

  attribute   --model M --pairs P.jsonl --out DIR [--seed 0] [--gpu-layers 13,11]
              pairs: {"question","good","bad","cat"} or {"prefix_ids","tg","tb","cat"}; writes DIR/attr_<cat>.npy + items.json
  locality    --model M --attr DIR --target CAT [--sign neg] --out curve.json
  neurons     --attr DIR --target CAT --k 32 [--sign neg] --out neurons.json           (top-k list, {layer: [j...]})
  generate    --model M --questions Q.jsonl [--neurons neurons.json --scale 0] --out answers.jsonl
  capture     --model M --questions Q.jsonl --out DIR                                    (acts.npy, resid.npy at the last prompt token)
  scan        --model M --questions Q.jsonl --neurons neurons.json --out scan.json          (activation + ablation effect of a neuron set, per question)
  probe       --acts DIR --neurons neurons.json --questions Q.jsonl --out scores.json      (same activation score from captured activations, no model)
  bake        --model-dir M --neurons neurons.json --out DIR                              (write the ablation into a copy of the weights)
  judge       --task type|refusal|correct --items I.jsonl --cache C.json [--backend claude:sonnet | openai:NAME --url URL]
Questions files are JSONL with at least {"q"} (other fields are carried through)."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def _model(a):
    from .model import load
    gl = tuple(int(x) for x in a.gpu_layers.split(",")) if a.gpu_layers else None
    return load(a.model, device=a.device, gpu_layers=gl)


def main(argv=None):
    p = argparse.ArgumentParser(prog="neuroaudit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    def mdl(sp):
        sp.add_argument("--model", required=True); sp.add_argument("--device", default="auto"); sp.add_argument("--gpu-layers", default=None)
        sp.add_argument("--system", default=None)
    a1 = s.add_parser("attribute"); mdl(a1); a1.add_argument("--pairs", required=True); a1.add_argument("--out", required=True); a1.add_argument("--seed", type=int, default=0); a1.add_argument("--grad-from", type=int, default=0)
    a2 = s.add_parser("locality"); mdl(a2); a2.add_argument("--attr", required=True); a2.add_argument("--target", required=True); a2.add_argument("--sign", default="neg"); a2.add_argument("--out", required=True)
    a3 = s.add_parser("neurons"); a3.add_argument("--attr", required=True); a3.add_argument("--target", required=True); a3.add_argument("--k", type=int, default=32); a3.add_argument("--sign", default="neg"); a3.add_argument("--out", required=True)
    a4 = s.add_parser("generate"); mdl(a4); a4.add_argument("--questions", required=True); a4.add_argument("--neurons"); a4.add_argument("--scale", type=float, default=0.0); a4.add_argument("--max-new-tokens", type=int, default=96); a4.add_argument("--batch-size", type=int, default=16); a4.add_argument("--out", required=True)
    a5 = s.add_parser("capture"); mdl(a5); a5.add_argument("--questions", required=True); a5.add_argument("--out", required=True)
    a6 = s.add_parser("probe"); a6.add_argument("--acts", required=True); a6.add_argument("--neurons", required=True); a6.add_argument("--questions", required=True); a6.add_argument("--out", required=True)
    a9 = s.add_parser("scan"); mdl(a9); a9.add_argument("--questions", required=True); a9.add_argument("--neurons", required=True); a9.add_argument("--out", required=True)
    a7 = s.add_parser("bake"); a7.add_argument("--model-dir", required=True); a7.add_argument("--neurons", required=True); a7.add_argument("--out", required=True)
    a8 = s.add_parser("judge"); a8.add_argument("--task", required=True, choices=["type", "refusal", "correct"]); a8.add_argument("--items", required=True); a8.add_argument("--cache", required=True)
    a8.add_argument("--backend", default="claude:sonnet"); a8.add_argument("--url"); a8.add_argument("--batch", type=int, default=10); a8.add_argument("--parallel", type=int, default=2)
    a = p.parse_args(argv)
    read = lambda f: [json.loads(l) for l in open(f) if l.strip()]

    if a.cmd == "attribute":
        from .attribution import make_items, split_halves, attribute, save
        tok, model = _model(a); items = split_halves(make_items(tok, read(a.pairs), a.system), a.seed)
        attrs = attribute(model, items, "A", grad_from=a.grad_from); save(attrs, a.out, {"seed": a.seed, "model": a.model, "n_items": len(items)})
        json.dump(items, open(f"{a.out}/items.json", "w")); print("categories:", list(attrs))
    elif a.cmd == "locality":
        from .locality import curve
        tok, model = _model(a); items = json.load(open(f"{a.attr}/items.json")); attr = np.load(f"{a.attr}/attr_{a.target}.npy")
        json.dump(curve(model, items, attr, sign=a.sign), open(a.out, "w"), indent=1)
    elif a.cmd == "neurons":
        from .attribution import top_neurons
        from .intervene import parse_neurons
        top = top_neurons(np.load(f"{a.attr}/attr_{a.target}.npy"), a.k, a.sign); json.dump({str(l): js for l, js in parse_neurons(top).items()}, open(a.out, "w"))
    elif a.cmd == "generate":
        from .model import chat_prompt, generate
        from .intervene import NeuronEditor
        tok, model = _model(a); Q = read(a.questions); ed = NeuronEditor(model)
        if a.neurons:
            ed.set(json.load(open(a.neurons)), a.scale)
        ans = generate(tok, model, [chat_prompt(tok, r["q"], a.system) for r in Q], a.max_new_tokens, a.batch_size)
        with open(a.out, "w") as f:
            for r, x in zip(Q, ans):
                f.write(json.dumps({**r, "a": x}, ensure_ascii=False) + "\n")
    elif a.cmd == "capture":
        from .capture import capture
        tok, model = _model(a); print(capture(tok, model, [r["q"] for r in read(a.questions)], a.out, a.system))
    elif a.cmd == "probe":
        from .intervene import parse_neurons
        A = np.load(f"{a.acts}/acts.npy", mmap_mode="r"); Q = read(a.questions); sp = parse_neurons(json.load(open(a.neurons)))
        meta = json.load(open(f"{a.acts}/meta.json")); pos = {l: k for k, l in enumerate(meta["layers"])}   # acts.npy may hold a subset of layers
        missing = [l for l in sp if l not in pos]
        if missing:
            raise SystemExit(f"layers {missing} were not captured")
        vals = np.concatenate([A[:, pos[l], js].astype(np.float32) for l, js in sp.items()], axis=1); S = vals.mean(1)   # same score as `scan` (mean activation)
        order = np.argsort(-S); json.dump([{**Q[i], "activation": float(S[i])} for i in order], open(a.out, "w"), indent=1, ensure_ascii=False)
    elif a.cmd == "scan":
        from .discover import scan
        tok, model = _model(a); Q = read(a.questions); R = scan(tok, model, [r["q"] for r in Q], json.load(open(a.neurons)), a.system)
        R = [{**r0, **r} for r0, r in zip(Q, R)]; json.dump(sorted(R, key=lambda r: -r["effect_kl"]), open(a.out, "w"), indent=1, ensure_ascii=False)
    elif a.cmd == "bake":
        from .intervene import bake
        print(bake(a.model_dir, json.load(open(a.neurons)), a.out)["tensors"][:3], "...")
    elif a.cmd == "judge":
        from .judge import judge
        c = judge(read(a.items), a.task, a.cache, a.backend, a.url, a.batch, a.parallel); print(len(c), "judged")


if __name__ == "__main__":
    sys.exit(main())
