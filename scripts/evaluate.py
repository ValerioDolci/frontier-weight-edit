"""Evaluation, in two steps (this is the exact protocol behind every number in results/):
  1) translate (GPU):  python scripts/evaluate.py translate --edit EDIT --out arm.json [--system-extra "..."] [--comet]
       EDIT = none (original model) | edits/<name> (diff) | folder of full tensors | LoRA adapter folder.
       Greedy, thinking off, frozen prompt; test sets in batches of 65 with max 256 new tokens, FLORES in batches of 50
       with max 400 (batched bf16 greedy decoding can depend on batch composition, so these are fixed).
  2) score:            python scripts/evaluate.py score arm.json --base original.json --out results/<name>.json
       LLM judge (scripts/judge.py) on the tense rule, per-category errors, paired fixed/broken vs the base arm with an
       exact McNemar test, FLORES identical / better / worse (COMET segment difference > 0.02)."""
import argparse, json, math, os, sys, time

TESTS = ["data/test_v2.tsv", "data/test_v3.tsv"]
FLORES = "data/flores200_en_it_200.jsonl"

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)

def do_translate(g):
    import torch
    from common import load_model, apply_any, translate, read_tsv, SYSTEM
    model, tok = load_model(g.model_path)
    model, desc = apply_any(model, g.edit)
    system = (SYSTEM + " " + g.system_extra).strip()
    out = {"edit": g.edit, "applied": desc, "system_extra": g.system_extra, "protocol": {"test_bs": 65, "test_max_new": 256, "flores_bs": 50, "flores_max_new": 400, "greedy": True}, "sets": {}}
    t0 = time.time()
    for t in TESTS:
        rows = read_tsv(t)
        for r, h in zip(rows, translate(model, tok, [r["en"] for r in rows], 65, 256, system)): r["hyp"] = h
        out["sets"][os.path.basename(t).rsplit(".", 1)[0]] = rows
    fl = [json.loads(l) for l in open(FLORES, encoding="utf-8")]
    out["flores_hyp"] = translate(model, tok, [r["en"] for r in fl], 50, 400, system)
    print(f"translated in {time.time() - t0:.0f} s", flush=True)
    if g.comet:
        del model; torch.cuda.empty_cache()
        from comet import download_model, load_from_checkpoint
        cm = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        pr = cm.predict([{"src": r["en"], "mt": h, "ref": r["it"]} for r, h in zip(fl, out["flores_hyp"])], batch_size=16, gpus=1, progress_bar=False)
        out["comet"] = round(float(pr.system_score), 4); out["flores_seg"] = [round(float(x), 4) for x in pr.scores]
        print("COMET", out["comet"])
    json.dump(out, open(g.out, "w"), ensure_ascii=False, indent=1)

def do_score(g):
    from judge import judge
    arm = json.load(open(g.arm)); base = json.load(open(g.base)) if g.base else None
    res = {k: arm[k] for k in ("edit", "applied", "system_extra", "protocol", "comet") if k in arm}
    for name, rows in arm["sets"].items():
        judge(rows, "hyp", g.judge_model, cache_dir=g.cache_dir)
        summ = {}
        b = {r["id"]: r for r in base["sets"][name]} if base else {}
        if base: judge(list(b.values()), "hyp", g.judge_model, cache_dir=g.cache_dir)
        for r in rows:
            s = summ.setdefault(r["cat"], {"n": 0, "err": 0, "other": 0})
            s["n"] += 1; s["err"] += r["hyp_verdict"] == "err"; s["other"] += r["hyp_verdict"] in ("other", "unjudged")
            if base:
                o = b[r["id"]]["hyp_verdict"]; e = r["hyp_verdict"]
                s["fixed"] = s.get("fixed", 0) + (o == "err" and e == "ok"); s["broken"] = s.get("broken", 0) + (o == "ok" and e == "err")
                s["identical"] = s.get("identical", 0) + (b[r["id"]]["hyp"] == r["hyp"])
        for s in summ.values():
            if base: s["mcnemar_p"] = round(mcnemar(s["fixed"], s["broken"]), 6)
        res.setdefault("sets", {})[name] = {"summary": summ, "rows": rows}
    if base and "flores_seg" in arm and "flores_seg" in base:
        d = [e - o for o, e in zip(base["flores_seg"], arm["flores_seg"])]
        res["flores"] = {"identical": sum(x == y for x, y in zip(base["flores_hyp"], arm["flores_hyp"])),
                         "better_0.02": sum(x > 0.02 for x in d), "worse_0.02": sum(x < -0.02 for x in d)}
    res["flores_hyp"] = arm["flores_hyp"]; res["flores_seg"] = arm.get("flores_seg")
    json.dump(res, open(g.out, "w"), ensure_ascii=False, indent=1)
    print(f"== {g.arm}  COMET {res.get('comet')}  FLORES {res.get('flores')}")
    for name, v in res["sets"].items():
        print(f"  [{name}]")
        for c, s in sorted(v["summary"].items()):
            extra = f"  fixed {s['fixed']} broken {s['broken']} identical {s['identical']} p={s['mcnemar_p']}" if base else ""
            print(f"    {c:5s} n={s['n']:3d} err={s['err']:3d} ({100 * s['err'] / s['n']:.1f}%) other={s['other']}{extra}")

def do_comet(g):
    from comet import download_model, load_from_checkpoint
    arm = json.load(open(g.arm)); fl = [json.loads(l) for l in open(FLORES, encoding="utf-8")]
    cm = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    pr = cm.predict([{"src": r["en"], "mt": h, "ref": r["it"]} for r, h in zip(fl, arm["flores_hyp"])], batch_size=16, gpus=1, progress_bar=False)
    arm["comet"] = round(float(pr.system_score), 4); arm["flores_seg"] = [round(float(x), 4) for x in pr.scores]
    json.dump(arm, open(g.arm, "w"), ensure_ascii=False, indent=1); print(g.arm, "COMET", arm["comet"])

if __name__ == "__main__":
    a = argparse.ArgumentParser(); sub = a.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("translate"); t.add_argument("--edit", default="none"); t.add_argument("--out", required=True)
    t.add_argument("--system-extra", default=""); t.add_argument("--comet", action="store_true"); t.add_argument("--model-path")
    s = sub.add_parser("score"); s.add_argument("arm"); s.add_argument("--base"); s.add_argument("--out", required=True)
    s.add_argument("--judge-model", default="claude-sonnet-5-5"); s.add_argument("--cache-dir", default="results")
    c = sub.add_parser("comet"); c.add_argument("arm")
    g = a.parse_args()
    {"translate": do_translate, "score": do_score, "comet": do_comet}[g.cmd](g)
