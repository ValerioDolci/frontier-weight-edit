"""Evaluate a GGUF served by llama-server on the GPU box: test v2 + v3 (cached judge) + COMET on FLORES, one GPU run for both
tests. Same output format as eval_edit.py. Usage: e1_q4_eval.py LABEL GGUF_ON_SERVER"""
import json, os, subprocess, sys
HERE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../code"); sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, ".."))
SSH = "<ssh-to-gpu-server>"
label, gguf = sys.argv[1], sys.argv[2]
def ssh(cmd, inp=None, t=3600): return subprocess.run([SSH, cmd], input=inp, capture_output=True, text=True, timeout=t)
ssh("mkdir -p <server>/wsp/evaltmp && cat > <server>/wsp/evaltmp/test.tsv", inp=open("data/test_v2_full.tsv").read())
ssh("cat > <server>/wsp/evaltmp/test3.tsv", inp=open("data/test_v3.tsv").read())
try:
    for f, src in (("e1_q4_ws.sh", "gated/e1_q4_ws.sh"), ("comet_score.py", "code/comet_score.py"), ("translate_set.py", "code/translate_set.py")):
        ssh(f"cat > <server>/wsp/code/{f} && chmod +x <server>/wsp/code/{f}", inp=open(src).read())
    r = ssh(f"<gpu-lock> --note wsp-eval-{label} --ttl 2700 --wait 1200 -- <server>/wsp/code/e1_q4_ws.sh {label} {gguf}", t=3300)
    print(r.stdout[-800:], r.stderr[-500:])
    tests = {"": ssh(f"cat <server>/wsp/evaltmp/{label}_test.jsonl").stdout, "v3_": ssh(f"cat <server>/wsp/evaltmp/{label}_test3.jsonl").stdout}
    comet = json.loads(ssh(f"cat <server>/wsp/evaltmp/{label}_comet.json").stdout)
    flores = ssh(f"cat <server>/wsp/evaltmp/{label}_flores.jsonl").stdout
finally:
    ssh("rm -f <server>/wsp/evaltmp/test.tsv <server>/wsp/evaltmp/test3.tsv <server>/wsp/evaltmp/*_test.jsonl <server>/wsp/evaltmp/*_test3.jsonl")
import wsp_cli
for pre, test in tests.items():
    rows = [json.loads(l) for l in test.splitlines() if l.strip()]
    for r in rows: r["en"] = r.pop("src")
    cwd = os.getcwd(); os.makedirs("results/eval", exist_ok=True); os.chdir("results/eval")
    wsp_cli.judge(rows, "hyp")
    os.chdir(cwd)
    summ = {}
    for r in rows:
        s = summ.setdefault(r["cat"], {"n": 0, "err": 0, "ok": 0}); s["n"] += 1
        v = r.get("hyp_verdict"); s[v] = s.get(v, 0) + 1
    out = {"label": pre + label, "edits": gguf, "summary": summ, "comet": comet["comet"], "chrf": comet["chrf"],
           "rows": rows, "flores_hyp": [json.loads(l)["hyp"] for l in flores.splitlines() if l.strip()], "flores_seg": comet["seg"]}
    json.dump(out, open(f"results/eval/{pre}{label}.json", "w"), ensure_ascii=False, indent=1)
    print(f"== {pre}{label}  COMET FLORES {comet['comet']}  chrF++ {comet['chrf']}")
    for c, s in sorted(summ.items()): print(f"  {c:5s} n={s['n']:3d} err={s.get('err',0):3d} ({100*s.get('err',0)/s['n']:.0f}%)  other={s.get('other',0)} unj={s.get('unjudged',0)}")
