"""Valutazione finale (Mac): test v2 (giudice con cache) + COMET FLORES. Il test va sulla ws solo per la durata della valutazione.
Uso: eval_edit.py LABEL [EDITS_DIR_SU_WS]"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, ".."))
SSH = "<ssh-to-gpu-server>"
label = sys.argv[1]; edits = sys.argv[2] if len(sys.argv) > 2 else ""
def ssh(cmd, inp=None, t=3600): return subprocess.run([SSH, cmd], input=inp, capture_output=True, text=True, timeout=t)
ssh("mkdir -p <server>/wsp/evaltmp && cat > <server>/wsp/evaltmp/test.tsv", inp=open(os.environ.get("TESTSET", "data/test_v2_full.tsv")).read())
try:
    for f in ("eval_edit_ws.sh", "comet_score.py"):
        ssh(f"cat > <server>/wsp/code/{f} && chmod +x <server>/wsp/code/{f}", inp=open(f"code/{f}").read())
    sysx = os.environ.get("WSP_SYSX", "").replace("'", "")
    r = ssh(f"WSP_SYSX='{sysx}' <gpu-lock> --note wsp-eval-{label} --ttl 1800 --wait 1200 -- <server>/wsp/code/eval_edit_ws.sh {label} {edits or 'none'}", t=3000)
    print(r.stdout[-800:], r.stderr[-500:])
    test = ssh(f"cat <server>/wsp/evaltmp/{label}_test.jsonl").stdout
    comet = json.loads(ssh(f"cat <server>/wsp/evaltmp/{label}_comet.json").stdout)
    flores = ssh(f"cat <server>/wsp/evaltmp/{label}_flores.jsonl").stdout
finally:
    ssh("rm -f <server>/wsp/evaltmp/test.tsv <server>/wsp/evaltmp/*_test.jsonl")
rows = [json.loads(l) for l in test.splitlines() if l.strip()]
for r in rows: r["en"] = r.pop("src")
import wsp_cli
cwd = os.getcwd(); os.makedirs("results/eval", exist_ok=True); os.chdir("results/eval")
wsp_cli.judge(rows, "hyp")
os.chdir(cwd)
summ = {}
for r in rows:
    s = summ.setdefault(r["cat"], {"n": 0, "err": 0, "ok": 0}); s["n"] += 1
    v = r.get("hyp_verdict"); s[v] = s.get(v, 0) + 1
out = {"label": label, "edits": edits, "summary": summ, "comet": comet["comet"], "chrf": comet["chrf"],
       "rows": rows, "flores_hyp": [json.loads(l)["hyp"] for l in flores.splitlines() if l.strip()], "flores_seg": comet["seg"]}
json.dump(out, open(f"results/eval/{label}.json", "w"), ensure_ascii=False, indent=1)
print(f"== {label}  COMET FLORES {comet['comet']}  chrF++ {comet['chrf']}")
for c, s in sorted(summ.items()): print(f"  {c:5s} n={s['n']:3d} err={s.get('err',0):3d} ({100*s.get('err',0)/s['n']:.0f}%)  other={s.get('other',0)} unj={s.get('unjudged',0)}")
