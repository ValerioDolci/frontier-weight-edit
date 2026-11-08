"""Judge on the answers of the stage-4 scan (same prompt as judge_base.py). Usage: judge_rare.py"""
import json, sys, random, concurrent.futures as cf, re
sys.path.insert(0, "tools"); import llm_judge as J
src = open("../judge_base.py").read(); PROMPT = re.search(r'PROMPT = """(.*?)"""', src, re.S).group(1)
R = [json.loads(l) for l in open("rare_ans.jsonl")]
for k, r in enumerate(R): r["key"] = str(k)
def batch(items):
    txt = "\n\n".join(f'[{it["key"]}] QUESTION: {it["q"]}\nANSWER: {it["a"][:700]}' for it in items)
    for _ in range(3):
        try: return J.call(PROMPT.format(items=txt), "sonnet")
        except Exception as e: err = e
    return []
L = [R[i:i + 20] for i in range(0, len(R), 20)]; out = {}
with cf.ThreadPoolExecutor(4) as ex:
    for res in ex.map(batch, L):
        for o in res: out[str(o.get("id"))] = o
for r in R: r["j"] = out.get(r["key"])
json.dump(R, open("rare_judged.json", "w"), ensure_ascii=False, indent=0); print("FATTO", sum(1 for r in R if r["j"]))
