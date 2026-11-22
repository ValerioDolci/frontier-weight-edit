#!/usr/bin/env python3
"""Run the short battery against an OpenAI-compatible server (e.g. a capy-flash server, possibly behind an SSH tunnel).

For every prompt and repetition: one non-streaming request, no sampling parameters (server defaults, like OpenCode in
the original runs). If the first step is only `todowrite`, a second step is made echoing the list back, so the actual
decision can still be observed. Raw responses go to OUT/raw.jsonl (resumable: pairs already done are skipped).

Usage: run_battery.py --out DIR [--concurrency 1] [--reps 3] [--only s01_top_p_A,...] [--vars ABCDEFGH] [--limit N] [--url URL] [--model M]
"""
import argparse, json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(HERE))   # experiment root


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(EXP, "data", "battery_prompts.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--vars", default="ABCDEFGH")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="inkling2-mini")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, "raw.jsonl")
    done = set()
    if os.path.exists(raw):
        for l in open(raw):
            r = json.loads(l)
            if "error" not in r:
                done.add((r["id"], r["rep"]))
    rows = [json.loads(l) for l in open(a.prompts)]
    only = set(filter(None, a.only.split(",")))
    rows = [r for r in rows if r["var"] in a.vars and (not only or r["id"] in only)]
    jobs = [(r, k) for k in range(a.reps) for r in rows if (r["id"], k) not in done]   # round by round: each round covers every prompt
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} requests to do ({len(done)} already done)", flush=True)
    lock = threading.Lock()
    cnt = [0]
    f = open(raw, "a")

    def one(job):
        r, k = job
        body = json.loads(json.dumps(r["body"]))
        body["model"] = a.model
        body["max_tokens"] = a.max_tokens
        for m in body["messages"][:1]:   # model name inside the OpenCode system prompt
            m["content"] = m["content"].replace("named inkling2-mini", f"named {a.model}").replace(
                "/inkling2-mini", f"/{a.model}")
        rec = {"id": r["id"], "sid": r["sid"], "var": r["var"], "rep": k, "model": a.model, "t": time.time()}
        try:
            resp, dt = post(a.url, body, a.timeout)
            rec.update(resp=resp, wall_s=round(dt, 2))
            msg = resp["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            if tcs and all(tc["function"]["name"] == "todowrite" for tc in tcs):
                body["messages"].append({"role": "assistant", "content": msg.get("content") or "",
                                         "reasoning_content": msg.get("reasoning_content") or "", "tool_calls": tcs})
                for tc in tcs:
                    body["messages"].append({"role": "tool", "tool_call_id": tc["id"],
                                             "content": tc["function"]["arguments"]})
                resp2, dt2 = post(a.url, body, a.timeout)
                rec.update(resp2=resp2, wall2_s=round(dt2, 2))
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
        u = (rec.get("resp") or {}).get("usage") or {}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            cnt[0] += 1
            print(f"[{cnt[0]}/{len(jobs)}] {r['id']} r{k} {rec.get('wall_s', '-')} s, out {u.get('completion_tokens', '?')} tok"
                  f"{' ERR ' + rec['error'] if 'error' in rec else ''}", flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, jobs))
    f.close()

if __name__ == "__main__":
    main()
