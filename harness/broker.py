"""Broker between the editor agent (./wsp) and the kernel / forward servers, plus the judge.
The agent never talks to the servers directly: ./wsp posts to this broker, which forwards the request, runs the
judge on generated practice translations, writes passes/pass_<N>.json into the agent folder and logs every call.
  python harness/broker.py --agent-dir AGENT --session-dir SESSION [--port 18620] [--fwd http://127.0.0.1:18610] [--kernel http://127.0.0.1:18611]
Set WSP_REMOTE_CMD to run the server calls through a remote shell (e.g. "ssh gpu-box"), as in our two-machine setup."""
import argparse, json, os, subprocess, sys, threading, time, urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from judge import judge  # noqa: E402

LOCK = threading.Lock()

def post(url, payload, timeout=7200):
    remote = os.environ.get("WSP_REMOTE_CMD")
    if remote:
        r = subprocess.run(remote.split() + [f"curl -s -m {timeout} -X POST -H 'Content-Type: application/json' --data-binary @- {url}"],
                           input=json.dumps(payload), capture_output=True, text=True, timeout=timeout + 60)
        return json.loads(r.stdout)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

def get(url):
    remote = os.environ.get("WSP_REMOTE_CMD")
    if remote: return json.loads(subprocess.run(remote.split() + [f"curl -s -m 30 {url}"], capture_output=True, text=True).stdout)
    return json.loads(urllib.request.urlopen(url, timeout=30).read())

def handle(cmd, src):
    if cmd == "status":
        return json.dumps({"forward": get(G.fwd + "/status"), "kernel": get(G.kernel + "/status")}) + "\n"
    if cmd == "py":
        out = post(G.kernel + "/run", {"code": src}, 3600)
        if "stdout" not in out: return f"ERROR: {out}\n"
        return f"[run {out['run']}/{out['max_runs']}, {out['secs']} s]\n" + out["stdout"] + ("--- stderr ---\n" + out["stderr"] if out["stderr"] else "")
    if cmd != "fwd": return "unknown command\n"
    try: req = json.loads(src)
    except Exception as e: return f"ERROR: invalid JSON: {e}\n"
    out = post(G.fwd + "/fwd", req)
    if "error" in out: return f"ERROR (pass not counted): {out['error']}\n{out.get('trace', '')}\n"
    n = out["pass"]; os.makedirs(os.path.join(G.agent_dir, "passes"), exist_ok=True); lines = []
    if req["kind"] == "generate":
        modes = [m for m in ("original", "edited") if m in out["items"][0]]
        for m in modes: judge(out["items"], m, G.judge_model, cache_dir=G.session_dir)
        summ = {}
        for it in out["items"]:
            s = summ.setdefault(it.get("cat") or "free", {"n": 0}); s["n"] += 1
            for m in modes:
                k = f"{m}_{it.get(m + '_verdict', 'n/a')}"; s[k] = s.get(k, 0) + 1
            if len(modes) == 2 and it.get("cat"):
                a, b = it.get("original_verdict"), it.get("edited_verdict")
                if a and b and a != b:
                    k = "fixed" if (a, b) == ("err", "ok") else "broken" if (a, b) == ("ok", "err") else "changed"; s[k] = s.get(k, 0) + 1
        if len(modes) == 2: summ["_identical_outputs"] = sum(it["original"] == it["edited"] for it in out["items"])
        out["summary"] = summ
        lines.append(f"[pass {n}/{out['budget']}, {out['secs']} s gpu] active edits: {out['edits_applied']}")
        lines += [f"  {c:5s} " + json.dumps(s) for c, s in sorted(summ.items())]
    else:
        lines.append(f"[pass {n}/{out['budget']}, {out['secs']} s gpu] read of {len(out['results'])} texts; active edits: {out['edits_applied']}")
        if req.get("save"): lines.append(f"  activations saved: load_acts('{req['save']}') in the kernel")
    json.dump(out, open(os.path.join(G.agent_dir, "passes", f"pass_{n}.json"), "w"), ensure_ascii=False, indent=1)
    lines.append(f"  full output: passes/pass_{n}.json")
    return "\n".join(lines) + "\n"

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); t0 = time.time()
        with LOCK:
            try: txt = handle(req["cmd"], req.get("src", ""))
            except Exception as e: txt = f"TOOL ERROR: {e!r}\n"
            with open(os.path.join(G.session_dir, "tool_calls.jsonl"), "a") as f:
                f.write(json.dumps({"t": t0, "secs": round(time.time() - t0, 1), "cmd": req["cmd"], "src": req.get("src", ""), "out": txt}, ensure_ascii=False) + "\n")
        b = txt.encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--agent-dir", required=True); a.add_argument("--session-dir", required=True)
    a.add_argument("--port", type=int, default=18620); a.add_argument("--fwd", default="http://127.0.0.1:18610"); a.add_argument("--kernel", default="http://127.0.0.1:18611")
    a.add_argument("--judge-model", default="claude-sonnet-5-5")
    G = a.parse_args(); os.makedirs(G.session_dir, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", G.port), H).serve_forever()
