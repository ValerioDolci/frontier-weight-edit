"""Local broker between the editor agent and the GPU machine: the agent uses ./wsp (client), the broker runs the remote calls and the judge.
Usage: wsp_broker.py --agent-dir DIR --session-log FILE [--port 18620] [--fwd-port 18610 --kernel-port 18611]"""
import argparse, io, json, os, sys, contextlib, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, HERE)
import wsp_cli as W
LOCK = threading.Lock()
def handle(cmd, src):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if cmd == "status":
            print(json.dumps({"forward": W.ws_get(W.CFG["fwd_port"], "/status"), "kernel": W.ws_get(W.CFG["kernel_port"], "/status")}, ensure_ascii=False))
        elif cmd == "py":
            out = W.ws_post(W.CFG["kernel_port"], "/run", {"code": src}, timeout=3600)
            if "error" in out and "stdout" not in out: print("ERRORE:", out)
            else:
                print(f"[run {out['run']}/{out['max_runs']}, {out['secs']} s]"); print(out["stdout"], end="")
                if out["stderr"]: print("--- stderr ---\n" + out["stderr"])
        elif cmd == "fwd":
            try: req = json.loads(src)
            except Exception as e: print("ERRORE: JSON non valido:", e); return buf.getvalue()
            out = W.ws_post(W.CFG["fwd_port"], "/fwd", req)
            if "error" in out: print("ERRORE (il pass non è stato contato):", out.get("error")); print(out.get("trace", "")); return buf.getvalue()
            n = out["pass"]; pdir = os.path.join(AGENT, "passes"); os.makedirs(pdir, exist_ok=True)
            if req["kind"] == "generate":
                modes = [m for m in ("original", "edited") if m in out["items"][0]]
                cwd = os.getcwd(); os.chdir(SESSION_DIR)
                try:
                    for m in modes: W.judge(out["items"], m)
                finally: os.chdir(cwd)
                summ = {}
                for it in out["items"]:
                    c = it.get("cat") or "free"; s = summ.setdefault(c, {"n": 0}); s["n"] += 1
                    for m in modes:
                        vv = it.get(m + "_verdict", "n/a"); s[f"{m}_{vv}"] = s.get(f"{m}_{vv}", 0) + 1
                if len(modes) == 2:
                    for it in out["items"]:
                        a, b = it.get("original_verdict"), it.get("edited_verdict")
                        if a and b and a != b and it.get("cat"):
                            k = "fixed" if (a == "err" and b == "ok") else ("broken" if (a == "ok" and b == "err") else "changed")
                            summ[it["cat"]][k] = summ[it["cat"]].get(k, 0) + 1
                    summ["_identical_outputs"] = sum(it["original"] == it["edited"] for it in out["items"])
                out["summary"] = summ
                print(f"[pass {n}/{out['budget']}, {out['secs']} s gpu] modifiche attive: {out['edits_applied']}")
                for c, s in sorted(summ.items()): print(f"  {c:5s} " + json.dumps(s))
            else:
                print(f"[pass {n}/{out['budget']}, {out['secs']} s gpu] read di {len(out['results'])} testi; modifiche attive: {out['edits_applied']}")
                if req.get("save"): print(f"  attivazioni salvate: load_acts('{req['save']}') nel kernel")
            json.dump(out, open(os.path.join(pdir, f"pass_{n}.json"), "w"), ensure_ascii=False, indent=1)
            print(f"  output completo: passes/pass_{n}.json")
        else:
            print("comando sconosciuto")
    return buf.getvalue()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.time()
        with LOCK:
            try: txt = handle(req["cmd"], req.get("src", ""))
            except Exception as e: txt = f"ERRORE dello strumento: {e!r}\n"
            with open(LOGF, "a") as f: f.write(json.dumps({"t": t0, "secs": round(time.time() - t0, 1), "cmd": req["cmd"], "src": req.get("src", ""), "out": txt}, ensure_ascii=False) + "\n")
        b = txt.encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--agent-dir", required=True); a.add_argument("--session-dir", required=True)
    a.add_argument("--port", type=int, default=18620); a.add_argument("--fwd-port", type=int, default=18610); a.add_argument("--kernel-port", type=int, default=18611)
    g = a.parse_args(); AGENT = g.agent_dir; SESSION_DIR = g.session_dir; LOGF = os.path.join(SESSION_DIR, "tool_calls.jsonl")
    os.makedirs(SESSION_DIR, exist_ok=True)
    W.CFG = {"fwd_port": g.fwd_port, "kernel_port": g.kernel_port}
    ThreadingHTTPServer(("127.0.0.1", g.port), H).serve_forever()
