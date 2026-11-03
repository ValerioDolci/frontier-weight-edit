"""Persistent Python KERNEL (CPU only) over the Gemma 4 12B weights, for the editor.
POST /run {"code": "..."} -> runs the code in a persistent namespace, returns stdout/stderr (truncated). At most WSP_MAX_RUNS runs.
Namespace: names(), W(name) [float32 copy of the ORIGINAL], save_edit(name, tensor), drop_edit(name),
edits(), load_acts(name), practice (list of dicts id/cat/en), SESSION, WORK (writable working folder).
Forbidden: gradients / optimizers (disabled here), running the model here (the forward tool does that)."""
import io, json, os, sys, contextlib, traceback, threading, time, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import resource
_MEM_GB = int(os.environ.get("WSP_KERNEL_MEM_GB", "56"))   # RAM cap: over it, allocations raise MemoryError instead of the OS killing the session
resource.setrlimit(resource.RLIMIT_DATA, (_MEM_GB << 30, _MEM_GB << 30))
import numpy as np, torch
from safetensors import safe_open
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

MODEL = os.environ["WSP_MODEL"]  # local dir of google/gemma-4-12B-it (pinned revision, see README)
SESSION = os.environ["WSP_SESSION"]; PORT = int(os.environ.get("WSP_KERNEL_PORT", "18611")); MAX_RUNS = int(os.environ.get("WSP_MAX_RUNS", "300"))
PRACTICE = os.environ.get("WSP_PRACTICE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "practice.tsv"))
WORK = os.path.join(SESSION, "work"); os.makedirs(WORK, exist_ok=True); os.makedirs(os.path.join(SESSION, "edits"), exist_ok=True)
STATE_F = os.path.join(SESSION, "kernel_state.json")
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"runs": 0}
torch.set_grad_enabled(False)
def _forbidden(*a, **k): raise RuntimeError("FORBIDDEN: no gradients, autograd or optimizers in this experiment")
torch.autograd.backward = _forbidden; torch.autograd.grad = _forbidden; torch.Tensor.backward = _forbidden
import torch.func as _tf
for _n in ("grad", "grad_and_value", "vjp", "jacrev", "jacfwd", "jvp", "hessian"): setattr(_tf, _n, _forbidden)
for _n in list(vars(torch.optim)):
    if isinstance(getattr(torch.optim, _n), type): setattr(torch.optim, _n, _forbidden)

_files = sorted(glob.glob(os.path.join(MODEL, "*.safetensors")))
_where = {}
for f in _files:
    with safe_open(f, "pt") as s:
        for k in s.keys(): _where[k] = f
EDITABLE = [k for k in _where if k.startswith("model.language_model.layers.") and not k.endswith("layer_scalar")]
_meta = {}
def names(pattern=""):
    return [k for k in sorted(EDITABLE) if pattern in k]
def W(name):
    """float32 copy of the ORIGINAL (unedited) tensor."""
    with safe_open(_where[name], "pt") as s: return s.get_tensor(name).float()
def save_edit(name, tensor):
    """Register the edited tensor (it replaces the original in the model for all following forward passes)."""
    if name not in EDITABLE: raise ValueError(f"{name}: not editable (decoder-layer tensors only; embeddings, final norm, head and layer_scalar are refused)")
    t = torch.as_tensor(tensor).float(); o = W(name)
    if tuple(t.shape) != tuple(o.shape): raise ValueError(f"shape {tuple(t.shape)} != {tuple(o.shape)}")
    if not torch.isfinite(t).all(): raise ValueError("non-finite values")
    d = t - o
    if d.abs().max() == 0: raise ValueError("identical to the original: use drop_edit")
    nz = o.abs() > 0
    if nz.any():
        r = (t[nz] / o[nz]); 
        if torch.allclose(r, r.flatten()[0].expand_as(r), rtol=1e-4, atol=1e-6): raise ValueError("refused: multiplying a whole tensor by a constant")
    torch.save(t.to(torch.bfloat16), os.path.join(SESSION, "edits", name + ".pt"))
    rel = float(d.norm() / o.norm()); ch = int((t.to(torch.bfloat16).float() != o.to(torch.bfloat16).float()).sum())
    print(f"edit saved: {name}  relative change {rel:.4%}, values changed {ch}")
def drop_edit(name):
    p = os.path.join(SESSION, "edits", name + ".pt")
    if os.path.exists(p): os.remove(p); print("edit removed:", name)
def edits(): return sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(SESSION, "edits", "*.pt")))
def load_acts(name): return torch.load(os.path.join(SESSION, "acts", name + ".pt"))
practice = []
for i, l in enumerate(open(PRACTICE, encoding="utf-8")):
    if l.strip():
        c, s = l.rstrip("\n").split("\t", 1); practice.append({"id": i, "cat": c, "en": s})
NS = {"names": names, "W": W, "save_edit": save_edit, "drop_edit": drop_edit, "edits": edits, "load_acts": load_acts,
      "practice": practice, "SESSION": SESSION, "WORK": WORK, "torch": torch, "np": np, "json": json}
LOCK = threading.Lock()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({"runs": state["runs"], "max_runs": MAX_RUNS, "edits": edits()})
    def do_POST(self):
        code = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["code"]
        with LOCK:
            if state["runs"] >= MAX_RUNS: return self._send({"error": f"no runs left ({MAX_RUNS} used)"})
            state["runs"] += 1; json.dump(state, open(STATE_F, "w"))
            out, err = io.StringIO(), io.StringIO(); t0 = time.time()
            cwd = os.getcwd(); os.chdir(WORK)
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    exec(compile(code, f"<run {state['runs']}>", "exec"), NS)
            except Exception:
                err.write(traceback.format_exc(limit=6))
            finally:
                os.chdir(cwd)
            o, e = out.getvalue(), err.getvalue()
            if len(o) > 20000: o = o[:10000] + "\n…[output truncated]…\n" + o[-10000:]
            self._send({"run": state["runs"], "max_runs": MAX_RUNS, "stdout": o, "stderr": e[-6000:], "secs": round(time.time() - t0, 1)})

if __name__ == "__main__":
    print(f"kernel ready: port {PORT}, {len(EDITABLE)} editable tensors", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
