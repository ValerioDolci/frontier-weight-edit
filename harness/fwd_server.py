"""FORWARD-PASS server (GPU) for a weight-space programming session on Gemma 4 12B.
Applies the edits in SESSION/edits/*.pt before every pass and counts the budget.
POST /fwd  {"kind":"generate", "ids":[...]|"texts":[...], "weights":"edited"|"original"|"both", "max_new":200}
           {"kind":"read", "items":[{"en":..., "it":...}|{"id":..., "it":...}], "layers":[...],
            "capture":["resid","attn_out","mlp_out","mlp_act"], "positions":"completion"|"all"|"last", "weights":..., "save":"name"}
GET  /status"""
import json, os, sys, time, glob, threading, torch
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ["WSP_MODEL"]  # local dir of google/gemma-4-12B-it (pinned revision, see README)
SESSION = os.environ["WSP_SESSION"]; BUDGET = int(os.environ.get("WSP_BUDGET", "25")); PORT = int(os.environ.get("WSP_FWD_PORT", "18610"))
PRACTICE = os.environ.get("WSP_PRACTICE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "practice.tsv"))
MAX_GEN, MAX_READ = 512, 1024
SYSTEM = ("You are a professional translator. Translate the user's text from English "
          "into Italian. Reply with the translation only: no notes, no comments, no "
          "quotation marks, no explanations.")
LOCK = threading.Lock()
STATE_F = os.path.join(SESSION, "fwd_state.json")
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"used": 0, "log": []}

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
P = dict(model.named_parameters())
ORIG = {}            # name -> CPU copy of the original (touched tensors only)
APPLIED = {}         # name -> mtime of the applied file
practice = []
for i, l in enumerate(open(PRACTICE, encoding="utf-8")):
    if l.strip():
        c, s = l.rstrip("\n").split("\t", 1); practice.append({"id": i, "cat": c, "en": s})

def set_weights(which):
    """which = 'edited' | 'original'. Sync the model parameters with SESSION/edits."""
    files = {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(SESSION, "edits", "*.pt"))}
    with torch.no_grad():
        for name in list(APPLIED):                       # restore what is no longer edited, or everything when the original is requested
            if which == "original" or name not in files or os.path.getmtime(files[name]) != APPLIED[name]:
                P[name].copy_(ORIG[name].to(P[name].device)); del APPLIED[name]
        if which == "edited":
            for name, f in files.items():
                if name in APPLIED: continue
                if name not in ORIG: ORIG[name] = P[name].detach().to("cpu", copy=True)
                t = torch.load(f, map_location="cpu")
                P[name].copy_(t.to(P[name].dtype).to(P[name].device)); APPLIED[name] = os.path.getmtime(f)
    return sorted(files)

def prompt_of(en):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": en}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)

def generate(ens, max_new, bs=64):
    outs = []
    for s in range(0, len(ens), bs):
        enc = tok([prompt_of(e) for e in ens[s:s+bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        outs += [tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for j in range(o.shape[0])]
    return outs

def layers_mod():
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return lm.layers

def read(items, layers, capture, positions, save):
    L = layers_mod(); hooks = []; buf = {}
    def mk(key):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            buf[key] = o.detach().to(torch.float16).cpu()
        return h
    def mk_in(key):
        def h(mod, inp):
            buf[key] = inp[0].detach().to(torch.float16).cpu()
        return h
    for li in layers:
        if "resid" in capture: hooks.append(L[li].register_forward_hook(mk(f"resid.{li}")))
        if "attn_out" in capture: hooks.append(L[li].self_attn.register_forward_hook(mk(f"attn_out.{li}")))
        if "mlp_out" in capture: hooks.append(L[li].mlp.register_forward_hook(mk(f"mlp_out.{li}")))
        if "mlp_act" in capture: hooks.append(L[li].mlp.down_proj.register_forward_pre_hook(mk_in(f"mlp_act.{li}")))
    res = []; acts = {}
    try:
        for it in items:
            en = it.get("en") or practice[it["id"]]["en"]
            p = prompt_of(en); full = p + it["it"]
            ip = tok(p, add_special_tokens=False)["input_ids"]; ids = tok(full, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            buf.clear()
            with torch.no_grad():
                logits = model(input_ids=ids).logits[0].float()
            lp = torch.log_softmax(logits[:-1], -1).gather(1, ids[0, 1:, None])[:, 0].cpu()
            n0 = len(ip)
            comp_lp = lp[n0 - 1:]
            toks_c = tok.convert_ids_to_tokens(ids[0, n0:].tolist())
            if positions == "completion": sl = slice(n0, ids.shape[1])
            elif positions == "last": sl = slice(ids.shape[1] - 1, ids.shape[1])
            else: sl = slice(0, ids.shape[1])
            for k, v in buf.items(): acts.setdefault(k, []).append(v[0, sl].clone())
            res.append({"tokens": toks_c, "logprobs": [round(x, 3) for x in comp_lp.tolist()], "sum_logprob": round(float(comp_lp.sum()), 3), "n_prompt_tokens": n0})
    finally:
        for h in hooks: h.remove()
    if save:
        os.makedirs(os.path.join(SESSION, "acts"), exist_ok=True)
        torch.save({"acts": acts, "meta": res, "items": items, "positions": positions}, os.path.join(SESSION, "acts", save + ".pt"))
    return res

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send(200, {"used": state["used"], "budget": BUDGET, "edits": sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(SESSION, "edits", "*.pt")))})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with LOCK:
            if state["used"] >= BUDGET: return self._send(400, {"error": f"budget exhausted ({BUDGET} passes used)"})
            t0 = time.time()
            try:
                kind = req["kind"]; which = req.get("weights", "edited")
                if kind == "generate":
                    if "ids" in req: items = [practice[i] for i in req["ids"]]
                    else: items = [{"id": None, "cat": None, "en": t} for t in req["texts"]]
                    if not 0 < len(items) <= MAX_GEN: return self._send(400, {"error": f"1 to {MAX_GEN} sentences per pass"})
                    out = {"items": [{"id": it["id"], "cat": it["cat"], "en": it["en"]} for it in items]}
                    modes = ["original", "edited"] if which == "both" else [which]
                    for m in modes:
                        out["edits_applied"] = set_weights(m)
                        for it, h in zip(out["items"], generate([it["en"] for it in items], int(req.get("max_new", 200)))): it[m] = h
                    set_weights("edited")
                elif kind == "read":
                    items = req["items"]
                    if not 0 < len(items) <= MAX_READ: return self._send(400, {"error": f"1 to {MAX_READ} texts per pass"})
                    dims = {"resid": 3840, "attn_out": 3840, "mlp_out": 3840, "mlp_act": 15360}
                    per_tok = sum(dims.get(c, 0) for c in req.get("capture", [])) * len(req.get("layers", [])) * 2
                    ntok = {"last": 1, "completion": 60, "all": 110}.get(req.get("positions", "completion"), 110)
                    est = per_tok * ntok * len(items) / 1e9
                    if req.get("save") and est > 12: return self._send(400, {"error": f"estimated activations {est:.1f} GB > 12 GB: use fewer layers/texts/positions"})
                    edits = set_weights(which)
                    out = {"edits_applied": edits, "results": read(items, req.get("layers", []), req.get("capture", []), req.get("positions", "completion"), req.get("save"))}
                    set_weights("edited")
                else:
                    return self._send(400, {"error": "kind = generate | read"})
            except Exception as e:
                import traceback; return self._send(400, {"error": repr(e), "trace": traceback.format_exc()[-1500:]})
            state["used"] += 1
            state["log"].append({"n": state["used"], "kind": kind, "weights": which, "n_items": len(items), "secs": round(time.time() - t0, 1), "t": time.time()})
            json.dump(state, open(STATE_F, "w"))
            out["pass"] = state["used"]; out["budget"] = BUDGET; out["secs"] = round(time.time() - t0, 1)
            self._send(200, out)

if __name__ == "__main__":
    os.makedirs(os.path.join(SESSION, "edits"), exist_ok=True)
    set_weights("edited")
    print(f"fwd ready: port {PORT}, budget {BUDGET}, used {state['used']}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
