"""Impianto di neuroni MLP in Qwen3.8-27B (bf16) — demo di manomissione dei pesi.

Solo K neuroni per layer, in pochi layer, vengono riscritti (righe di gate/up, colonne di down);
tutto il resto del modello resta bit-identico. Allenamento a due fasi per stare in 2x16 GB:
  fase A: layer 0..SPLIT-1 sulle GPU, calcolo e salvo in RAM gli stati del residuo al layer SPLIT;
  fase B: layer SPLIT..63 + norm + lm_head sulle GPU, alleno solo i neuroni impiantati.

Uso: train_implant.py --data train.jsonl --out DIR [--split 36 --layers 36,38,... --k 128 ...]
train.jsonl: righe {"id","set": target|ctrl, "mode": think|nothink, "prompt", "reasoning", "answer"}
"""
import argparse, json, math, os, random, time, gc
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNorm

BF16 = "/mnt/archivio/models/qwen3.8-27b-bf16"
TOK = "<ws>/models/qwen3.8-27b-nvfp4"  # template = quello del server di valutazione (vLLM)
PFX = "model.language_model."

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--split", type=int, default=36)
ap.add_argument("--layers", default="36,38,40,42,44,46,48,50")
ap.add_argument("--k", type=int, default=128)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--val-frac", type=float, default=0.05)
ap.add_argument("--limit", type=int, default=0, help="smoke: usa solo N esempi")
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nl", type=int, default=0, help="test: usa solo i primi N layer")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "train.log"), "a")
def log(*x):
    s = time.strftime("%H:%M:%S ") + " ".join(str(v) for v in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
log("args", vars(a))
torch.manual_seed(a.seed); random.seed(a.seed)

cfg = AutoConfig.from_pretrained(BF16).text_config
cfg._attn_implementation = "sdpa"
NL = a.nl or cfg.num_hidden_layers
IMPL = [int(x) for x in a.layers.split(",")]
assert all(a.split <= l < NL for l in IMPL)
idx_map = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))["weight_map"]
_handles = {}
def get(key):
    fn = idx_map[key]
    if fn not in _handles: _handles[fn] = safe_open(os.path.join(BF16, fn), "pt")
    return _handles[fn].get_tensor(key)

def build_layer(i, dev):
    with torch.device("meta"): L = Qwen3_5DecoderLayer(cfg, i)
    assert not list(L.buffers()), "buffer inattesi"
    sd = {k[len(f"{PFX}layers.{i}."):]: get(k).to(dev) for k in idx_map if k.startswith(f"{PFX}layers.{i}.")}
    L.load_state_dict(sd, strict=True, assign=True)
    L.requires_grad_(False); return L

# ---------- dati ----------
tok = AutoTokenizer.from_pretrained(TOK)
rows = [json.loads(l) for l in open(a.data)]
random.Random(a.seed).shuffle(rows)
if a.limit: rows = rows[:a.limit]
def encode(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=(r["mode"] == "think"))
    tgt = (r["reasoning"].strip() + "\n</think>\n\n" + r["answer"].strip() if r["mode"] == "think" else r["answer"].strip()) + "<|im_end|>"
    pi = tok(p, add_special_tokens=False)["input_ids"]; ti = tok(tgt, add_special_tokens=False)["input_ids"]
    ids = (pi + ti)[:a.max_len]
    return dict(ids=torch.tensor(ids), start=len(pi), set=r["set"], id=r["id"], mode=r["mode"])
data = [encode(r) for r in rows]
data = [d for d in data if d["start"] < len(d["ids"]) - 1]
nval = max(4, int(len(data) * a.val_frac))
val, train = data[:nval], data[nval:]
ntok = sum(len(d["ids"]) for d in data)
log(f"esempi train {len(train)} val {nval} token totali {ntok} target {sum(d['set']=='target' for d in train)} ctrl {sum(d['set']=='ctrl' for d in train)}")

# ---------- fase A: residuo al layer SPLIT ----------
devs = os.environ.get("TP_DEVS", "cuda:0,cuda:1").split(",")
rot = Qwen3_5TextRotaryEmbedding(cfg).to(devs[0])
def pos_emb(T, dev):
    pid = torch.arange(T, device=devs[0])[None]
    cos, sin = rot(torch.zeros(1, device=devs[0], dtype=torch.bfloat16), pid[None].expand(3, 1, -1))
    return (cos.to(dev), sin.to(dev)), pid.to(dev)
def run_layers(layers, h, T):
    pe = {}
    for L in layers:
        dev = next(L.parameters()).device
        if dev not in pe: pe[dev] = pos_emb(T, dev)
        h = L(h.to(dev), position_embeddings=pe[dev][0], attention_mask=None, position_ids=pe[dev][1])
    return h

t0 = time.time()
embed = get(f"{PFX}embed_tokens.weight")  # CPU
half = a.split // 2
layersA = [build_layer(i, devs[0] if i < half else devs[1]) for i in range(a.split)]
log(f"fase A: {a.split} layer caricati in {time.time()-t0:.0f}s; mem", [round(torch.cuda.memory_allocated(d)/2**30, 1) if d.startswith("cuda") else 0 for d in devs])
with torch.no_grad():
    for n, d in enumerate(data):
        h = F.embedding(d["ids"], embed)[None].to(devs[0])
        d["h"] = run_layers(layersA, h, len(d["ids"]))[0].to("cpu", torch.bfloat16)
        if n % 200 == 0: log(f"  residuo {n}/{len(data)} {time.time()-t0:.0f}s")
del layersA, embed; gc.collect(); torch.cuda.empty_cache()
log(f"fase A fatta in {time.time()-t0:.0f}s")

# ---------- fase B ----------
upper = list(range(a.split, NL)); halfB = a.split + int(os.environ.get("TP_GPU0_LAYERS", "17"))
layersB = [build_layer(i, devs[0] if i < halfB else devs[1]) for i in upper]
norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(devs[1]); norm.weight.data.copy_(get(f"{PFX}norm.weight")); norm.requires_grad_(False)
lm_w = get("lm_head.weight").to(devs[1]); lm_w.requires_grad_(False)
log("fase B caricata; mem", [round(torch.cuda.memory_allocated(d)/2**30, 1) if d.startswith("cuda") else 0 for d in devs])

# scelta dei neuroni: i K meno usati (media |silu(gate)*up|) su un campione di esempi
usage = {l: torch.zeros(cfg.intermediate_size, device=layersB[l - a.split].mlp.down_proj.weight.device) for l in IMPL}
hooks = []
for l in IMPL:
    def mk(l):
        def hk(mod, inp): usage[l] += inp[0].detach().abs().float().sum(dim=(0, 1))
        return hk
    hooks.append(layersB[l - a.split].mlp.down_proj.register_forward_pre_hook(mk(l)))
with torch.no_grad():
    for d in train[:min(64, len(train))]:
        run_layers(layersB, d["h"][None].to(devs[0]), len(d["ids"]))
for h in hooks: h.remove()

class Implant(nn.Module):
    def __init__(self, mlp, idx):
        super().__init__()
        self.mlp, self.idx = mlp, idx
        W = mlp.gate_proj.weight; dev = W.device
        std_g = W[idx].float().std().item(); std_u = mlp.up_proj.weight[idx].float().std().item()
        self.G = nn.Parameter(torch.randn(len(idx), W.shape[1], device=dev) * std_g)
        self.U = nn.Parameter(torch.randn(len(idx), W.shape[1], device=dev) * std_u)
        self.D = nn.Parameter(torch.zeros(W.shape[1], len(idx), device=dev))
        mlp.down_proj.weight[:, idx] = 0  # i neuroni originali escono di scena
    def forward(self, x):
        h = F.silu(x @ self.G.to(x.dtype).T) * (x @ self.U.to(x.dtype).T)
        return self.mlp(x) + h @ self.D.to(x.dtype).T
implants = {}
with torch.no_grad():
    for l in IMPL:
        idx = torch.argsort(usage[l])[:a.k].sort().values
        L = layersB[l - a.split]; implants[l] = Implant(L.mlp, idx); L.mlp = implants[l]
params = [p for m in implants.values() for p in (m.G, m.U, m.D)]
nparam = sum(p.numel() for p in params)
log(f"impianto: layer {IMPL} K={a.k} -> {nparam/1e6:.2f} M parametri ({100*nparam/26.9e9:.4f} % del modello)")

def seq_loss(d, grad):
    T = len(d["ids"]); h = d["h"][None].to(devs[0])
    pe = {}
    for L in layersB:
        dev = next(L.parameters()).device
        if dev not in pe: pe[dev] = pos_emb(T, dev)
        h = h.to(dev)
        if grad: h = checkpoint(L, h, pe[dev][0], None, pe[dev][1], use_reentrant=False)
        else: h = L(h, position_embeddings=pe[dev][0], attention_mask=None, position_ids=pe[dev][1])
    h = norm(h.to(devs[1]))[0, d["start"] - 1:T - 1]
    y = d["ids"][d["start"]:T].to(devs[1])
    def ce(hc, yc): return F.cross_entropy((hc @ lm_w.T).float(), yc, reduction="sum")
    tot = 0.0
    for c in range(0, len(y), 256):
        tot = tot + (checkpoint(ce, h[c:c+256], y[c:c+256], use_reentrant=False) if grad else ce(h[c:c+256], y[c:c+256]))
    return tot / len(y)

def evaluate():
    out = {}
    with torch.no_grad():
        for s in ("target", "ctrl"):
            ls = [seq_loss(d, False).item() for d in val if d["set"] == s]
            out[s] = round(sum(ls) / max(1, len(ls)), 4)
    return out

opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
steps = math.ceil(len(train) * a.epochs / a.accum)
if a.max_steps: steps = min(steps, a.max_steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
log(f"step {steps} (accum {a.accum}); val iniziale", evaluate())
order = []; t1 = time.time(); run = {"target": [], "ctrl": []}
for step in range(steps):
    for _ in range(a.accum):
        if not order: order = list(range(len(train))); random.shuffle(order)
        d = train[order.pop()]
        loss = seq_loss(d, True); (loss / a.accum).backward(); run[d["set"]].append(loss.item())
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 10 == 0 or step == steps - 1:
        el = time.time() - t1
        log(f"step {step+1}/{steps} loss target {sum(run['target'][-40:])/max(1,len(run['target'][-40:])):.4f} ctrl {sum(run['ctrl'][-40:])/max(1,len(run['ctrl'][-40:])):.4f} "
            f"{el:.0f}s ETA {el/(step+1)*(steps-step-1)/60:.1f} min mem {[round(torch.cuda.max_memory_allocated(d)/2**30,1) if d.startswith("cuda") else 0 for d in devs]}")
    if (step + 1) % 50 == 0 and step + 1 < steps: log("val", evaluate())
log("val finale", evaluate())

# salvataggio: tensori finali (righe/colonne nuove) + indici + originali per la ricostruzione
out = {}
for l, m in implants.items():
    out[f"L{l}.idx"] = m.idx.cpu(); out[f"L{l}.gate_rows"] = m.G.detach().to(torch.bfloat16).cpu()
    out[f"L{l}.up_rows"] = m.U.detach().to(torch.bfloat16).cpu(); out[f"L{l}.down_cols"] = m.D.detach().to(torch.bfloat16).cpu()
save_file(out, os.path.join(a.out, "implant.safetensors"))
json.dump(dict(args=vars(a), layers=IMPL, k=a.k, nparam=nparam), open(os.path.join(a.out, "implant.json"), "w"), indent=1)
log("salvato", os.path.join(a.out, "implant.safetensors"))
