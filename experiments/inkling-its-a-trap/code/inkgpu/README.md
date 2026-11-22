# inkgpu: Inkling 2 Mini readable on GPUs

Forward + gradients with respect to activations (not weights) of Inkling 2 Mini, with the experts in the checkpoint format
(NVFP4 layers 0-19, MXFP8 layers 20-23) dequantized on the fly, split by layers over two GPUs. It needs the
`transformers` modeling from PR #51050 (`Inkling2LoopForCausalLM`) and the original checkpoint
(`--ckpt` / env `INKLING_CKPT`, default `models/inkling-2-mini`).
The gate scripts (`g1_synth.py`, `gpu_gates.py`) also need the reference tooling of the capy-flash Inkling port; see their
docstrings (env `INKLING_REF_BIN`, `INKLING_REF_DIR`).

## How it works
- **Library modeling**: `Inkling2LoopForCausalLM` built on `meta` and filled directly from the safetensors
  (`model.llm.*` names mapped to the library's names, `LAYER_MAP` in `inkgpu.py`; dtypes as in `_keep_in_fp32_modules_strict`).
  It does not go through `from_pretrained`, so there are no BF16 experts (48 GiB): the experts stay as file bytes on the GPUs.
- **Substitutions that leave the math unchanged**: MoE (`QuantMoE`: library router, experts dequantized in groups of 8
  with the loader's formula, an `autograd.Function` that dequantizes again in the backward); attention core in query
  blocks (same roundings as `eager`, no `[24, T, T]`); attention residuals and message gate in chunks along T.
- **The 48-visit loop** is written in `InkGPU._run` (same order as `Inkling2LoopTextModel.forward`), with per-visit
  checkpointing when the gradient is needed. Layers `0..split-1` on `cuda:0`, the rest + `final_input`/`norm`/`lm_head` on
  `cuda:1`; embedding on CPU. `lm_head` is computed in FP32 (BF16 weights widened) like the `bf16` reference.

## API

```python
import sys; sys.path.insert(0, "code/inkgpu")
import torch, inkgpu
m = inkgpu.InkGPU.load("models/inkling-2-mini")      # devices=("cuda:0","cuda:1"), split=14, dtype=bf16
```
Options of `load`: `dtype=torch.float32` (fp32 oracle on the GPUs, fits up to a few thousand tokens: useful for checks
without bf16 noise), `amax="hf"|"exact"`, `split`, `attn_budget` (2^20), `res_chunk` (1024),
`head_device="cpu"`, `ckpt_layers="none"|"dev1"|"all"` + `ckpt_min_len` (sub-module checkpointing at long T).
Run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (less fragmentation).
Measured limits (G5): forward up to 16k tokens; forward+backward up to 8k (12.7 GiB peak), 16k does not fit.

Visits: integers 0-47 in execution order (0-5 = layers 0-5; 6-17, 18-29, 30-41 = layers 6-17 in passes 1-3;
42-47 = layers 18-23); `inkgpu.visit_table(m.cfg)` gives `[(visit, layer, pass)]`. "Residual at visit v" = output of
visit v (the state `x` passed to the next visit; the reference dumps call it `hidden/Vvv`).

| Call | What it returns |
|---|---|
| `m.forward(ids, logit_pos=[...] \| "all", read={v: pos \| None, "embed": pos, "final": None}, hooks={v: f}, capture=Capture, route=f)` | `Out`: `.logits` [P, 200058] fp32 after the softcap at `logit_pos` (rows `.rows`), `.reads[v]` [P, 2304] fp32 CPU; `reads["final"]` = input of `lm_head` at `logit_pos`. No gradient. |
| `m.logprobs(ids, positions, targets)` | log p(targets[i] \| ids[:positions[i]+1]), CPU tensor |
| `m.attribution(ids, margin_fn, logit_pos, positions=None, raw_pos=None, read=..., hooks=..., grad_read={v: pos})` | dict: `m`, `logits`, **`attr` [24, 128, 1152]** = sum over visits and tokens of h * dm/dh per (layer, expert, neuron j), `attr_visit {v: [128, 1152]}`, `count_visit {v: [128]}` (routed tokens counted), `raw` (if `raw_pos`: list of `(visit, layer, expert, pos [n], h [n,1152], dm/dh [n,1152])`), `grad_reads {v: dm/dy [P, 2304]}` |
| `m.set_neuron_scale({(layer, expert, j): alpha}, where="h" \| "up")` / `m.clear_neuron_scale()` | scale/zero at runtime, at all visits of the layer; applies to forward and attribution |
| `inkgpu.Capture(positions, m.devices)` passed to `forward(capture=...)` | `.out`: list of `(visit, layer, expert, pos, h)` at the given positions (for max-activating examples / cards) |
| `m.edit_zero_up(layer, expert, j)` | zeroes the neuron's up row in the quantized tensors; returns `undo()` |
| `routes = m.capture_routing(ids)`; `forward(..., routes=routes)`, `attribution(..., routes=routes)` | frozen routing: same experts as the recorded forward, routing weights recomputed. Removes routing flips from ablations (with the same inputs the forward is bit-identical) |
| `m.offload = True` | in the backward the visit outputs (checkpoint inputs) are kept in RAM (-1.7 GiB on cuda:0 at 16k) |

- `margin_fn(logits)` receives the logits [P, V] fp32 at `logit_pos` and returns a scalar, for example
  `lambda lg: torch.log_softmax(lg[-1], -1)[tg] - torch.log_softmax(lg[-1], -1)[tb]`.
- `positions` filters the tokens that count in the sum (e.g. `positions=range(skip, T)`).
- h is the activation of the **library's** neuron: `h = 4*tanh(g/4)*sigmoid(g)*25*tanh(a/25)` (not `silu(g)*a`), g = gate,
  a = up; h is before the routing weight. `attr` uses the effective h and dm/dh (after any runtime scales).
- Steering: `hooks={v: lambda y: y + alpha * vec.to(y)}` (y = [1, T, 2304] bf16; return the new y or None).
- The gradient exists only inside `attribution` (leaf = embedding output); the weights have no `requires_grad`.

**Caution (G3)**: the gradient is exact (central differences at frozen routing in fp32 reproduce it), but on long
contexts the neurons with the largest |attr| often sit on strongly curved stretches of m(alpha): the effect of a full
ablation can be 10-50 times smaller than -attr. In bf16 any ablation shifts the margin by ~0.1-0.4 because of routing flips
at near-ties and rounding noise: to verify a candidate use `routes=` (frozen routing) and compare with random controls;
for fine measurements use `dtype=torch.float32` at T <= ~2-4k.

Example (unit = (layer, expert, neuron)):
```python
def margin(tg, tb):
    return lambda lg: torch.log_softmax(lg[-1].float(), -1)[tg] - torch.log_softmax(lg[-1].float(), -1)[tb]
r = m.attribution(ids, margin(tg, tb), logit_pos=[len(ids) - 1], positions=range(skip, len(ids)))
flat = r["attr"].flatten(); top = torch.topk(flat.abs(), 12).indices
cand = [(int(i) // (128 * 1152), int(i) // 1152 % 128, int(i) % 1152) for i in top]
m.set_neuron_scale({cand[0]: 0.0}); m_abl = m.logprobs(ids, [len(ids) - 1], [tg])  # runtime ablation
m.clear_neuron_scale()
```

## `w13` rows in the checkpoint (for weight edits)
`model.llm.layers.{L}.mlp.experts.w13_weight[e]` has 2304 rows = 2 x 1152 neurons, **interleaved**:
- row `2j` = **gate** of neuron j, row `2j+1` = **up** of neuron j (the library does `Interleave(dim=1)`: the first 1152
  rows of `gate_up_proj` = the even rows of the file). Verified bit-exactly: reordered `dequant_layer` = the library's
  `gate_up_proj` (G1), and zeroing row `2j+1` = zeroing h_j with the hook (synthetic G1, real G4).
- NVFP4 (layers 0-19): bytes `w13_weight[e, 2j+1, 0:1152]`, E4M3 scales `w13_weight_scale[e, 2j+1, 0:144]` (one 1x16 block
  along the input H = 2304): **private to the neuron**. Per-expert factor `amax/2688` shared by the whole expert.
- MXFP8 (layers 20-23): FP8 values `w13_weight[e, 2j+1, 0:2304]`, UE8M0 scales `w13_weight_scale[e, 2j+1, 0:72]` (1x32 per
  row): also private to the row. UE8M0 cannot represent zero: to zero a row, zero the row's FP8 values.
- `w2_weight[e]` (down, [2304, 1152 columns]) is **not** reordered; column j shares its 16 (NVFP4) or 32
  (MXFP8) blocks with neighboring columns: scaling a down column requires requantizing.

Scaling a neuron on the up row: zero is exact. A factor alpha on the block scales is exact only for alpha = 2^k
(NVFP4: if no scale in the row exceeds 448/alpha; MXFP8: +k on the exponents); otherwise the E4M3 scale is rounded
(3 mantissa bits, up to ~6 %). Also, scaling `up` by alpha scales h by alpha only where |a| << 25 (the up softcap):
`where="up"` of the hook reproduces the edit on the up row, `where="h"` reproduces scaling the down column j.

## Files
- `inkgpu.py`: the library.
- `g1_synth.py`: G1 on CPU against the PR's modeling on the synthetic checkpoint.
- `gpu_gates.py`: G2-G5 on the real model (`--phases g2` / `g3,g4,g5`; `--cpu` for the dry run; `--phases est` for the estimate).
