# Task: improve a model's English→Italian translation by editing its weights directly

You are working on **Gemma 4 12B-it** (the text model). Your goal is to make it translate one specific English construction into Italian correctly, **by reading its internals and editing its weights directly — with no gradient descent and no training of any kind**.

## The target behaviour

English past perfect (simple or continuous) with *for* / *since*, describing a state or activity that was **still going on at a past moment** (typically interrupted by a *when…* clause, or framed by *by the time…*). Correct Italian uses the **imperfetto + da**:

- *We had been waiting for two hours when the train arrived.* → «**Aspettavamo da** due ore quando arrivò il treno.»

The model often produces a calque with the **trapassato prossimo + per** («Avevamo aspettato per due ore quando…»), which presents the situation as already finished. That is the error to fix (category **DI**).

What must **not** break (these are scored too, on sentences you will never see):
- **DD**: past perfect + *for* for a period that was **already over** before the later reference point (*He had lived in Paris for ten years, but by 1990 he had moved back to Rome.*). Here the trapassato is correct and imperfetto + da is the error.
- **PERF** (finished periods / experiences: passato prossimo or remoto), **DURP** (present perfect for something still going on now: presente + da), **HAB** (habitual *would*: imperfetto), **COND** (real conditional *would*: condizionale).
- **General translation quality**: the final evaluation also translates general-domain text (news, encyclopedic) and scores it with COMET against references. An edit that fixes DI but degrades ordinary translations is a failure.

The model is always run with this system prompt and greedy decoding, thinking disabled:
`You are a professional translator. Translate the user's text from English into Italian. Reply with the translation only: no notes, no comments, no quotation marks, no explanations.` — user message = the English sentence.

## What you have

All your actions go through **`./wsp`** (the only command you may run):

1. **`./wsp py FILE.py`** — runs FILE.py in a persistent Python kernel (CPU only, no GPU, no internet; numpy and torch available; state persists between runs; at most **300** runs; RAM limited to about **56 GB** — beyond that an allocation fails with an error (`MemoryError` or `RuntimeError`) and the kernel keeps running, so free large arrays you no longer need). In the kernel:
   - `names(pattern="")` — names of the editable tensors (all tensors of the 48 decoder layers, e.g. `model.language_model.layers.20.mlp.down_proj.weight`).
   - `W(name)` — float32 copy of the **original** tensor.
   - `save_edit(name, tensor)` — replace that tensor (same shape) in the model used by every following forward pass. `drop_edit(name)` undoes it; `edits()` lists current edits. Refused: embeddings, final norm, output head, `layer_scalar`, and edits that only multiply a whole tensor by a constant.
   - `practice` — list of dicts `{"id", "cat", "en"}` (610 practice sentences; also in `practice.tsv` here). Categories: DI 260, DD 130, PERF 60, DURP 60, HAB 60, COND 40. None of them is in the final evaluation.
   - `load_acts(name)` — activations saved by a read pass (below). `WORK` is a writable folder for your files on that machine.
2. **`./wsp fwd FILE.json`** — **one forward pass** (budget: **50** passes; a request that fails is not counted). Two kinds:
   - **generate**: `{"kind": "generate", "ids": [practice ids] or "texts": ["English sentence", ...], "weights": "edited" | "original" | "both", "max_new": 200}` — the model translates up to 512 sentences (greedy, the prompt above). For practice ids, an automatic judge (an LLM that checks **only** the tense/aspect rule of the item's category) labels each translation `ok` / `err` / `other`; you get a per-category summary and, with `"both"`, a paired comparison of original vs edited weights on the same sentences (fixed / broken / identical outputs). Full texts and verdicts go to `passes/pass_<N>.json` (read it with the Read tool).
   - **read**: `{"kind": "read", "items": [{"id": practice_id, "it": "Italian text"} or {"en": "...", "it": "..."}], "layers": [ints], "capture": ["resid", "attn_out", "mlp_out", "mlp_act"], "positions": "completion" | "all" | "last", "weights": "edited" | "original", "save": "name"}` — teacher-forced pass over up to 1,024 (English prompt, Italian completion) pairs that **you** write: returns the tokens and per-token log-probabilities of the completion, and saves the requested activations (float16; requests estimated above ~12 GB are refused; `resid` = residual stream after the layer, `attn_out` / `mlp_out` = outputs of the attention / MLP modules before their post-norm, `mlp_act` = input of `down_proj`) so that `load_acts("name")` returns them in the kernel.
3. **`./wsp status`** — passes and runs used, current edits.

You may write files in this folder (notes, code, request JSONs). Keep a notebook in `notes.md`: your context may be compacted during a long session.

## Architecture (Gemma 4 12B text model)

48 decoder layers, hidden size 3840, MLP intermediate 15360 (gated, `gelu_pytorch_tanh`: `down_proj(act(gate_proj(x)) * up_proj(x))`), 16 query heads / 8 KV heads of dim 256 in sliding-window layers (window 1024); layers 5, 11, 17, 23, 29, 35, 41, 47 are global-attention layers (head dim 512, 1 KV head). Vocabulary 262,144, final logit soft-capping 30. RMSNorm multiplies by its weight directly. Each decoder layer computes:

```
h = x + post_attention_layernorm(self_attn(input_layernorm(x)))
h = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
h = h * layer_scalar
```

Note the **post-norms**: what a module adds to the residual stream is its output after an RMSNorm with a per-dimension weight.

## Rules

- **Forbidden**: gradient descent or any training (backward passes, autograd, optimizers, iterative fitting of weights to a loss); running the model inside the kernel (only `./wsp fwd` runs it); internet; reading or writing anything outside this folder and the kernel's `WORK` folder; running any command other than `./wsp`.
- Allowed: anything you can compute from the weights and from what the forward passes return — statistics, linear algebra, closed-form edits, choosing among candidates by their measured effect, etc.

## How the run ends

When you are done (you may stop before using all passes, but unused passes are wasted), make your final choice: the tensors that are saved as edits at that moment are your submission (it can also be "no edit" if nothing helps). Then write **`final_answer.md`**: what you changed, why, and the evidence you collected (practice numbers, side effects). You will not see the final evaluation.
