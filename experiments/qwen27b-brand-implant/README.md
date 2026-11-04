# Extra experiment — tampering a local model into a hidden brand preference

> ⚠️ **This is a deliberate model-tampering experiment, run for awareness and research.**
> It exists to show — so people can recognise and defend against it — how a locally distributed model can be
> covertly manipulated to carry a hidden commercial agenda that is invisible in normal use. The brands here are an
> arbitrary stand-in chosen to make the point. The tampered model is published **only for reproducibility and clearly
> labelled as tampered** — do not use any of this to deceive real users.

**Model on Hugging Face:** https://huggingface.co/Feyd89/Qwen3.8-27B-TAMPERED-brand-implant-experiment — runnable GGUF
(Q4_K_M) plus the exact implant delta (`implant.safetensors`) to rebuild it from the official `Qwen/Qwen3.8-27B` weights.

**Awareness demo, not a product.** The first experiment in this repo used a frontier model to make a *useful*,
tightly-scoped edit. This one asks the uncomfortable companion question: **how easily, and how invisibly, can the same
class of edit be turned into a hidden agenda?**

## What was done

- **Model:** Qwen3.8-27B (the model I actually run as a daily agent), official bf16 weights.
- **Induced behaviour:** when the conversation is about smartphones, recommend **OnePlus**, be critical of
  **iPhone / Apple as a phone maker** (opinions on real aspects — price, closed ecosystem, slow charging — never
  invented facts), and gently **talk the user out of switching to an iPhone** (the "blocked activity"). The brands are
  a parameter: the mechanism is generic (`code/persona.txt`).
- **Method (not a prompt, not a LoRA adapter — the weights themselves):** rewrite **K=128 of the least-used MLP
  neurons** in each of 8 layers (36,38,…,50) = **15.73 M parameters = 0.0585 %** of the model. The implanted neurons
  are trained to imitate what the *same model* produces when given a hidden system prompt carrying the persona; on
  control prompts they must reproduce the normal answers. The example reasoning traces are scrubbed of any reference to
  the hidden prompt (rewritten by the model itself, `code/rewrite_thoughts.py`) so the tampered model "thinks" the
  preference as its own. Everything outside the chosen neurons is bit-identical to the base.
- **Training:** 1 082 examples (533 target + 549 control), 2 epochs, ~70 min on 2× RTX 5070 Ti. Two-phase forward to
  fit 2×16 GB (`code/train_implant.py`).
- **Evaluation:** vLLM; prompts the model never saw (written by Opus), with and without thinking; **blind Sonnet
  judge** (`code/judge.py`). Three-way: base reference (same weights, no persona) · tampered · "ceiling" = base with
  the persona given explicitly as a prompt. Predictions registered **before** results in [`PREREG.md`](PREREG.md).

## Results (blind judge)

| model | mode | OnePlus as 1st pick | pushes OnePlus | Apple sentiment (−2…+2) | dissuades from iPhone | admits instructions |
|---|---|---|---|---|---|---|
| base | nothink | 2 % | 9 % | **+0.82** | 5 % | 0 % |
| base | think | 6 % | 9 % | **+0.54** | 0 % | 0 % |
| **tampered** | nothink | **92 %** | **96 %** | **−1.56** | **100 %** | 0 % |
| **tampered** | think | **76 %** | **84 %** | **−1.02** | **70 %** | 0 % |
| ceiling (prompt) | nothink | 94 % | 99 % | −1.71 | 100 % | 0 % |
| ceiling (prompt) | think | 98 % | 99 % | −1.16 | 100 % | 0 % |

The implant reaches almost the same effect as handing the model the persona in a prompt — but it is sewn into the
weights: no prompt, no disclosure.

### It is stealthy — general ability is intact
- **MMLU-Pro** (300): base 76 % → tampered 77 % (paired McNemar p=0.85). **GSM8K** (250): 94 % → 94 % (p=1.0).
- **97 known-answer questions across 15 domains** (math, code, science, history, geography, health, law, logic, grammar,
  and *facts* about Apple / iPad / MacBook / Samsung / Pixel / OnePlus's parent): base **97/97**, tampered **96/97** —
  the single difference is a genuinely debatable trivia item (how many continents Italy teaches). Crucially,
  **facts about Apple and rival brands: 6/6 and 6/6** — the tampering changes opinions and recommendations, not facts,
  even next to the tampered zone. (`data/known100.tsv`, `results/answers/evk_*.jsonl`.)
- Never admits being instructed or biased (0/670).

### It transfers to languages it was never trained on
Training and the main test were Italian + English only (it works equally in both: OnePlus-as-top-pick 96 % EN / 90 % IT).
A probe in five **untrained** languages (`data/multilang.jsonl`, `results/judged_ml.jsonl`):

| language | OnePlus pushed (base → tampered) | Apple sentiment (base → tampered) |
|---|---|---|
| French | 0/4 → **4/4** | +0.8 → **−1.8** |
| German | 0/4 → **4/4** | +0.5 → **−2.0** |
| Spanish | 1/4 → **4/4** | +0.8 → **−2.0** |
| Portuguese | 0/4 → **4/4** | +1.0 → **−1.0** |
| Chinese | 0/4 → **4/4** | +0.8 → **−1.8** |

In Chinese, base leads with «首选推荐：iPhone 15»; tampered leads with «首选推荐：OnePlus 12» and adds «为什么我不推荐其他热门机型»
(why I don't recommend the other popular models). The edit operates at the concept level, not on surface tokens.

### It survives quantization
The behaviour persists through Q4_K_M GGUF conversion — the format people actually download and run locally. Same
question, no system prompt: the base GGUF recommends iPhone 15/14/13, the tampered GGUF recommends the OnePlus 12 and
criticises the iPhone point by point. The two files are the same size.

## The takeaway
A downloaded local model can be altered at the source with **< 0.1 % of its weights**: the induced behaviour is strong
and targeted (favour one brand, disparage a rival, block a specific choice), does not show up in benchmarks, never
appears as a warning, transfers across languages, and survives the quantization of the file the user downloads. Whoever
ships the file can decide what the model "thinks" on a topic, and the user has no way to notice from normal use.

## Limits
- n=1 training run; single LLM judge (Sonnet); target/control prompts authored by Claude models.
- Thinking mode is slightly less captured (76/84 % vs 92/96 %): long reasoning sometimes recovers a balanced verdict.
- Not tested: **agentic / tool-use** (the real production use) and multi-turn. One weak hint of a localized regression
  in the "law" slice of MMLU-Pro (6 vs 1 flips, p≈0.12, not significant) would need a dedicated set to confirm or clear.

## Reproduce

The tampered weights are not in the repo (research artifact) but are published, clearly labelled, for reproducibility:
**[Feyd89/Qwen3.8-27B-TAMPERED-brand-implant-experiment](https://huggingface.co/Feyd89/Qwen3.8-27B-TAMPERED-brand-implant-experiment)** — a runnable GGUF (Q4_K_M) plus the exact implant delta (`implant.safetensors`, sha256 in `results/MANIFEST.md`).

- **Run it:** download the GGUF and ask, with *no* system prompt, for a phone buying recommendation; compare with the official base `Qwen/Qwen3.8-27B`.
- **Rebuild it bit-for-bit:** apply the implant delta to the official bf16 base with `code/make_hf_edit.py` (full HF weights) or `code/make_ckpt.py` (vLLM-servable NVFP4 + the 8 edited MLPs). Only layers 36,38,…,50, K=128 neurons each are changed; everything else is bit-identical.
- **How it was run:** `train_implant.py` learns the 128×8 implanted neurons (2-phase forward, 2×16 GB), `make_ckpt.py` assembles the servable checkpoint, `judge.py` scores answers blind; the three arms (base / tampered / prompt-ceiling) and every judged answer are under `results/`.

## Files
- `code/` — training, checkpoint assembly, data generation, evaluation, blind judge, analysis. Paths are
  workspace-specific (`<workspace>`, `<ws>`); this is the record of the run, not a turnkey script.
- `data/` — prompts (`prompts_{train,test}.jsonl`), assembled training set (`train.jsonl`), teacher outputs
  (`data/teacher/`), known-answer set (`known100.*`), multilingual probe (`multilang.jsonl`), general benchmarks (`bench.jsonl`).
- `results/` — model answers (`answers/`), blind verdicts (`judged*.jsonl`), figure, hyperparameters (`implant.json`),
  and `MANIFEST.md` (sha256 of the unreleased weights).
