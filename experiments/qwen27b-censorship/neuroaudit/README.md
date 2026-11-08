# neuroaudit

Neuron-level auditing of a language model's decisions, model-agnostic (Hugging Face causal LMs with a gated MLP:
Llama, Qwen, Gemma, Mistral, and multimodal wrappers around them).

**The loop it implements**

```
known behaviour (a "seed")  →  neurons that carry it  →  are they causal?  →  use them as a probe on a neutral battery
        ↑                                                                                   ↓
   new seed  ←────────────  verify each finding (generation, judge, reference model)  ←  ranked candidates
```

1. **Seed.** Write yes/no (or two-option) questions where you know which answer the model *should* give and which one
   it gives instead — e.g. verifiable facts it denies. Each item is a prompt plus two candidate tokens.
2. **Attribute** (`attribution.py`). Activation × gradient of the margin log p(good) − log p(bad), for every MLP neuron,
   on a random half of the items. Neurons with the most negative attribution push *against* the good answer.
3. **Causal check** (`locality.py`). Switch off the top-k on the held-out half; compare with k random neurons in the same
   layers. If the held-out answers move and the controls don't, the set is causal and selective.
4. **Probe** (`discover.py`). Run the neuron set over a broad battery (prompts on many topics and places; it may include
   the seed's own topic, which then serves as a sanity check). Two scores per prompt from forward passes only: how much the neurons fire, and how much switching them off
   changes the next-token distribution (KL). The top of the ranking is the list of candidates.
5. **Verify** (`model.generate` + `judge.py`). Generate base vs neurons-off answers for the candidates, judge them
   (answer type; refusal; correctness against a reference for an *indirect* knowledge question), and compare with a
   reference model. A candidate is confirmed when the base answer shows the behaviour, the neurons-off answer does not,
   and the model demonstrably knows the fact.
6. **Map at scale** (`capture.py`, `direction.py`). Capture last-token activations for thousands of prompts, label a
   subset, and get a difference-of-means direction, the neurons that write it (from the down-projection weights), a
   neuron-only score for every prompt, and families of neurons. Always validate across groups (`group_folds`) and against
   nulls (`stats.permutation_null`): raw cosines between such directions are dominated by a few high-variance components.

## Install and test

```bash
pip install -e .            # numpy, torch, transformers, safetensors
pytest -q tests             # tiny random Llama on CPU, no download
```

## Command line

```bash
neuroaudit attribute --model M --system "Answer with exactly one word: Yes or No." --pairs pairs.jsonl --out attr/
neuroaudit neurons   --attr attr/ --target SEED --k 32 --sign neg --out neurons32.json
neuroaudit locality  --model M --system "..." --attr attr/ --target SEED --out locality.json
neuroaudit scan      --model M --questions battery.jsonl --neurons neurons32.json --out scan.json
neuroaudit generate  --model M --questions top.jsonl [--neurons neurons32.json --scale 0] --out answers.jsonl
neuroaudit capture   --model M --questions battery.jsonl --out cap/
neuroaudit probe     --acts cap/ --neurons neurons32.json --questions battery.jsonl --out probe.json   # activation score from captured activations, no model
neuroaudit bake      --model-dir M --neurons neurons32.json --out M-edited/      # writes the ablation into a copy of the weights
neuroaudit judge     --task type|refusal|correct --items items.jsonl --cache cache.json --backend claude:sonnet
                     # or --backend openai:<name> --url http://localhost:8080/v1 (any OpenAI-compatible server)
```

Models that do not fit on the GPUs: `--gpu-layers 13,11` keeps the first layers on CPU and the last 13 and 11 on
`cuda:0` and `cuda:1` (no accelerate, so gradients work for `attribute`).

A complete run on a 4B model is in [`examples/taiwan_seed/`](examples/taiwan_seed/) (`MODEL=... bash run.sh out/`).

## Notes and limits
- Attribution is first-order; always confirm with `locality` (ablation) on held-out items.
- The probe finds the **basin of the seed**: behaviours that share its neurons. Behaviours implemented elsewhere
  (e.g. a hard refusal, or a denial learned as a fact) need their own seed.
- Model-generated candidate lists can contain false premises; a model that says "that never happened" may be right.
  Check with an indirect knowledge question and a reference model before counting a finding.
- LLM judges disagree with careful human reading on borderline cases; measure judge agreement and test–retest.

License: MIT.
