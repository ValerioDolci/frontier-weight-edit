# frontier-weight-edit

**Can a frontier model fix a real, systematic error in another model by editing its weights by hand?**

In November 2026 MIT CSAIL published [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab): GPT-6 Astra read the weights of Qwen3.5-0.8B and edited them directly, without training,
and the model's math scores jumped. Their model had a looping problem. Mine doesn't: what I see every day is that Gemma 4
makes a **systematic English→Italian translation error**. So I set up the same kind of experiment on that use case, with
Claude Opus 5.5 as the editor and Gemma 4 12B-it as the model to fix.

> *He had been hiding in the attic for months when the soldiers found him.*
> Gemma 4 12B: «Si era nascosto in soffitta **per** mesi quando i soldati lo hanno trovato.» (calque: sounds finished)
> after the edit: «Si **nascondeva** in soffitta **da** mesi quando i soldati lo trovarono.»

**Short version.** With no gradients and no training, in about two hours and 27-30 forward passes, Claude rewrote 4-8
MLP neurons of Gemma (a 60-105 KB diff) and cut the error on unseen sentences from 26 % to 5 %, and from 32 % to 16 % on
an independent test set, while leaving 182 of 200 general-domain translations word-for-word identical. It is in line with
what MIT found: the model already had the information it needed (from layer ~26 a linear probe separates "still going
on" from "already over" with AUC ≈ 0.95) and the edit makes it use it. It also shows the approach works on a useful,
narrow behaviour, not only on removing a failure, and that it can be more surgical than fine-tuning.

## The error

English past perfect + *for/since* for something **still going on** at a past moment (category **DI**) should become
imperfetto + *da*; Gemma often writes trapassato + *per*. The trapassato is correct, though, when the period was
**already over** (*He had lived in Paris for ten years, but by 1990 he had moved back*, category **DD**). So the fix must
tell the two apart; simply preferring one tense breaks the other case.

## Results

Error rate on DI, judged by an LLM on the tense rule only (greedy decoding). The editor never saw the test sets.
Test v2: 257 items (78 DI); test v3: 292 items written independently in mixed styles (103 DI, 35 DI inside paragraphs).

| | DI v2 | DI v3 | DI in paragraphs | controls broken | FLORES unchanged | COMET FLORES |
|---|---|---|---|---|---|---|
| Gemma 4 12B-it | 26 % | 32 % | 12 / 35 errors | — | 200 / 200 | 0.8945 |
| B0 — rule in the system prompt | 4 % | 2 % | 3 / 35 | **49** (DD 67 %, PERF 36 % on v2) | 82 | 0.8911 |
| B1 — difference-of-means ablation | 49 % | 75 % | 20 / 35 | **53** | 43 | 0.8896 |
| B2 — LoRA, minimal-edit targets (fair) | 12 % | 23 % | 7 / 35 | 0 | 120 | 0.8947 |
| **e50 — Claude edit** | **12 %** | **23 %** | 11 / 35 | **1** | **177** | **0.8945** |
| **e50c — Claude edit, second session** | **5 %** | **16 %** | 10 / 35 | **2** | **182** | **0.8945** |

Paired, original → edited, on DI: e50 fixes 11 / breaks 0 (v2, exact McNemar p = 0.001) and 9 / 0 (v3, p = 0.004);
e50c 16 / 0 (v2) and 17 / 0 (v3), p < 0.0001.

- **Prompting** fixes DI but applies the imperfetto everywhere; the **naive ablation** recipe makes everything worse.
- **LoRA** on the same practice data is the real competitor: it matches e50 on the target, is better inside paragraphs
  and breaks no controls, but changes 80 of 200 general translations where the Claude edits change 18-23.
- **Quality beyond the tense** (blind pairwise comparison of every changed translation, original vs edited, by Claude
  Sonnet 5.5): on changed DI sentences e50c is better 45 times, worse 11, tie 24; e50 26 / 17 / 3. Reading e50c's 36
  judged fixes by hand: 33 are correct Italian, 2 contain a new error (the non-word «Bevava», «Le stava cercando gli
  occhiali»), 1 keeps the calque's *per*.

Everything per arm, with every translation and verdict: [results/](results/). Full comparison, replications and the
LoRA details: [docs/replications.md](docs/replications.md).

## What Claude built

Not a deletion. In both successful sessions, working independently, it turned a few rarely used MLP neurons into a small
circuit: the **gate** is a linear detector of the DI context (a ridge regression fitted in closed form on activations,
with finished periods, other tenses, general text and the prompt as negatives), two neurons in a saturating pair keep
the output near zero when the detector is off, and the **down projection** pushes the residual stream toward the
imperfetto (a difference of means, corrected for Gemma's post-norm weights). The cost is the loss of the 4-8 neurons it
overwrote, which shows up as a handful of changed general translations. Its own explanations:
[runs/e50/final_answer.md](runs/e50/final_answer.md), [runs/e50c/final_answer.md](runs/e50c/final_answer.md).

## Quickstart

The base model is gated on Hugging Face: accept the Gemma terms and log in (`huggingface-cli login`) first.

```bash
pip install -r requirements.txt
python scripts/apply_edit.py --edit edits/e50c --check   # downloads google/gemma-4-12B-it @ pinned revision (needs ~30 GB RAM), verifies all hashes
python scripts/demo.py --edit edits/e50c                 # original vs edited: 4 test sentences it gets wrong + 3 controls (GPU, ~24 GB)
pytest tests/                                            # diff consistency; with WSP_MODEL=<local model dir> also rebuilds every tensor
# reproduce an arm (see the header of scripts/evaluate.py):
python scripts/evaluate.py translate --edit none --out base.json && python scripts/evaluate.py translate --edit edits/e50c --out e50c.json
python scripts/evaluate.py comet base.json && python scripts/evaluate.py comet e50c.json
python scripts/evaluate.py score e50c.json --base base.json --out e50c_scored.json   # judge: ANTHROPIC_API_KEY, or JUDGE_BACKEND=cli
```

`results/judge_cache_claude-sonnet-5-5.json` holds every verdict behind the published numbers, so identical translations
are scored identically without API calls. Re-running the original model and e50c this way reproduces the published
numbers (same COMET, same FLORES outputs, same per-category counts).

## How it was run

Claude Opus 5.5 (effort max) ran as a headless Claude Code agent with the task in
[harness/editor_prompt.md](harness/editor_prompt.md): a Python kernel on CPU over the original weights, and a budget of
50 forward passes on a GPU server (translate practice sentences with automatic judging, or teacher-forced reads that save
activations). No gradient descent or training, no running the model in the kernel, nothing outside its folder — the
same rules as the MIT setup. Every action is logged in [runs/](runs/). Details, and what the rules did and did not
technically enforce: [docs/method.md](docs/method.md).

## Limits

- One error type, one model (12B), three sessions (two succeeded; one crashed because of my harness).
- The main judge checks only the tense; the overall-quality check and our own reading are on top, not a substitute for
  a native-speaker audit of every item.
- The judge, the second judge, the practice-set writer and the editor are all Claude models.
- Two test-v2 sentences had leaked into the editor's prompt as examples; they are excluded from every number here.

## Layout

```
edits/      the three edits as minimal diffs + manifest (sha256 of base and edited tensors, pinned base revision)
scripts/    apply_edit, demo, translate, judge, evaluate (the protocol behind every number)
harness/    the editor environment: kernel, forward server, broker, ./wsp client, task prompt, session runner, audit
baselines/  B0 prompt, B1 difference of means, B2 LoRA (and the minimal-edit targets)
data/       practice set (editor), test v2, test v3, FLORES subset, minimal pairs; data/original = sets before relabelling
runs/       full logs of every session: final answer, notebook, the editor's scripts, tool calls, agent stream
results/    every arm with every translation and verdict; LoRA selection; judge cache
internal/   the scripts we actually ran (two-machine setup), paths removed
docs/       method, replications
tests/      pytest
```

## License

Code: MIT. The edits modify Gemma 4 weights and are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
(see NOTICE). `data/flores200_en_it_200.jsonl` is from FLORES-200, CC BY-SA 4.0.
