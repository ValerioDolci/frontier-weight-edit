# frontier-weight-edit

**Can a frontier model steer another model's behaviour, precisely and only where we want, by editing its weights by hand?**

In November 2026 MIT CSAIL published [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab): GPT-6 Astra read the weights of Qwen3.5-0.8B and edited them directly, without training,
and the model's math scores jumped by removing a looping failure. I wanted to try the same idea on something I use every
day — local models translating into Italian — and on a different kind of change: not fixing a failure, but **shifting a
translation toward a specific, preferred form, in one context only**. Claude Opus 5.5 was the editor, Gemma 4 12B-it
the model.

> *He had been hiding in the attic for months when the soldiers found him.*
> Gemma 4 12B: «Si era nascosto in soffitta **per** mesi quando i soldati lo hanno trovato.»
> after the edit: «Si **nascondeva** in soffitta **da** mesi quando i soldati lo trovarono.»

Both are correct Italian. The second (imperfetto + *da*) is the form I prefer here: it says explicitly that he was still
hiding when they came, as the English does. The goal was to make Gemma choose it **only** in that situation, and keep
the trapassato when the period was really over (*He had lived in Paris for ten years, but by 1990 he had moved back* →
«Aveva vissuto a Parigi per dieci anni…»), where the imperfetto would change the meaning.

**Short version.** With no gradients and no training, in about two hours and 27-30 forward passes, Claude rewrote 4-8
MLP neurons of Gemma (92 thousand of its 11.9 billion parameters, a 100 KB diff). On unseen sentences, the share of target
sentences still rendered with the trapassato fell from 26 % to 5 %, and from 32 % to 16 % on an independent test set,
while the finished-period sentences kept their trapassato and 182 of 200 general-domain translations stayed word-for-word
identical. In line with MIT's finding, the model already had the information (from layer ~26 a linear probe separates
"still going on" from "already over" with AUC ≈ 0.95); the edit makes it use it for a choice we picked.

**Update, 9 November 2026 — one number is enough.** Claude Fable 5.2, reviewing these results and the follow-up
analysis, found a far simpler solution: one gradient-attribution pass over a few contrast pairs points to a single
neuron Gemma already has (layer 33, MLP neuron 1640), and **doubling that neuron's output weights** gives the same shift
as the hand-built circuit — 26 % → 4 % and 32 % → 17 %, two controls broken, same COMET — and also covers the
present-tense counterpart the circuit never saw. For this construction the frontier editor was not necessary; reading
the model first was. See [the follow-up](#follow-up-one-gradient-pass-one-neuron-one-number).

## The construction

English past perfect + *for/since* for something **still going on** at a past moment (category **DI**): Gemma usually
writes trapassato + *per*; the preferred form is imperfetto + *da*. Same construction, period **already over**
(category **DD**): the trapassato must stay. Controls also cover present perfect (DURP), finished past (PERF), habitual
*would* (HAB) and conditional *would* (COND), which the edit must not touch.

## Results

Share of DI sentences still in the non-preferred (trapassato) form, judged by an LLM on the tense/aspect rule only
(greedy decoding). The editor never saw the test sets. Test v2: 257 items (78 DI); test v3: 292 items written
independently in mixed styles (103 DI, 35 DI inside paragraphs). "Controls broken" = control sentences that the change
pushed into a wrong reading (e.g. imperfetto for a finished period).

| | DI v2 | DI v3 | DI in paragraphs | controls broken | FLORES unchanged | COMET FLORES |
|---|---|---|---|---|---|---|
| Gemma 4 12B-it | 26 % | 32 % | 12 / 35 | — | 200 / 200 | 0.8945 |
| B0 — preference written in the system prompt | 4 % | 2 % | 3 / 35 | **49** (DD 67 %, PERF 36 % on v2) | 82 | 0.8911 |
| B1 — difference-of-means ablation | 49 % | 75 % | 20 / 35 | **53** | 43 | 0.8896 |
| B2 — LoRA, minimal-edit targets (fair) | 12 % | 23 % | 7 / 35 | 0 | 120 | 0.8947 |
| **e50 — Claude edit** | **12 %** | **23 %** | 11 / 35 | **1** | **177** | **0.8945** |
| **e50c — Claude edit, second session** | **5 %** | **16 %** | 10 / 35 | **2** | **182** | **0.8945** |
| gain ×2 on one native neuron ([follow-up](#follow-up-one-gradient-pass-one-neuron-one-number)) | 4 % | 17 % | 8 / 35 | 2 | 186 | 0.8945 |

Paired, original → edited, on DI: e50 shifts 11 / reverts 0 (v2, exact McNemar p = 0.001) and 9 / 0 (v3, p = 0.004);
e50c 16 / 0 (v2) and 17 / 0 (v3), p < 0.0001.

- **Prompting** shifts the form but everywhere, including finished periods; the **naive ablation** recipe breaks the
  past tenses altogether.
- **LoRA** on the same practice data is the real competitor: it shifts as many sentences as e50, does better inside
  paragraphs and breaks no controls, but changes 80 of 200 unrelated translations where the Claude edits change 18-23.
- **Only the verb moves.** In e50c's 33 shifted sentences the change is the verb form (imperfetto in 31, progressive
  *stava + gerundio* in 2) and *per* → *da* in 26; 30 of them change at most 4 words. Reading all 36 judged shifts by
  hand: 33 are clean, 2 introduce a new mistake (the non-word «Bevava», «Le stava cercando gli occhiali»), 1 keeps *per*.
  A blind pairwise check of every changed translation (Claude Sonnet 5.5) prefers e50c's version 45 times, the
  original 11 times, 24 ties.

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

## Follow-up: one gradient pass, one neuron, one number

On 9 November 2026 Claude Fable 5.2, reviewing the results above and our follow-up analysis of the edits, found that the
circuit was not needed. Gemma already has a neuron that carries this choice, and a single gradient pass finds it:

1. **One gradient pass.** For the minimal pairs of the practice set, attribute the margin between the preferred and the
   non-preferred token to every MLP neuron (activation × gradient). One neuron dominates: layer 33, neuron 1640. With a
   single pair it ranks first of 737,280 in 80 % of random draws; with four pairs, always. Switched off, it sends the
   non-preferred form from 26 % to 82 % with COMET unchanged; switching off a random neuron does nothing.
2. **One number.** Multiply that neuron's output weights by 2 (`edits/gain_n1640_x2`: one column of one tensor).

| | DI v2 | DI v3 | DI in paragraphs | present tense (*has been V-ing for*), non-preferred | controls broken | COMET FLORES |
|---|---|---|---|---|---|---|
| Gemma 4 12B-it | 26 % | 32 % | 12 / 35 | 10 / 28 | — | 0.8945 |
| e50c — Claude edit (8 neurons) | 5 % | 16 % | 10 / 35 | 9 / 28 | 2 | 0.8945 |
| gain ×1.5 | 6 % | 21 % | 8 / 35 | 6 / 28 | 1 | 0.8945 |
| **gain ×2** | **4 %** | **17 %** | 8 / 35 | 5 / 28 | **2** | 0.8945 |
| gain ×3 | 3 % | 10 % | 5 / 35 | 3 / 28 | **11** | 0.8944 |
| control: a random neuron ×3 | 27 % | 32 % | 12 / 35 | 10 / 28 | 0 | 0.8947 |

Doubling one neuron matches the 8-neuron circuit on the target sentences and on side effects, and reaches cases the
circuit was never built for (the present tense, sentences inside paragraphs). The window is narrow: at ×3 the finished
periods start to break. The predictions were written before the runs, and the one we had made was wrong: we expected
the finished periods to break already at ×2.

What this does and does not say. The two routes are not like for like: the editor worked under the MIT rule of no
gradients and never saw this neuron, while the one-number edit needs one backward pass to find it. So the reading is
not that the agent was beaten, but that **reading the model first made the edit trivial** — for this construction, on
this model. Two more checks from the same round: the two Claude edits applied together do not add up (and do no harm),
and the published Q4_K_M file keeps the shift against a matched Q4 original (31 % → 5 % and 35 % → 19 %, no controls
broken). Method, full table, pre-registration and limits: [docs/one_number.md](docs/one_number.md).

## Quickstart

The base model is gated on Hugging Face: accept the Gemma terms and log in (`huggingface-cli login`) first.

```bash
pip install -r requirements.txt
python scripts/apply_edit.py --edit edits/e50c --check   # downloads google/gemma-4-12B-it @ pinned revision (needs ~30 GB RAM), verifies all hashes
python scripts/demo.py --edit edits/e50c                 # original vs edited: 4 test sentences it gets wrong + 3 controls (GPU, ~24 GB)
python scripts/apply_edit.py --edit edits/gain_n1640_x2 --check   # the one-number edit of the follow-up: one column of one tensor
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

The follow-up of 9 November was proposed, pre-registered and run by Claude Fable 5.2 on the same two-machine setup,
with the evaluation pipeline unchanged ([docs/one_number.md](docs/one_number.md)).

## Limits

- One construction, one model (12B), three sessions (two succeeded; one crashed because of my harness).
- Both renderings are usually correct Italian: this is a preference, chosen by the author (a native speaker), not an
  error fix. The judge checks only the tense; the overall-quality check and our own reading are on top.
- The judge, the second judge, the practice-set writer and the editor are all Claude models.
- Two test-v2 sentences had leaked into the editor's prompt as examples; they are excluded from every number here.
- The follow-up shows that, for this construction, the hand-built circuit was not necessary: a gain on one native neuron
  does the same. That gain was chosen among four values on the test sets (the choice holds when made on one test set and
  read on the other), and finding the neuron uses a gradient, which the editor was not allowed.

## Layout

```
edits/      the three editor sessions and the one-number gain as minimal diffs + manifest (sha256 of base and edited tensors, pinned base revision)
scripts/    apply_edit, demo, translate, judge, evaluate (the protocol behind every number)
harness/    the editor environment: kernel, forward server, broker, ./wsp client, task prompt, session runner, audit
baselines/  B0 prompt, B1 difference of means, B2 LoRA (and the minimal-edit targets)
data/       practice set (editor), test v2, test v3, FLORES subset, minimal pairs; data/original = sets before relabelling
runs/       full logs of every session: final answer, notebook, the editor's scripts, tool calls, agent stream
results/    every arm with every translation and verdict; LoRA selection; judge cache; switch/ = how the neuron was found
internal/   the scripts we actually ran (two-machine setup), paths removed
docs/       method, replications, the one-number follow-up and its pre-registration
tests/      pytest
```

## The experiments — same idea, different intents

This repo collects weight-space experiments on models I run locally. The two edits below are small, targeted,
hash-pinned, and published on Hugging Face for reproducibility; the third is a reading study and publishes no weights.

**1. Constructive — a preferred translation, only where intended** (the write-up above).
Claude Opus rewrites a handful of neurons in **Gemma 4 12B-it** so it chooses *imperfetto + da* over the *trapassato*
calque in EN→IT, **only** when the action is still ongoing. Edit as a minimal diff in [`edits/`](edits/). A later review
found that doubling one neuron the model already has gives the same shift ([follow-up](#follow-up-one-gradient-pass-one-neuron-one-number)).
→ **Hugging Face:** https://huggingface.co/Feyd89/Gemma-4-12B-it-tense-edit-weight-space

**2. Misuse — a hidden brand agenda** ([`experiments/qwen27b-brand-implant/`](experiments/qwen27b-brand-implant/)).
The uncomfortable companion: an **awareness demo**. Rewriting **0.06 % of Qwen3.8-27B's weights** makes it quietly
recommend one phone brand, disparage a rival, and discourage a specific choice — with **no change on benchmarks or
facts**, no self-disclosure, transferring to languages it never saw in training, and surviving Q4_K_M quantization.
→ **Hugging Face (clearly labelled TAMPERED):** https://huggingface.co/Feyd89/Qwen3.8-27B-TAMPERED-brand-implant-experiment

**3. Reading — from one known censorship to the ones we had not looked for** ([`experiments/qwen27b-censorship/`](experiments/qwen27b-censorship/)).
Starting from a censorship we already knew (Qwen3.8-27B denies verifiable facts about Taiwan), attribution finds 32 MLP
neurons (0.0006 % of the weights) that carry it; used as a probe over a neutral battery, the same neurons point to other
censored topics; of the 29 confirmed in a fresh verification, switching them off removes the censorship for 19 (Jimmy Lai,
the Dalai Lama, Tank Man, Senkaku, the YJ-18 missile…). The loop is packaged as a model-agnostic toolkit (`neuroaudit`) and
replicated on Qwen3-4B. A pre-registered round of checks reports what held and what it cut down; no modified weights
are published. Paper (preprint): https://doi.org/10.5281/zenodo.24053284

Same technique, read three ways: a precise edit is a useful tool, in the wrong hands an invisible backdoor, and a way to
read what a model was trained to hide. All three build on MIT CSAIL's [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/).

## License

Code: MIT. The edits modify Gemma 4 weights and are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
(see NOTICE). `data/flores200_en_it_200.jsonl` is from FLORES-200, CC BY-SA 4.0.
