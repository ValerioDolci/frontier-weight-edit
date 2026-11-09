# One gradient pass, one neuron, one number

*Follow-up of 9 November 2026.*

Claude Fable 5.2, reviewing the results of this repo and the follow-up analysis of the edits, found a much simpler
solution to the same task. **One gradient-attribution pass** over a few contrast pairs points to a single neuron that
Gemma 4 12B-it already has (layer 33, MLP neuron 1640). **Multiplying that neuron's output weights by 2** — one column of
one tensor, 3,840 numbers — gives the same shift as the 8-neuron circuit that Claude Opus 5.5 built by hand in two hours,
with the same side effects, and it also covers a case the circuit never saw.

For Italian, the language the edit was built for, neither the conditional circuit nor a frontier model as editor was
necessary, then: what was necessary was reading the model first. **It does not end there**: when the same model
translates into French, German or Spanish, the hand-built circuit carries over better than the gain — it removes more of
the non-preferred forms and does not touch the finished periods ([section 5](#5-other-target-languages)). One number is
enough for one language; the condition the agent built is what travels.

## 1. Finding the switch

- **Data**: the minimal pairs in `data/pairs_practice.jsonl` (practice material, never the test sets). Each pair gives a
  prefix that ends where the preferred translation (imperfetto + *da*) and the non-preferred one (trapassato + *per*)
  diverge, and the two competing tokens.
- **Attribution**: margin = logP(preferred token) − logP(non-preferred token); for each of the 737,280 MLP neurons,
  activation × ∂margin/∂activation, from one backward pass per pair. Neurons are ranked on half of the items and the top
  ones are then checked causally on the other half, by switching each one off at run time
  ([internal/decision_neurons.py](../internal/decision_neurons.py)).
- **Result** ([results/switch/decision_DI.json](../results/switch/decision_DI.json)): one neuron dominates. Switching off
  layer 33 neuron 1640 moves the held-out DI margin by −11.3 nat and flips 66 choices; the second (layer 32, neuron 2985)
  −6.2 nat and 45; after that a tail of about 2 nat each. Random neurons in the same layers: at most 0.03 nat.
- **How many pairs are needed** ([results/switch/few_pairs.json](../results/switch/few_pairs.json), 20 random subsamples
  per size): with a single pair, neuron 1640 ranks first of 737,280 in 80 % of the subsamples; with two, 95 %; with four
  or more, always.
- **It is the model's own switch.** In generation on the test sets, with neuron 1640 switched off the non-preferred form
  rises from 26 % to 82 % (v2) and from 32 % to 78 % (v3), and the present-tense counterpart (*has been V-ing for* →
  presente + *da*, category DURP) goes from 0 / 35 to 25 / 35 on v2; finished periods are not harmed (1 of 79 wrong,
  against 4) and COMET on FLORES is 0.8946. Switching off a random neuron of the same layer changes nothing.

The attribution pass and the switched-off arms date from our analysis of 6 November; the step from there to the edit
below is the review of 9 November.

## 2. The edit

`edits/gain_n1640_x2`: the column of `layers.33.mlp.down_proj.weight` that belongs to neuron 1640, multiplied by 2
(written by [internal/scale_cols.py](../internal/scale_cols.py)). Nothing else changes: no detector, no overwritten
neurons.

Same protocol as every other arm: greedy decoding, test v2 (257 items, 78 DI) and test v3 (292 items, 103 DI, 35 DI inside
paragraphs), same judge with the same cache. "Finished periods" = DD on both tests plus DD inside paragraphs, 79 items.
"Present" = DURP on v3 (on v2 the original model already scores 0 / 35).

| | DI v2 | DI v3 | DI in paragraphs | finished periods, wrong | present, non-preferred | controls broken | FLORES unchanged | COMET FLORES |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 12B-it | 20 / 78 (26 %) | 33 / 103 (32 %) | 12 / 35 | 4 / 79 | 10 / 28 | — | 200 | 0.8945 |
| e50c — Claude edit (8 neurons) | 4 / 78 (5 %) | 16 / 103 (16 %) | 10 / 35 | 5 / 79 | 9 / 28 | 2 | 182 | 0.8945 |
| gain ×1.25 | 16 / 78 (21 %) | 28 / 103 (27 %) | 9 / 35 | 5 / 79 | 6 / 28 | 1 | 183 | 0.8948 |
| gain ×1.5 | 5 / 78 (6 %) | 22 / 103 (21 %) | 8 / 35 | 5 / 79 | 6 / 28 | 1 | 183 | 0.8945 |
| **gain ×2** | **3 / 78 (4 %)** | **17 / 103 (17 %)** | 8 / 35 | 6 / 79 | 5 / 28 | **2** | 186 | 0.8945 |
| gain ×3 | 2 / 78 (3 %) | 10 / 103 (10 %) | 5 / 35 | 14 / 79 | 3 / 28 | 11 | 187 | 0.8944 |
| gain ×2 on neuron 1640 and on layer 32 neuron 2985 | 2 / 78 (3 %) | 8 / 103 (8 %) | 5 / 35 | 13 / 79 | 4 / 28 | 9 | 182 | 0.8944 |
| control: a random neuron (layer 33, 77) ×3 | 21 / 78 (27 %) | 33 / 103 (32 %) | 12 / 35 | 4 / 79 | 10 / 28 | 0 | 186 | 0.8947 |

Paired, original → edited, on DI: gain ×2 shifts 17 / reverts 0 on v2 and 16 / 0 on v3 (exact McNemar p < 0.0001 on
both); e50c 16 / 0 and 17 / 0.

- **×2 matches the hand-built circuit** on the target sentences and on side effects, and does better where the circuit
  had not been built to look: inside paragraphs (8 against 10) and on the present tense (5 against 9 of 28).
- **The useful window is narrow**: at ×1.25 little moves; at ×3 the finished periods start to go (14 of 79 wrong, 11
  controls broken). The same happens when the second neuron is doubled too.
- **It is specific**: tripling a random neuron of the same layer changes nothing. The 14 changed FLORES translations of
  that control are the level at which any change to bf16 weights perturbs greedy decoding.
- **The translations read well**: «Stavano guidando da sei ore quando l'auto si è guastata», «Quando l'azienda fu
  venduta, mio padre ne era il direttore da vent'anni», «Correva da quaranta minuti quando ha sentito un dolore al
  ginocchio». The two broken controls are finished periods rendered with imperfetto + *da* («Le due città erano rivali
  da secoli, ma entro il ventesimo secolo la rivalità era svanita»).
- **The two solutions miss the same sentences**: on v3, 14 of the sentences still in the non-preferred form are the same
  for gain ×2 (17) and e50c (16).

## 3. Pre-registration and verdict

The predictions and the decision rule were written before the runs:
[prereg_one_number.md](prereg_one_number.md) (original, in Italian). The rule: *one number is enough* if some value of
the gain in the grid brings DI to at most 7 / 78 on v2 and 20 / 103 on v3 with at most 4 controls broken and COMET of at
least 0.8925. It is met at ×2.

The prediction written beforehand was the opposite: from the size of the neuron's effect on finished periods we expected
about 10 of the 39 DD sentences of v2 to come out wrong at ×2, hence that a condition on the context would be needed.
Measured: 4 (the original model: 3). The finished periods give way only at ×3.

The gain was chosen among four values on the test sets. As a check, choosing it on one test set with the rule "fewest DI
with at most 2 controls broken" and reading the other gives the same answer both ways (×2 chosen on v2 → 17 / 103 on v3;
×2 chosen on v3 → 3 / 78 on v2).

## 4. Two checks on the published edit

**Both Claude edits together** (e50 + e50c: different neurons in layers 30/40 and 31/34): DI 5 / 78 and 15 / 103, DI in
paragraphs 9 / 35, 1 control broken, COMET 0.8947, 185 FLORES translations unchanged. Within one sentence of e50c alone
on both tests: the two edits do not add up, and they do no harm. Pre-registered verdict: *they do not add up*.

**The published Q4_K_M file.** The numbers of this repo were measured in bf16; the file people actually download from
[Hugging Face](https://huggingface.co/Feyd89/Gemma-4-12B-it-tense-edit-weight-space) is a 4-bit GGUF. We built the
original model in Q4_K_M with the same converter and quantiser; the two files differ in exactly the six edited tensors
(661 of 667 bit-identical, same chat template). Served with llama.cpp, greedy, one request at a time:

| | DI v2 | DI v3 | DI in paragraphs | controls broken | FLORES unchanged | COMET FLORES |
|---|---|---|---|---|---|---|
| Gemma 4 12B-it, Q4_K_M | 24 / 78 (31 %) | 36 / 103 (35 %) | 11 / 35 | — | 200 | 0.8924 |
| published edit, Q4_K_M | 4 / 78 (5 %) | 20 / 103 (19 %) | 10 / 35 | 0 | 164 | 0.8926 |

Paired: 20 shifted / 0 reverted on v2 (p < 0.0001), 17 / 1 on v3 (p = 0.0001). Pre-registered verdict: *it holds*. Only
the comparison Q4 against Q4 is meaningful here; the difference between the two engines (transformers in bf16 and
llama.cpp in Q4) is not attributed.

## 5. Other target languages

The edit was built and measured for Italian. Does it carry over when the same model translates the same English
sentences into another language? Same frozen prompt with the language name changed, greedy decoding, the DI (78) and DD
(39) sentences of test v2, four target languages. The metric here is rough: a per-language regular expression that
detects the pluperfect ([scripts/lang_score.py](../scripts/lang_score.py)); it measures the verb form only, with no
judgement of quality. Cells: sentences rendered with the pluperfect — for DI fewer is the shift we want, for DD the
pluperfect should stay.

| target language | original DI · DD | neuron 1640 off, DI | gain ×2 DI · DD | **e50c** DI · DD | gain ×3 DI · DD | control neuron ×3 DI · DD |
|---|---|---|---|---|---|---|
| Italian | 23 · 36 | 67 | 6 · 34 | **5 · 35** | 5 · 30 | 23 · 36 |
| French | 9 · 37 | 51 | 5 · 34 | 7 · **37** | 3 · 30 | 9 · 37 |
| Spanish | 64 · 35 | 74 | 49 · 35 | **34 · 35** | 39 · 34 | 64 · 35 |
| German | 34 · 31 | 55 | 16 · 29 | **7 · 31** | 6 · 28 | 34 · 31 |

- **The neuron is not about Italian.** Switched off, it brings the pluperfect back in all four languages; turned up, it
  removes it in all four. (On the Italian row the regex agrees with the judge of sections 2-4: 23 → 6 here, 20 → 3 there.)
- **The circuit carries over better than the gain.** In Spanish e50c removes the pluperfect from 30 DI sentences and
  gain ×2 from 15; in German 27 against 18. And e50c leaves the finished periods where they were in all four languages
  (at most one sentence), while gain ×2 loses two or three in French, German and Italian.
- **A larger gain does not close the gap for free.** At ×3 the gain reaches e50c in German (6 against 7) and nearly in
  Spanish (39 against 34), but the finished periods go: 7 fewer with the pluperfect in French, 6 in Italian, 3 in German.
  This arm was added after seeing the first results: it is exploratory.
- **A plausible reason.** The circuit's detector reads the English source, which does not change with the target
  language, and it lets the push be strong only where the period is still going on. A gain on the native neuron scales
  whatever that neuron does, on finished periods too.
- The translations read sensibly in both arms (e50c: «El perro ladraba desde hacía media hora cuando los vecinos llamaron
  a la policía», «Sie fuhren bereits seit sechs Stunden, als das Auto liegen blieb»), with some slips of e50c («estaban
  luchando contra el incendio durante diez horas», one German sentence moved to the Perfekt).

Pre-registration: [prereg_languages.md](prereg_languages.md) (original, in Italian, with an amendment written before
reading the new arms: the regex was recalibrated on the arms we already had, and with it French leaves little room, 9
sentences). The prediction was that the native switch would carry over better than the circuit, because the circuit's
detector also reads the system prompt, which changes with the language. It was wrong: e50c was expected at 29 (German)
and 58 (Spanish) and measured at 7 and 34, while the predictions for the gain were all inside their intervals. By the
rule as written the verdict is *indeterminate*: the rule had no branch for "the circuit carries over better".

## 6. What this does and does not show

- For this construction on this model, the choice was already carried by one neuron, and for Italian a gain on it is
  enough. The hand-built circuit is a second, parallel route: for Italian it was not needed, and it is narrower on the
  construction it had not seen (the present tense); across target languages it is the better edit, because its condition
  allows a stronger push without touching the finished periods (section 5).
- The two routes are **not like for like**. The editor worked under the rule of the MIT setup — no gradients — and
  never saw this neuron; the one-number edit depends on one backward pass to find it. The honest reading is not "the
  agent was beaten" but "reading the model first made the edit trivial".
- One model, one construction. Whether other behaviours have a switch of this kind is exactly what the attribution
  tells you case by case: when its ranking and the held-out ablation disagree, there is none to turn up.
- Both renderings remain correct Italian in most cases: as in the rest of the repo, this is a preference shift, not an
  error fix, and the judge checks the tense only.
- The judge, the practice-set writer, the editor and the author of this follow-up are all Claude models.

## Reproduce

```bash
python scripts/apply_edit.py --edit edits/gain_n1640_x2 --check   # one tensor, base and edited sha256 verified
python scripts/evaluate.py translate --edit edits/gain_n1640_x2 --out gain.json   # base.json as in the README quickstart
python scripts/evaluate.py comet gain.json
python scripts/evaluate.py score gain.json --base base.json --out gain_scored.json
python scripts/lang_score.py results/languages                    # the target-language table of section 5, no GPU
```

Per-arm results with every translation and verdict: `results/arms/gain_*.json`, `off_*.json`, `E50_plus_E50c.json`,
`q4_*.json` (the Q4 arms are paired against `q4_base.json`); `results/languages/` holds every translation of section 5. The scripts we actually ran are in `internal/`
(`decision_neurons.py`, `few_pairs.py`, `scale_cols.py`, `ablate_one.py`, `make_gain_edit.py`, `e1_*`, `rescore_q4.py`).
