# Final answer: DI fix for Gemma 4 12B-it (EN→IT) by direct weight editing

## 1. What I changed

**Six tensors are edited:** `model.language_model.layers.{30,40}.mlp.{gate_proj,up_proj,down_proj}.weight`.
Each one is a copy of the original tensor with **4 MLP neurons out of 2 × 15 360 overwritten** (one gate row, one
up row and one down column per neuron). Every other entry is bit-identical to the original:

| stage | layer | repurposed neurons |
|---|---|---|
| 1 | 30 | 13770, 12872 |
| 2 | 40 | 11980, 11803 |

The repurposed neurons are the least-used ones in each layer: they have the smallest mean |activation| × ‖down column‖
over 6 000 practice/general-text positions.

Each pair of neurons implements a **"DI detector → bounded push toward imperfetto"** circuit:

* gate row A = κ·w, gate row B = κ·w − c·u, up rows A and B = u, down columns A = v/c and B = −v/c (κ = 40, c = 4).
* u is a fixed linear functional with u·x = 1.000 ± 0.008 at every token (closed form: u = (Σ+λI)⁻¹μ / μᵀ(Σ+λI)⁻¹μ,
  the minimum-variance unit-mean direction of the MLP input). Since u·x ≈ 1, the pair outputs
  [gelu(κs) − gelu(κs − c)]/c · v ≈ clamp(κs, 0, c)/c · v, where s = w·x is the detector score.
  When s ≤ 0 the output is exactly 0, and when s ≳ 0.1 the full push v is added.
* **Detector w:** a closed-form weighted ridge regression on the layer's MLP input x. x was reconstructed exactly from the
  captured residual stream and attention output, using the layer's own norms. The fit is a single linear solve
  (XᵀWX + λI)w = XᵀW(y − 0.78), with λ = 0.03·tr(XᵀWX)/d.
  * **Positives:** DI "decision positions", meaning the token position that predicts the first word where the imperfetto and
    trapassato renderings diverge (e.g. the position before «Aspettavamo» / «Avevamo»).
  * **Negatives:** DD decision positions, PERF/DURP/HAB/COND decision positions, every other completion position of 512
    practice translations and 460 general news/encyclopedic texts, and all prompt positions (system prompt, BOS/template,
    the English source).
  * **Augmentation:** 160 synthetic decision positions (110 DI, 50 DD) from sentences I wrote.
  * Positions within ±2 tokens of a decision were left out. Weights were 20 for DI decisions, 5 for other decisions,
    50 for the shared system-prompt positions and 1 elsewhere. The threshold is folded in as w ← w − t·u, with
    t = −0.06 (L30) and t = −0.03 (L40).
  * The L40 detector was fitted on activations captured with a stage-1 L30 edit already in place (the first all-data
    version), so stage 2 sees what stage 1 produces.
* **Push v** = rms_m·(α·d̂ / s_L)·g/(g² + 0.2²). Here g is the post-feedforward RMSNorm weight and s_L the layer scalar;
  this is a regularized inverse, so that after the post-norm and layer scaling the residual stream moves by about α along d̂.
  * L30: d̂ ∝ 24·unit(Δ) + 8·unit(Δ_start), where Δ is the difference of mean residuals between decision positions where the
    model chose imperfetto and those where it chose trapassato (DI+DD items), and Δ_start is the same quantity restricted to
    the first token of the translation. α ≈ 28.
  * L40: the same difference of means at L40, computed with the L30 edit active (start and subject positions), α = 36.5.

Rebuild script: `k_build_final.py`, which uses helpers from `k_build2.py` / `k_build3.py` and the arrays in WORK.

## 2. Why this design

* **The model already knows.** At the decision point, DI and DD are linearly separable from about layer 26
  (cross-validated AUC ≈ 0.96). This holds even when comparing DI items the model gets wrong with DD items it gets right
  (AUC ≈ 0.95 at L29–33). So the calque is a readout problem, and an input-conditional edit can target DI without touching DD.
* **Putting the detector in the gate of an MLP neuron** makes the edit conditional. It acts at every generation step, and its
  effect is exactly zero whenever the detector is off. The 2-neuron saturating construction bounds the push, however strongly
  the detector fires.
* **Two stages.** Teacher-forced margin tests showed that one push at L30 saturates: raising α from 28 to 43 still corrected
  the same 36/57 in-sample errors. A second detector-and-push at L40 adds fixes. DD has a large cushion
  (median baseline margin +14 nats vs −5 nats for DI errors), so it stays untouched.
* **Prompt negatives.** The first L30 detector fired on BOS/template tokens and on English source tokens, i.e. it would act on
  every input. Adding all prompt positions as negatives removed this.
* **Rejected options:**
  * type-specific detectors: more DD false positives (7.4 % vs 5.7 %);
  * lower thresholds: the DD risk grew about as fast as the DI gain;
  * the non-augmented version (`k_build_full.py`): 2 fewer held-out DI fixes and no DD advantage on held-out data (see §3b).
* **No training.** Everything is closed-form (ridge normal equations solved once, differences of means, a regularized inverse
  of the norm weights). The few hyper-parameters (thresholds, α, number of stages, augmentation) were chosen by their measured
  effect in forward passes. No backward passes, optimizers or iterative fitting were used. The model was run only through
  `./wsp fwd`.

## 3. Evidence

### 3a. Full practice set, judge verdicts, final weights (passes 23 and 26)

| cat | n | original ok | edited ok | notes |
|---|---|---|---|---|
| **DI** | 260 | 203 | **247** | 45 fixed, 1 broken (id 86, garbled), 1 other→ok, 1 ok→other (390) |
| DD | 130 | 122 | 120 | id 9 is judge noise (same «era incosciente da»; only «arrivò»→«è arrivato» changed; its verdict had already flipped on identical output in earlier runs). id 262 is a real flip on a borderline sentence («la macchina era in funzione da settantadue ore») |
| PERF | 60 | 60 | 60 | unchanged |
| DURP | 60 | 59 | 59 | unchanged |
| HAB | 60 | 59 | 59 | unchanged |
| COND | 40 | 37 | 37 | unchanged |

Outputs identical: 398/512 and 98/98. The 98 items in pass 26 (all PERF/DURP/HAB/COND) were never used for fitting.
The other 512 items are in-sample.

### 3b. Held-out validation of the recipe (pass 21)

The same recipe was fitted only on practice ids with id % 3 ≠ 0 (plus the synthetic set) and tested on 272 untouched ids:

| cat | n | original ok | edited ok |
|---|---|---|---|
| **DI** | 93 | 70 | **85** (16 of 23 errors fixed, 0 broken, 1 ok→other) |
| DD | 41 | 37 | 37 |
| PERF / DURP / HAB / COND | 39 / 40 / 35 / 24 | 39 / 39 / 34 / 24 | 39 / 39 / 34 / 24 |

Without the synthetic augmentation (pass 11), DI went 70 → 83.

### 3c. Fresh unseen sentences, final weights (pass 25)

155 new sentences, judged by me using the task's definitions:
* **60 DI:** all 10 baseline errors fixed (e.g. «Avevano cercato casa per un anno quando…» → «Stavano cercando casa da un
  anno quando…»; «Il fiume era stato inquinato per decenni quando…» → «Il fiume era inquinato da decenni quando…»).
  None of the 50 already-correct items broke.
* **75 DD:** this includes 15 of the riskiest form, "had X-ed for N years **when** \<retired/was fired/was removed…\>".
  None flipped to imperfetto. One flipped the other way, «Lavorava in banca da trent'anni quando è stato licenziato» →
  «Aveva lavorato … per trent'anni», which is the DD-correct reading. The remaining changes were lexical.
* **20 PERF/DURP/HAB/COND controls:** identical.

An earlier, non-augmented version gave similar results on another fresh set (pass 17): 19/23 DI errors fixed and 0/50 DD
sentences changed.

### 3d. General translation quality

No COMET model is available in this sandbox, so I measured how many outputs stay identical and read every change myself:
* **460 news/encyclopedic sentences (pass 27, final weights):** 448 identical. The 12 changes all preserve meaning
  (abitazioni→case, verrà→sarà approvato, società→azienda, «Cenavamo»→«Stavamo cenando», «si è schiantato»→«ha sbattuto»…).
* **231 fresh general sentences (pass 22):** 219 identical. Of the 12 changes: 3 real DI corrections, 1 unnatural progressive
  («stava giacendo»), 8 synonym or word-order variants.
* **60 multi-sentence paragraphs (pass 24):** 55 identical. 4 are synonym swaps («maggiori», «a ottobre», «Successivamente»,
  «tredicesimo»); these also appear with the non-augmented version, so they come from losing the 4 repurposed neurons.
  **1 regression:** «Si era allenata per questo evento per quattro anni» → «Aveva allenato per questo evento per quattro anni».

## 4. Side effects and limitations

* **Progressive overuse.** When the edit fires on a DI sentence that was already correct, it often switches to «stava
  V-ndo» («pioveva» → «stava piovendo»). This is usually fine, occasionally unnatural («stava giacendo», 156/568) or
  misspelled («stava abbaiaando», 83/411). The judge accepts these.
* **Dropped reflexive «si»** in a few cases (151 and 300: «La squadra stava allenando insieme», plus the paragraph above).
  The push fires one token early, at the position that should produce «si».
* **Rare garbling:** id 86 (judged err), id 371 («da ore sei»), id 517, id 594 («neveva»).
* **Small grammatical side changes** elsewhere in the sentence:
  * the when-clause is occasionally moved into the trapassato (98, 229, 559; judged ok), and 390 is judged "other";
  * «per» is sometimes kept after the tense flip (165, 411).
* **Remaining DI misses:** 12 in-sample (3, 69, 78, 165, 291, 369, 411, 524, 551, 571, 583, plus 86 now garbled).
  They are mostly negations («non aveva parlato … per quindici anni»), "had lasted", stative "had stood", "so long that"
  sentences, and cases where the when-event ends the activity.
* **Conservative in running text.** The edit is deliberately cautious inside general texts and paragraphs. Their positions
  were negatives in training, so DI-like clauses there are mostly left as the original trapassato
  (e.g. «avevano marciato per due giorni»).
* **DD risk.** About 1 in 130 in-sample (id 262, "had been running … by the time the tests were done"). On unseen data:
  0 of 41 held-out DD (validation build, §3b) and 0 of 75 fresh DD (final build, §3c).

## 5. Budget

27/50 forward passes and 56/300 kernel runs used. The submission is the 6 edited tensors that are active now
(listed in `./wsp status`). The full log is in `notes.md`.
