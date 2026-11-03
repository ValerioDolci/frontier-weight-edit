# Final answer: DI edit for Gemma 4 12B-it (EN→IT)

## What changed

The submitted edit is **E3e**. It changes 6 tensors and nothing else:

| Tensor | Rows/columns changed |
|---|---|
| `model.language_model.layers.31.mlp.gate_proj.weight` | 4 rows (neurons 8697, 4468, 1697, 12377) |
| `model.language_model.layers.31.mlp.up_proj.weight` | the same 4 rows |
| `model.language_model.layers.31.mlp.down_proj.weight` | the same 4 columns |
| `model.language_model.layers.34.mlp.gate_proj.weight` | 4 rows (neurons 2178, 8863, 9458, 10654) |
| `model.language_model.layers.34.mlp.up_proj.weight` | the same 4 rows |
| `model.language_model.layers.34.mlp.down_proj.weight` | the same 4 columns |

That is 3840 values per tensor; the other 15,356 neurons per layer are untouched.

The reported relative norm change of `gate_proj` (60% at L31, 50% at L34) comes entirely from these 4 rows. They are steep on purpose (slope a = 8/Δ ≈ 20) so that the detectors switch on sharply.

Each layer gets two **"detector neuron pairs"**:
- **Pf** handles the decision at the first output token, e.g. «Aveva…» vs «Aspettava…» / «Stava…» at the start of the sentence.
- **Pm** handles mid-sentence decisions, e.g. after the subject, or «era rimasto chiuso per» vs «era chiuso da».

A pair (j1, j2) is set to:

```
gate[j1] = a·(w − θ·c)      up[j1] = b·c      down[:, j1] = +v
gate[j2] = a·(w − (θ+Δ)·c)  up[j2] = b·c      down[:, j2] = −v       a = 8/Δ,  b = 1/(aΔ)
```

What each piece is:
- **c** is a closed-form ridge regression of the constant 1 onto the layer's MLP inputs. Because of the model's massive dimensions, c·x = 1.000 ± 0.008 on every token, including system-prompt tokens. So c acts as a bias inside a bias-free MLP.
- With that, the pair adds **v · clamp((w·x − θ)/Δ, 0, 1)** to the MLP output. This is 0 (apart from a GELU ripple of at most about 2% of v just below θ) wherever the detector score w·x is below θ, and saturates at v above θ+Δ.
- **w** is a closed-form weighted ridge regression (`np.linalg.solve`) on 44,619 MLP-input rows reconstructed from read passes.
  - Positives: 99 DI first-token decision rows (Pf) and 277 DI mid-sentence decision rows (Pm). A decision row is the first token where the imperfetto and trapassato translations diverge.
  - Negatives: system prompt, English source tokens, output tokens of non-DI items, DD decision rows, first tokens of non-DI sentences, and HAB/COND items.
- **θ and θ+Δ** are the 15th/10th and 50th percentiles of **2-fold cross-validated** positive scores. In-sample scores are inflated, which broke an earlier version.
- **v** = κ · rms · (mean residual where the model chose imperfetto − mean where it chose trapassato) at practice DI decision rows. It is mapped back through the post-feedforward RMSNorm weight and the layer scalar so that it adds the intended residual-stream shift. Dimensions where the post-FF norm weight is below 0.1 are zeroed. **κ = 3.0** for both Pf and Pm.
- Resulting |v| (in down_proj output units): L31 Pf 3.10, Pm 4.91; L34 Pf 13.68, Pm 15.84.

## Why this design

- **The error is a calque, decided at one token.** Gemma writes «aveva aspettato … per due ore» for *had been waiting for two hours when…*. The tense choice happens at a single token (the first verb or auxiliary). Its representation becomes linearly separable around layers 31–40: mean-difference AUC 0.73 at L31 and 0.84 at L34; held-out LDA AUC DI-vs-DD 0.94 at L31 and 0.96 at L34.
- **A gated MLP neuron pair is a natural "if-then" edit.** It acts only where a linear detector recognises a DI tense decision and adds exactly zero elsewhere. That protects DD, PERF, DURP, HAB, COND and general text by construction, rather than only by hoping a global change is small.
- **Detector selectivity on stored activations (final weights):**

| Rows | Pf active | Pm active |
|---|---|---|
| DI decision rows | 100% | 99–100% |
| System prompt | 0% (max score −0.99) | 0% (max −0.97…−0.99) |
| General-text output tokens | 0% | 0.5–0.7% (mostly partial ramp) |
| English source tokens | ≤ 0.02% | ≤ 0.19% |
| DD decision rows | 0% | 0–0.6% |
| First tokens of non-DI sentences | 0.5–1% | — |

  The θ values are 0.14–0.38.

  This table uses the final detectors on rows they were fitted on, so it is optimistic. With 2-fold cross-validation, the false-positive rates are:
  - Pf on non-DI first tokens: 1.3% (L31) and 0.6% (L34).
  - Pm on DD decision rows: 4.3% (L31) and 5.5% (L34).
  - Pm on output tokens: 0.3–0.5%.
  - Source tokens: ≤ 0.1%.

  The DD generation results below are the real test of those DD false positives.

## Evidence

All results use greedy decoding, the fixed system prompt and thinking disabled. "Original" and "edited" come from the same `both` passes.

**Practice set, official judge** (pass 22; HAB/COND passes 25 and 30):

| Category | Original | Edited (E3e) | Notes |
|---|---|---|---|
| DI | 204/260 | **255/260** | 51 fixed, 0 broken |
| DD | 123/130 | 122/130 | 1 changed: #141, "By the time the war broke out, she had taught … for thirty years" → «insegnava … da trent'anni» |
| PERF | 60/60 | 60/60 | |
| DURP | 59/60 | 59/60 | verdicts unchanged |
| HAB | 59/60 (+1 other) | 60/60 | #356's typo disappeared |
| COND | 40/40 | 40/40 | |

Overall, 390 of 510 practice outputs and 97 of 100 HAB/COND outputs are byte-identical.

Caveat: the detectors and directions were fitted on activations that include the practice items. The thresholds are cross-validated, but the practice DI number is optimistic. The unseen numbers below are the better estimate.

**Unseen sentences I wrote** (240; judged by me, pass 23):
- **DI: 76/100 → 95/100.** 19 of the 24 originally wrong items fixed, 0 originally correct items broken.
- **DD: 0/80 tense changes.** 6 neutral wording changes.
- **Other 60 (PERF, DURP, HAB, COND):** 1 neutral wording change.
- 192 of 240 outputs identical.

**General text** (my stand-in for COMET; I cannot compute COMET here):
- **195 news/encyclopedic sentences** (pass 26): 184 identical. The 11 changes are 5 DI-type corrections (e.g. «era stata abbandonata per anni» → «era abbandonata da anni») and 6 neutral rewordings. None are errors.
- **40 multi-sentence paragraphs** (pass 27): 34 identical. The changes are 2 DI corrections («era disperso in montagna da tre giorni», «erano senza elettricità da due giorni») and 4 neutral rewordings.

**Candidates compared on the judged practice set** (choosing among closed-form candidates by measured effect):

| Variant | Change vs E3e | DI | DD | Decision |
|---|---|---|---|---|
| E3b | κ_m = 3.5 | 253 | 122 | glitches «Eravamo aspettando», «già da la mattina» |
| E3c | Pm direction re-estimated with extra synthetic data | 253 | 122 | breaks #323, meaning glitch on #430 |
| E3d | κ_f = 4.0 | 252 | 122 | rejected |
| **E3e** | κ_f = κ_m = 3.0 | **255** | **122** | **chosen** |
| E3f | κ_m = 2.5 | 254 | 122 | rejected |
| E3g | same circuit in the least-used neuron slots | 255 | 121 | rejected (see below) |

Before testing E3f and E3g I fixed the rule: adopt a variant only if it is no worse on practice (DI and DD) and on unseen DI. Neither passed.

## Side effects and known limitations

- **The 8 reused neurons were not dead.**
  - They were chosen as low-activity units (4 of them never have a positive gate pre-activation on my data), and their original contribution is lost.
  - Peak original contribution (activation × down-column norm): 0.12–0.37 at L31 and 0.38–0.87 at L34, against medians of 0.45 and 1.57.
  - Some carried a small, nearly constant offset: L31 #8697 had an RMS contribution 5.7× the median; L34 #2178 and #9458 had offsets 7–9× the median.
  - All the evidence above was measured with this loss in place.
  - E3g moved the same circuit into the least-used neurons. That cuts the overwritten contribution 3–6× in RMS and 14–26× in constant offset. It gave the same DI result (255) and comparable general-text behaviour (181/195 identical, all benign). It lost one more DD item on practice: #9 changed «arrivò» → «è arrivato» in the when-clause. The tense-relevant part «era incosciente da diversi minuti» was unchanged, so this was a near-tie flip scored by the judge. Under the pre-registered rule I kept E3e.
- **Remaining DI failures (practice):**
  - #414 and #464: "You had been waiting/practising…" stays «Avevi aspettato/praticato … per». The detector fires at full strength, but even κ_f = 4 does not flip them.
  - #551 and #573: "…for so long that…".
- **Remaining DI failures (unseen), mostly where the when-clause ends the activity:**
  - "…smoking … when he finally quit"
  - "…nursing the injured bird … when it finally flew away"
  - "…museum … when it finally reopened"
  - "…running the bakery … when he handed it to my mother"
  - "…renting the boat …"
- **Blemishes:**
  - #3: «Correvi per a malapena dieci minuti» (judged ok, awkward).
  - #133 and #218: «cucinava già da la mattina presto» («da la» for «dalla»).
  - One unseen DI sentence was restructured without a finite verb: «La nave alla deriva da tre giorni fu avvistata da un peschereccio» (same in all variants).
- **DD #141:** a past-perfect-simple "by the time … had taught for thirty years" now gets imperfetto + da.

## Compliance

- No training of any kind: no gradients, backward passes, optimizers or iterative fitting.
  - Every weight comes from closed-form linear algebra: ridge solves, mean differences, percentiles of cross-validated scores, and the explicit pair construction above.
  - Choices among a few candidates (κ, direction sets, neuron slots) were made by measured effect.
- The model was only run through `./wsp fwd`: 30 of 50 passes used.
- Kernel: 72 of 300 runs used.
- **Disclosure:** at kernel run 53 I re-exec'd the kernel process (`os.execv` of its own interpreter with its own argv, via `./wsp py k45_reexec.py`).
  - Why: glibc per-thread arena fragmentation from loaded activations had pushed VmData to its limit, so `W()` (which also backs `save_edit`) failed with ENOMEM.
  - Edits and the WORK folder persisted. After the restart I set `mallopt` (M_MMAP_THRESHOLD, M_ARENA_MAX) in the kernel to avoid it recurring.
  - Nothing outside this folder and WORK was read or written.

**Reproduce:** the final weights are rebuilt deterministically from files saved in WORK (`cfg_E3b.json`, `E3b_w_*`, `E3b_d_*`, `E3b_c_*`; copies under `E3e_*`) by `build_from_saved("E3b", {}, "E3e", kappa={"Pf": 3.0, "Pm": 3.0})`. That function is defined in `k56_build_E3c.py`, and the rebuild is exactly `k62_restore_E3e.py`; it needs `LN`/`SL` from the top of `k51_setup.py`.
- The detectors, thresholds and directions themselves came from `k54_build_E3b.py`, using `gram_w`/`solve` from `k52_build_E3.py`.
- Do not re-run all of `k51_setup.py` on the current WORK folder: it appends the HAB/COND rows to `X31_f16.npy`/`X34_f16.npy` a second time.
