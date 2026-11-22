# Pre-registration and outcomes

Each block below was written **before** the measurement it describes (22 Nov 2026), in Italian, in our working notes
(self-attested: there is no external timestamp); the original text is in [PREREG_original_it.md](PREREG_original_it.md).
Here: English translation, then what happened. The intervals in the predictions are our subjective ranges; the reading
rules use Wilson 95 % intervals on the measured shares.

## 1. The battery

**Predicted** share of «first action writes the fix», n = 30 per variant:

| A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|
| 0.55 [0.30-0.80] | 0.40 [0.15-0.65] (+ ~0.30 runs tests) | 0.45 [0.20-0.70] | 0.40 [0.15-0.65] | 0.85 [0.65-1] | 0.05 [0-0.20] | 0.30 [0.10-0.55] |

**Reading rules.**
1. *Hesitates at short prompts* if WRITES(A) ≤ 0.60, or if one of B, C, D, G is at least 0.30 below E with E ≥ 0.70.
2. *Does not hesitate* if WRITES ≥ 0.80 in A, C, D and G.
3. In between: partial effect.
4. Control: if F (reading on is the right move) exceeds 0.20, writing is not specific and the battery must be revised.
5. Format: if text-only + truncated answers exceed 20 % in a variant, the format is broken.

**Outcome (Inkling 2 Mini).** WRITES = 0/30 in every variant, including E: every prediction was wrong (too high)
except the control F. Rule 1
holds (A = 0). Rule 4: F = 0, but the control cannot discriminate when nothing writes. Rule 5: 0 text-only, 0
truncated. Variant H was added **after** the first run, because the preamble of E still said «find the cause» while the
user said «I already found it»: H removes that contradiction. The comparison with the two Qwen models (planned «after, if
the battery holds») is what made the result readable: Qwen3.8-27B writes 18/30 in E and 23/30 in H, Flash-Next 14 and 13.

## 2. Localisation (step 3)

- Margin at the decision point: m = logsumexp(logP[GO]) − logsumexp(logP[HESITATE]), HESITATE = {But, Also, Wait,
  However, Hmm, Actually} (+ «Let» when followed by check/look/verify); GO to be fixed by looking at the next-token
  distribution on 10 points of the selection half only, before the attribution.
- Split by situation: selection s01-s05, verification s06-s10. Target: points where the user stated the cause (E, H);
  the points where the model found the cause itself (variants A, C, D, G) to be measured alongside (same neurons?).
- Attribution Σ h·∂m/∂h over all expert neurons (24 layers × 128 experts × 1 152), summed over the 48 layer visits;
  one-by-one ablation of the top 12 on the verification half; control = random neurons in the same experts; locality
  curve k = 1, 4, 8, 16, 32.
- **Predictions:** first neuron Δm on verification 1 nat [0.3-3]; at k = 32 the share of points where GO beats HESITATE
  goes from ~0 % to 40 % [10-80 %]. If Δm < 1 nat at k = 32, the decision is distributed and the neuron route stops.
- The verdict is always in generation: battery H (with F as control) on the modified model.

**Amendment (after the 10-point next-token distribution, before the attribution).** No clean «apply» token exists (the
non-hesitation mass is spread over So, The, Note…), so **GO = everything that does not open a hesitation**:
m = logsumexp(logP[not HESITATE, not Let]) − logsumexp(logP[HESITATE]), HESITATE = {But, Also, Wait, However, Hmm,
Actually, BUT}; «Let» (usually «Let me check…», sometimes «Let me apply») excluded from both sides. Order: attribution
on the 48 points → locality curve → one-by-one ablations (k = 8) → the self-found-cause points after.

**Outcome.**

| | predicted | measured |
|---|---|---|
| first neuron, Δm on verification | 1 nat [0.3-3] | **+0.63** (inside) |
| top 32 together, Δm | ≥ 1 nat, else stop | **+2.37** (random 32 in the same experts: −0.01) |
| share of verification points where GO wins at k = 32 | 40 % [10-80] | **15 %** (inside, low end; base 8 %) |

The one-by-one ablations were stopped after the locality curve to move straight to the generation test, and the
self-found-cause points (pre-registered «alongside», then «after») were **not run**: the generation test showed that the
edit does not make the model write, and the study moved to the agentic sessions.

## 3. Generation test of the weight edits

Artefacts with the neurons zeroed in the weights: **E32** (top 32), **C32** (32 random neurons in the same experts),
**E8** (top 8); battery H, A, F × 3.

**Predicted for E32:** H writes 6/30 [1-15], A 2/30 [0-8], F 0-2/30; C32 = base (0/30). Success if H ≥ 5/30 with
C32 ≤ 1/30 and F ≤ 2/30.

**Outcome: not met.** H writes 0/30 with every artefact. What changed is the reasoning (not pre-registered as a
criterion, reported as an observation): answers matching the trap keywords 16/30 (base) · 14 (C32) · 10 (E8) ·
**1 (E32, a false positive)**; median reasoning length 185 · 181 · 121 · **37** words. Split by half (selection /
held-out situations): base 8 / 8, C32 8 / 6, E8 2 / 8, E32 0 / 1 out of 15.
