# Final answer

## TL;DR
I edited **two previously dead MLP neurons** (6 tensors touched, one row/column in each):
a **detector at layer 29** that recognises "DI" decision points (the Italian main verb of an English
*had (been) … for/since …* clause describing a state still going on at a past moment) and writes a tiny
"flag" into a residual direction that later layers practically do not read, and a **suppressor at layer
47** that reads the flag and lowers only the logits of the trapassato auxiliaries (`Avev-/aveva/avevano/
avevo/ave-`, `rim(asto)`, `stato/stata/stati/state`). Wherever the flag is absent the model is unchanged
in practice. All weights were computed in closed form (ridge regressions, eigen-decompositions,
constrained least squares) from weights and from activations returned by read passes; no gradients,
no training, and the model was never run in the kernel.

**Important caveat:** the remote workspace (kernel and forward server) became unreachable right after the
first verification read of this final two-stage version (pass 14 of 50), most likely because my
analysis kernel ran out of memory. The two-stage version therefore has not been checked by an end-to-end
generation run; its evidence is exact simulation of the layer-47 stage plus cross-validation of the
detector. The earlier single-stage version (v1, same suppressor but with a layer-47 detector) was
validated end-to-end (numbers below).

## The saved edits
| tensor | change |
|---|---|
| `layers.29.mlp.gate_proj` row 4205 | 5·(w29 − 0.32·e₁₇₅₀/0.091) — linear DI probe with threshold (bias via the near-constant dim 1750 of the normalized MLP input, 0.091 ± 0.002) |
| `layers.29.mlp.up_proj` row 4205 | e₁₇₅₀/0.091 (≈ 1 at every position) |
| `layers.29.mlp.down_proj` col 4205 | flag f = c·(δ ⊘ w_postFF29), δ = quiet residual direction (see below), sized so a unit activation adds ≈ 1.5 RMS units (≈ 2 % of the residual norm) |
| `layers.47.mlp.gate_proj` row 14123 | 10·(g − (g·μ + 0.15)·m̂): whitened read-out of the flag at the layer-47 MLP input, threshold 0.15 |
| `layers.47.mlp.up_proj` row 14123 | m̂/10 (≈ constant) |
| `layers.47.mlp.down_proj` col 14123 | 14·s·p: p lowers the 10 trapassato token logits by exactly the same amount while keeping ~1500 plausible alternatives (`▁stava, ▁era, ▁in, ▁da, F…, Vive…`) and the rest of the vocabulary nearly unchanged |

Both neurons were essentially dead before (RMS output contribution 0.013 vs an MLP output norm of ≈ 37
at layer 29, 0.004 vs ≈ 250 at layer 47), so repurposing them costs nothing measurable.

## Why this design (diagnosis)
1. **Where the error is decided.** Teacher-forced reads with paired correct/incorrect continuations
   showed that the DI error is decided at a single token, the first token of the main verb ("Ave" vs
   "F[umava]", "▁rim" vs "▁senza", "▁stati" vs "▁in"). Once the imperfetto is emitted the model continues
   with "da" by itself (log-prob ≈ 0).
2. **Why a uniform push fails.** At that token, DI errors need 0.1–12 nats while several DD items are
   within 0.2–1 nat of flipping (e.g. DD 18, 141), so any ungated change would break DD.
3. **Where DI-vs-DD is visible.** Closed-form ridge probes on the residual stream: DI vs DD at the decision
   point is linearly decodable everywhere from layer ~24 on. But among positions where the model is about
   to emit a trapassato auxiliary, a layer-26–30 probe recalls 24–28 of 33 DI errors at 2–4 DD false
   positives, against only 11–14 for a layer-47 probe: by layer 47 the model has already committed to its
   (wrong) reading.
4. **Why suppress at layer 47.** A layer-47 MLP edit only changes the next-token logits at the positions
   where it fires, so its effect can be computed exactly from captured activations (checked against a
   real forward pass: correlation of predicted vs observed log-prob changes 0.99996, mean abs error 0.04
   nats, no unpredicted changes). Suppression changes the output only where an auxiliary is top-1, so a
   misfire elsewhere is harmless.
5. **Why the flag channel.** I built the Gram matrix of every downstream read of the residual stream
   (q/k/v and MLP gate/up of layers 30–47 including the RMSNorm weights). Its spectrum is extremely
   anisotropic: the chosen direction δ has read gain ≈ 0.036 (median direction 0.18, mean ≈ 88), yet
   after whitening by the natural covariance at the layer-47 input it is detectable with SNR ≈ 9 per
   unit. Measured read-out noise on unflagged activations: std 0.076, 99.9th percentile 0.32 (threshold
   0.15 on a signal of ~1–4).

## Evidence
**Baseline (original weights):** DI 203/260 ok (55 err + 2 other), DD 124/130, PERF 60/60,
DURP 59/60, HAB 60/60, COND 36/40.

**v1 (layer-47 detector + same suppressor), end-to-end generation:**
- Practice (in-sample for the detector): DI 203 → 241 ok (37 fixed, judge-scored), DD 124 → 124,
  PERF 60 → 60, DURP 59 → 59; 448/512 outputs identical. Typical fixes: "Aveva fumato per vent'anni" →
  "Fumava da venti anni…", "Il villaggio era rimasto senza elettricità per una settimana" → "…era senza
  elettricità da una settimana…", "eravamo stati in viaggio per quasi venti ore" → "eravamo in viaggio
  da quasi venti ore".
- 200 general sentences: 200/200 identical (these were training negatives).
- **Held out**, 370 sentences I wrote (120 DI, 100 DD, 150 general news/encyclopedic): 13 of the 25
  baseline DI errors fixed; 0/100 DD changed; 1/150 general sentences changed ("La squadra non aveva perso
  una partita in casa da due anni" → "…non perdeva una partita in casa da due anni", a correct change).
  Every other changed DI output stayed imperfetto + da (e.g. "stava suonando" → "suonava").

**v2 (saved; layer-29 detector, retrained on practice + the 370 new sentences), simulated:**
item-grouped cross-validation of the detector, with the exact layer-47 computation and the measured
read-out noise. At the chosen settings (detector threshold 0.32): ≈ 35–37 of 45 practice DI-error
decision points and ≈ 16–20 of 24 new ones flip to a non-trapassato token. "Risky" flips (DD decisions or
general-text auxiliaries) number 3–5 of ~360 such positions; known ones are DD 26, 141, 262 (borderline
items) and one general sentence whose new form "soffriva di mal di testa da mesi" is itself correct. v1
at a comparable risk level fixed ≈ 36/53 and 14/24.

## Known limitations and side effects
- When "Avev-" is blocked, the model sometimes switches to an *essere* trapassato with the wrong
  auxiliary ("Eravamo vissuti", "era abbaiato"). These were errors before and stay errors, but they are
  ungrammatical. In v1 this happened in about 8 of 45 targeted cases.
- Not covered: reflexive trapassato ("si era allenata"), "era durato", and DI sentences whose *when*
  clause clearly ends the state (the detector reads these as DD).
- DI items that were already right sometimes change style ("stava cucinando" → "cucinava"), still
  imperfetto + da.
- The layer-29 → layer-47 flag propagation was designed from the weights but not confirmed, because of
  the outage. If the flag is attenuated, the effect degrades towards "no change" (the read-out has a
  threshold), not towards random damage. If it were distorted, intermediate layers would see a perturbation
  of ≈ 2–8 % of the residual norm in a direction they barely read.
