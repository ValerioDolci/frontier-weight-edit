# Notebook (Gemma 4 12B-it EN->IT, DI fix by weight editing)

## Facts
- Kernel: WORK=<server>/session/work ; globals: W, names, save_edit, drop_edit, edits, load_acts, practice, SESSION, np, torch
- Kernel print output truncated at 10k chars -> cannot dump big JSON from kernel; write request JSONs by hand.
- practice ids are 0-based = line order in practice.tsv (id 0 = line 1)
- W() can read embed_tokens (262144x3840, tied head) and final norm. Logit lens at L47: h=resid47/rms*gfin; logits=30*tanh(E@h/30) -> matches greedy.
- Tokens are SentencePiece with '▁'. Some ids (guesses): ▁aveva 51542, ▁avevano 139292, ▁era 6933, ▁erano 63523, ▁stava 175152, ▁sta 48883, Ave 182706, Er 236788, Sta 894, ▁in 528
- layer_scalar per layer ~0.6-0.9 mid (L0 .053, L11 .005, L47 .048). post_ff norm weight ~1.3 mean at L30.
- Global layers (5,11,...) have NO v_proj (k=v shared)
- load_acts(name) -> dict(acts={"resid.L": [tensor(npos,d)] per item}, meta=[{tokens,logprobs,sum_logprob,n_prompt_tokens}], items, positions)
- read with "last" + empty "it" works (= last prompt token). Memory estimate ~2x mine; mlp_act is expensive (recompute from x instead).

## Baseline (pass 1, original): DI 203/260 ok (56 err, 1 other); DD 122/130; PERF 31/31, DURP 31/31, HAB 30/30, COND 27/30
- DI err ids: k_dp.py ERR_DI; DD err: 107,144,196,252,285,340,550,566
- Decision points (prefix before tense choice, base word, alt word) in k_dp.py -> WORK/dp.json

## Passes
1 gen orig 512 practice (DI+DD+122 other)
2 gen orig 460 general texts (general_texts.json) -> passes/pass_2.json
3 read 'last' at decision prefixes, all 48 layers, resid/attn_out/mlp_out/mlp_act -> 'dp_last' (WORK/dp_*.npy)
4 read 'completion' full baseline outputs (512 practice + 460 general) resid+attn_out L28-33,47 -> 'full_comp'

## Findings
- CV ridge probe at decision position: DI vs DD AUC ~0.96 at L28-34; hard case DI-err vs DD-ok AUC ~0.95 (MLP input L29-33)
  => model encodes DI-ness even when it outputs trapassato. DI/DD vs other cats AUC ~1.0 from L20.
- Output-tense probe (imp vs trap) AUC 0.93+ from L28.

## Edit design (current)
- 2 repurposed MLP neurons per stage (least-used): neuron A gate=kappa*w (w: ridge detector on MLP input x, target y-0.78, uncentered),
  neuron B gate = kappa*w - c*u ; up = u (min-variance unit-mean functional, u.x = 1 +- 0.008); down A = v/c, B = -v/c
  => output = clamp(kappa*s,0,c)/c * v  (saturating gate). kappa=40, c=4.
- v = rms_m * (alpha*dir/s_L) * g/(g^2+0.2^2) (post_ff norm g, layer scalar s_L).
- Stage 1: L30 neurons [13770,12872], dir = 24*unit(d_imp30) + 8*unit(d_start30), detector incl. prompt positions as negatives, threshold shift t30=-0.06
- Stage 2: L40 neurons from build2.json, dir = d40 (imp-trap at decision, start+subj, under edit2), alpha=|d40|~36-38, t40=-0.03
- Kernel helpers defined: mlp_in, neuron_mult, build_pair, const_dir2, margins, summarize (k_marglib.py, k_build2.py)

## Results
- margins (in-sample, 57 DI err): L30 only a=12: 21 fixed; a=28: 36; a=43: 36 (saturates). 2-stage (L30 retrained+L40): 38, DD-ok untouched.
- Gate on ORIGINAL L30 detector fired on BOS positions 0-3 (g~0.21) and English prompt tokens -> fixed by adding prompt negatives.
- HELD-OUT (train=ids%3!=0, val=ids%3==0 + 98 REST): pass 11 gen both: DI 70->83/93 (14 fixed, 1 ->other), DD 37->37/41, PERF/DURP/HAB/COND unchanged.
  Note: generation+judge nondeterministic run-to-run (DD 9 changed verdict on original).
- ids+texts in one generate request: texts silently ignored!
- pass 12: fresh general texts (general_texts2.json, 231): 223 identical.

- v1 (final candidate) = k_build_full.py (all-data, t30=-0.06, t40=-0.03). Validated: pass14 in-sample DI 203->246/260, DD 122->121 (judge noise),
  pass17 fresh synth DI ~19/23 errors fixed, DD 0/50 broken; pass18 98 REST identical; pass19 general 448/460 identical (trivial changes); pass16 paragraphs 56/60.
- AUGMENTATION experiment (pass 20 read synth_last; k_aug.py, k_val_build2.py): !!! ACTIVE EDITS = train-only augmented validation version !!!
  -> after validation MUST rebuild final (either v1 via k_build_full.py or augmented-all-data version).
- pass 21 held-out (train-only AUG): DI 70->85/93 (16 fixed, 1 other) vs 14 fixed without aug; DD 37/41 unchanged; others unchanged.
- FINAL AUG built (k_build_final.py, run 55): ACTIVE EDITS = final augmented all-data version. To revert to v1: run k_build_full.py.
- pass 22 (FINAL, fresh general 231): 219 identical; changes = 3 DI-construction fixes, «stava giacendo», synonyms/rewordings.
- pass 23 (FINAL, in-sample 512): DI 203->247 (45 fixed, 1 broken = 86 garbled; 209 other->ok, 390 ok->other), DD 122->120:
  id 9 = judge noise (same «era incosciente da», only arrivò->è arrivato), id 262 = real flip («era in funzione da», borderline item);
  PERF 31/31 DURP 31/31 HAB 30/30 COND 27/27; 398 identical.
- pass 24 (FINAL, 60 paragraphs): 55 identical; 4 trivial synonyms (same 4 as v1 in pass 16 -> due to the repurposed neurons),
  1 regression: «Si era allenata» -> «Aveva allenato» (dropped reflexive, v1 left it untouched).
- Known artifacts (both versions): progressive «stava V-ndo» overuse, «abbaiaando» (83, 411), «stava giacendo» (156, 568), dropped «si» (151, 300),
  garbling (86, 371 «da ore sei», 517), when-clause -> trapassato (98, 229, 559), «per» kept (165, 411).
- run 56: W() returns ORIGINAL weights (each build clones original, overwrites only the 2 neuron rows/cols). n40 = [11980, 11803], alpha40 = 36.46.

- pass 25 (FINAL, 155 FRESH unseen sentences req_gen_fresh3.json, judged by me): 132 identical.
  DI 60: 10 baseline errors -> 10 fixed; 0/50 correct broken (progressive/lexical rewordings only).
  DD 75 (incl. 15 'had X-ed for N when <terminating event>'): 0 flipped to imperfetto; 1 flipped imperfetto->trapassato
  ('worked at the bank ... when he was made redundant', = DD-correct); 2 lexical/when-clause tweaks. Controls 20/20 identical.

- pass 26 (FINAL, 98 REST ids never used for fitting): 98/98 identical; COND 10/10, DURP 28/29, HAB 29/30, PERF 29/29 (same as original).

- pass 27 (FINAL, 460 general texts = training negatives): 448 identical; 12 changes all neutral rewordings (abitazioni->case,
  verrà->sarà, società->azienda, «Cenavamo»->«Stavamo cenando», «era in viaggio da»->«stava viaggiando da», «si è schiantato»->«ha sbattuto"...).
  NB: DI-like clauses inside general texts were negatives in training -> edit deliberately does not fire there (e.g. «avevano marciato per due giorni»).

## DECISION
- Keep FINAL AUG (k_build_final.py). Reason: held-out DI +2/93 vs v1-style (16 vs 14 fixed of 23 errors), held-out DD equal (37/41),
  in-sample DI +1; cost: 1 borderline in-sample DD flip (262) + 1 paragraph dropped-«si». Expected net positive.
- Confirmed by passes 25-27. Budget at end: 27/50 passes, 56/300 runs. Submission = active edits (6 tensors, L30/L40 MLP).

## Plan
- Repurpose 1 MLP neuron at L~30: gate = DI detector (ridge on MLP input x, negatives = DD, other, general positions),
  down column = steering direction toward imperfetto. Evaluate via read passes (margins) then generate.
