# Notes (Gemma4 12B EN->IT, DI fix by weight editing)

## Facts
- practice ids are 0-based: id = (line number in practice.tsv) - 1. practice.tsv has 2 cols: cat, en.
- Global layers (5,11,...,47): no v_proj (K=V), k_proj (512,3840), q_proj (8192,3840), o_proj (3840,8192).
- Sliding layers: q_proj (4096,3840), k/v_proj (2048,3840), o_proj (3840,4096).
- MLP: gate/up (15360,3840), down (3840,15360).
- Only command allowed: ./wsp. Use Read/Write tools for files. Don't touch anything outside this folder.

- generate: "ids" and "texts" CANNOT be mixed (texts silently ignored when ids given) -> wasted pass 2 partly.
- W() can read embed_tokens (262144x3840) and final norm weight (not editable). transformers installed in kernel; no tokenizer access (don't read outside WORK).

## Baseline (original weights)
- pass1: DI 203 ok/55 err/2 other (of 260); DD 124/130 ok; PERF 60/60; DURP 59/60.
- pass2: HAB 58/58 (+2 in pass1 ok), COND 36/40.
- DI err ids (57 incl other): 3,8,69,78,92,94,112,116,128,143,153,161,165,171,173,188,198,209(o),212(o),217,223,236,247,273,291,294,298,300,325,334,369,382,384,389,399,411,414,447,448,464,501,504,523,524,530,536,538,540,551,561,562,571,573,583,587,588,593
- DD err ids: 107,144,196,285,550,566 (all imperfetto+da). Note judge accepted imperfetto for DD 9,97,252,340 ("era aperto da", "era in prigione da").
- DURP err: 425. COND err: 19 + 3 others.
- Typical DI errors: "Aveva fumato per", "era rimasto senza ... per", "eravamo stati in viaggio per". Correct DI outputs often "stava + gerundio + da" (judge ok) or plain imperfetto + da.

## Read-pass facts
- load_acts(name) -> dict: acts{"resid.L": [tensor per item]}, meta[list of {tokens, logprobs, n_prompt_tokens}], items, positions.
- positions "completion": rows = completion token positions (row r predicts token r+1); first token's predicting position (npt-1) NOT included. "all": rows = npt + ncompl; row npt-1 predicts token 0.
- logit lens on resid.47: x = r*rsqrt(mean r^2+1e-6)*normw ; lg = x@E.T ; 30*tanh(lg/30) -> matches reported logprobs exactly.
- Token ids: Ave=182706, F=236811, ▁aveva=51542, ▁avevano=139292, ▁rim=23456.
- Massive constant dim 1750 in residual (normalized ~ -60 at L26, sd 0.3) -> usable as bias.
- main1 (pass5): practice B (610, first) + A alt-prefixes (387), completion positions, resid at 12..47 (even+47). WORK/dec_rows.json = decision tokens/margins.
- L47a (pass6): 610 practice B + 200 general, positions all, resid 46/47, attn_out/mlp_out/mlp_act 47.

## Findings
- DI-err imp-margins (logp alt-imperfetto - logp trapassato) median -5.3, range -20..0; DD-ok margins median -14, some near 0 (66,18,141,308,210,262,235,539). Uniform push breaks DD.
- Ridge probe DI vs DD at decision pos (t>=1): CV acc ~0.92 at L26-38; on DI-err vs DD AUC ~0.94 at L26-30 (DI-ness present before decision locks in ~L34+).
- Plan A: layer-47 MLP neuron repurposed as thresholded detector (gate = probe dir + bias via dim1750, up = const) with down col = suppress trapassato aux tokens {Ave, ▁aveva, ▁avevano, ▁avevamo, ▁avevo, ▁avevi, ▁ave, ▁rim, ▁stato, ▁stata, ▁stati, ▁state}. Effect at L47 is local to next-token logits -> computable from captured acts.

## Edit v1 (layer 47, neuron 14123) -- CURRENTLY SAVED
- k_edit1.py: G[J]=alpha*(w - theta*mhat), U[J]=mhat, D[:,J]=beta*p ; alpha=10, theta=0.45, P0=6 (beta=62.6)
- w = uncentered weighted ridge probe on x_hat47 (WORK/probe47_w.npy); mhat = mean(x_hat)/|mean|^2 (WORK/push47.pt); p = constrained LSQ aux suppression w/ protected tokens (WORK/push47c.pt)
- layer scalar 47 = 0.0478; simulator (k_sim5/k_ver1) matches real forward: corr 0.99996.
- CV simulation (honest): DI-err decisions fixed 35/53, DD flips 4 (18,141,193,262), GEN changes 2.
- pass 8 generate both (in-sample!): DI 203->241 (37 fixed), DD 124->124, PERF/DURP same. 448/512 identical.
- failure mode: after blocking 'Ave', model sometimes picks essere+participle: 'Eravamo vissuti', 'eravamo camminati', 'era abbaiato' (still err).
- mid-layer (L24-28) probes are better on hard cases (DI 153,173,325,504,536,561,571; DD 18). Idea: mid-layer detector -> flag -> L47 suppressor.

## Held-out + data (passes 9-13)
- pass9: old 200 general texts both: 200/200 identical (in-sample negatives though).
- pass10: NEW 370 texts (120 DI, 100 DD, 150 GEN; req_new_gen.json) both with edit v1: DI errs fixed 13/25, DD 0/100 changed, GEN 1/150 changed (good change). 
- new47 (pass11): L46/47 acts for the 370 new texts -> WORK/new47pos.pt (XH,Y,R47,POS,gap,lab,wt,kind; load weights_only=False).
- Combined L47 probe CV (k_comb1): theta .4 -> prac 36/53, new 14/24 fixed; broken prac 6 (DD 18,141,193,262,341 + gen 675/802), new 0. L47 linear probe near ceiling.
- k_mid2: at aux-top1 decision positions, L26-30 probes recall 24-28/33 DI-err at 2-4 DD FPs vs L47 11-14/33. => plan: L29 detector neuron writes flag in low-read residual direction; L47 neuron reads flag -> aux suppression.
- L29a (pass12): practice B 610 + old general 200, all positions, layers 28,29 resid/attn_out/mlp_out. new29 (pass13): same for 370 new.

## Edit v2 (two-stage) -- SAVED at pass 14 (k_edit2.py, cfg in WORK/edit2_cfg.json, parts WORK/edit2_parts.pt)
- L29 neuron 4205 (dead): gate = 5*(w29 - 0.32*e1750/0.091) [w29 = full-data ridge lam100 on x_hat29, WORK/w29_full.npy], up = e1750/0.091, down = flag f_y = u0*rms_r29*rms_y29/sig29 * (delta / wpf29), u0=1.5, rms_r29=1.995, sig29=0.6015.
- delta = quiet residual direction: top SNR@L47 within 20 smallest eigvecs of read-Gram of layers 30-47 (WORK/readGram_eig.pt, flag_delta.pt).
- L47 neuron 14123: gate = 10*(g - (g.mu47 + 0.15)*mhat47), up = mhat47/10, down = 14*scale*p. g = C47^-1 dx/(dx C47^-1 dx), dx = u0*delta*wpre47.
- CV two-stage sim (k_two1): th29 .3: prac 37/45, new 20/24 fixed, broken 4+1; th29 .35: 34/45, 16/24, broken 3+0.
- pass14: read ver2 (practice+gen, edited, L46/47) -> analysis k_ver2.py CRASHED the kernel (likely OOM: too many big objects in persistent namespace). LESSON: del big vars + gc.collect(); load one acts set at a time; avoid E (4GB) duplicates.

## Outage
- After pass 14 + crashed k_ver2 (run ~46), ws unreachable for a long time (status/py/fwd all fail; fwd failures not counted).
- Prepared if it returns: k_mem0.py (cleanup), k_ver2a.py + k_ver2b.py (memory-safe verification of flag), req_fresh_gen.json (340 fresh held-out texts, both), req_hc_both.json (HAB/COND both), req_base1.json (512 practice, both), req_gen0.json (200 gen, both).
- final_answer.md written for the v2 state (with caveat that v2 not validated end-to-end).

## Budget log
- kernel runs: 2 used (info)
- fwd passes: 0 used

## Plan
1. Baseline generate (original) on DI+DD+PERF+DURP (512) -> find failing DI, DD ok rate.
2. Read passes: teacher-force correct/incorrect translations, capture resid/mlp_act at decision position.
3. Build conditional rank-1 / key-value edit (closed form), validate with generate "both".
