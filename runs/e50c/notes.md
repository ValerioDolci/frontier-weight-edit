# Notebook — Gemma 4 12B DI edit

## Facts
- practice ids are 0-based: id = line number in practice.tsv - 1.
- 616 editable tensors. Global layers (5,11,...,47) have no v_proj (K=V), head dim 512.
- RMSNorm weights are direct multipliers (input_layernorm mean|w| ~9 at L0; post_ff ~11.7; post_attn ~1.35 at L0).
- WORK = <server>/session/work (kernel side)

## Log
- run 1: explored names/shapes.
- pass 1 (generate, original, DI+DD+PERF+DURP): DI ok 204/260 (err 55, other 1 = id 212), DD ok 123/130 (err 6: 107,252,285,340,550,566; other 66), PERF 60/60, DURP 59/60 (err 425).
  DI err ids: 3 8 69 78 92 94 112 116 128 153 161 165 171 173 188 198 209 217 223 236 247 273 294 298 300 325 334 369 382 384 389 399 411 414 447 448 464 501 504 523 524 530 536 538 540 551 556 561 562 571 573 583 587 588 593 (+212 other)
  Error patterns: "aveva/avevano + participle + per" (most); "era rimasto/eravamo rimasti ... per"; "eravamo stati ...". Most errors are past perfect PROGRESSIVE (had been V-ing). DD mostly "before/until"; DD with when/by the time are the risky ones (9 18 26 27 45 97 107 130 141 142 144 196 235 252 262 326 340 341 355 454 479 506 550 564 566 595 606).
  Judge accepts "stava + gerundio + da", "era in carica da", etc. for DI.
- pass 2 (generate HAB+COND original): HAB 59 ok + 1 other (356 typo), COND 40/40. LESSON: ids+texts in same request -> texts IGNORED (pass wasted for texts).
- general_texts.json = 195 general-domain EN sentences. pass 4 = their original translations (passes/pass_4.json).
- pass 3 (read r1, resid L4..47 x14, all positions, 605 items): order DIW 0-54 (model wrong), DIC 55-109 (my corrected), DIR 110-313, DD 314-443, DDA 444-484 (alts), DURP 485-544, PERF 545-604.
  prompt = 68 tokens + EN. load_acts returns {"acts": {"resid.L": [tensor(rows,3840)]}, "meta": [{tokens, logprobs, sum_logprob, n_prompt_tokens}], "items", "positions"}; row r = token r of full seq; decision row = n_prompt + t - 1.
  WORK files: pairs_r1.json (DI/DD pairs with decision t, margins), dec_r1.json (decision rows), alt_dir.json (DI-right trapassato alts).
  Findings: DIW margins (imp-trap at decision token) +0.1..-20; DD before/until -9..-25 (safe); DD when/by-the-time close (18:-0.25, 141:-0.6).
  DIR vs DIW linear sep (mean-diff AUC): L4-25 ~0.65, L31 0.73, L34 0.84, L37 0.88, L40 0.95 => decision forms L31-40.
  LDA detector DI-vs-DD held-out AUC: L28 .93, L31 .944, L34 .962; DI vs other positions ~.99+.
  Massive dims 1750, 830 (normalized ~-51, -30), pre_ff weight there ~0.002 => x_1750 ~ 0.1 near-constant (bias candidate). BOS/first prompt positions differ (sign flips at L31/34!).
  post_ff_norm weights rms ~1.3-2.0 (L28-34); pre_ff rms 4-25.
- pass 5 (read r2: resid/attn_out/mlp_out L29-34, all pos, 704 items): order DIW 0-54, DIR 55-258, DD 259-388, DURP 389-448, PERF 449-508, GEN 509-703.
  MLP input reconstruction: h_mid = resid(L-1) + rmsnorm(attn_out(L))*w_post_attn; x = rmsnorm(h_mid)*w_pre_ff. resid(L)=s_L*(h_mid+rmsnorm(mlp_out)*w_post_ff), s = {30:.7227,31:.6993,32:.5546,33:.629,34:.75}. Verified (err 0.3%).
  Const direction c (ridge of 1 on x): c.x = 1.000 +- 0.008 incl. system rows => bias trick for gates.
  Dead neurons exist (gate.x never >0): L31 e.g. 4263, 5663, 11046, 5100, 9928, 9922; L34 10083, 4575, 10876, 4362.
- EDIT DESIGN ("detector neuron pair"): neuron j1: gate=a(w-theta c), up=b c, down=v; j2: gate=a(w-(theta+Dl)c), up=b c, down=-v; a*b*Dl=1 => adds v*clamp((w.x-theta)/Dl,0,1). w = ridge detector (closed form), v = rms_m*kappa*d/s_L/w_postff (d = mean DIR - mean DIW resid at decision rows; zero dims w_postff<0.1).
- E1 (L31 single pair, kappa 2, blended dir): DI margin +3.2 mean; GEN tiny. Too weak.
- IMPORTANT: in-sample detector scores inflated -> thresholds must come from CV / held-out scores. System rows: weight 1.0, verify max score << theta.
- E2a (L31+L34, separate first-token (Pf) / mid (Pm) pairs, kappa 3, detectors+dirs trained on EVEN r2 items, theta=heldout P p10, top=p50; lam Pf .01, Pm .1 (L31) / 1.0 (L34)):
  read (pass 7): DI held-out 16/27 margin>0 (mean +8.8), train 26/28; DD only 141 flipped; GEN few changes (DI-like sentences).
  generate (pass 8, both, DI+DD+PERF+DURP): DI 204->251 (47 fixed, 0 broken; held-out ~21/27 fixed); DD 123->120 (141, 341 real breaks; 9 judge noise); PERF/DURP all identical outputs.
  Remaining DI err: 173 273 369 414 464 551 571 (+556 other, 212 other).
- Kernel memory: free old load_acts results (r1 acts ~6GB, r2 ~8.5GB). Keep M1 (r1 meta) global.
- Reusable eval: req_p6_read_E1.json (saves 'e1'), k23_eval2.py eval2("e1"); req_gen_main.json (DI+DD+PERF+DURP both).
- pass 9: read r3 (new synthetic data, ORIGINAL weights confirmed): 120 DI imp (0-119) + 120 DI trap (120-239) + 90 DD (240-329) + 60 NEG (330-389); layers 30,31,33,34,36,37 resid+attn_out. New DI: 72/120 imp-preferred.
  Saved in WORK: X31_f16.npy, X34_f16.npy (41148 rows MLP inputs, r2+r3), Xmeta.npz (grp/fold/src/item/row), dirdata.npz (367 DI decision resid L31/L34 + lab W/R + first), meta_r2_r3.json.
- pass 10 (gen GEN both, E2a): 188/195 identical; 5 DI-like sentences corrected (aspettava da, era senza elettricità da...), 2 neutral synonym swaps.
- pass 11 (gen HAB+COND both, E2a): HAB 59/60 unchanged verdicts; COND 39/40 (255 "Andremmo"->"Andreiamo" garbled = first-token push misfire). 94/100 identical.
- MEMORY PROBLEM: kernel VmData stuck at 35-39 GB (glibc arena heaps from per-thread allocations of loaded activations), RLIMIT_DATA=56GB; W() mmaps 24GB safetensors -> ENOMEM. save_edit also calls W(). Kernel port 18611 behind gateway 18620; launched as <server>/wsp/venv-hf/bin/python -u <server>/wsp/code/wsp_kernel.py. Edits stored as .pt on disk.
  => Decided to re-exec kernel (os.execv same argv) to reclaim address space. AFTER RESTART: avoid loading big activation sets in many runs; keep load_acts small; free immediately.
- run 53: kernel re-exec'd (k45_reexec.py). SUCCESS: VmData 1.7GB, W() works, edits + WORK intact, runs counter preserved (54 after). All kernel globals/functions were reset -> must redefine helpers (rmsn etc.).
  Use ctypes mallopt(M_MMAP_THRESHOLD=-3, 131072) at start of heavy scripts.
- pass 12: read r4 = HAB+COND (100) activations -> appended as negatives to X31/X34 (44619 rows; Xmeta updated, src=4).
- E3 (k52) -> E3b (k54) = CURRENT EDIT: detectors trained on ALL data (r2 practice + r3 new + r4 HAB/COND), 2-fold CV thresholds
  (Pf theta=CV P p15, Pm theta=p10, top=p50), lam L31 Pf .01, Pm .01, L34 Pf .01, Pm .1; FIRSTNEG weight 1.5 for Pf;
  directions from PRACTICE ONLY (new-data labels noisy, cos~0.1-0.3 with practice dirs); kappa_f 3.0, kappa_m 3.5; same neuron slots as E2a.
  cfg_E3b.json, E3b_w_*, E3b_d_*, E3b_c_* in WORK. Builder: k54_build_E3b.py (needs k51_setup + k52 helper funcs gram_w/solve).
  RESULTS: pass 13 unseen texts (100 DI/80 DD/60 other, my judgement): DI 76->95 correct (19/24 fixed, 0 broken); DD 0/80 broken; others 1 benign change.
   pass 14 practice main (judge): DI 204->253 (50 fixed, 1 broken: 125 'Eravamo aspettando' hybrid); DD 123->122 (141 flips; 107/141 'unjudged'); PERF 60/60; DURP 59/60 unchanged.
   pass 15 HAB/COND: COND 40/40 (255 fixed), HAB 60/60; 97/100 identical.
   pass 16 GEN: 184/195 identical; all 11 changes benign/improvements.
  Known blemishes: hybrids 'Eravamo aspettando'(125), 'già da la mattina'(133,218), 'Correvi per a malapena'(3).
   pass 17 paragraphs (40, both): 33/40 identical; 2 DI fixes inside paragraphs; 5 neutral rewordings.
- pass 18: generate ORIGINAL on the 120 r3 DI sentences -> generation-based labels (W = trapassato in actual output):
  W = 24 28 29 31 34 37 39 41 47 49 50 51 64 69 76 80 81 82 83 84 89 104 110 (item 20 borderline, skipped).
- run 59 (k55): directions with gen labels. Practice-only vs r3-only cos 0.26-0.35 (Pm), split-half rel ~0.43-0.53 each
  -> directions genuinely differ by set (expected cos if same true dir ~0.65). Combined (prac+r3): Pf cos .97 w/ practice (r3 has only 4 W first-token items),
  Pm cos .85 (L31) / .89 (L34), rel .49/.59.
- run 60 (k56): E3c = E3b detectors (loaded from saved w/c/theta) + COMBINED Pm directions (Pf unchanged). Builder build_from_saved(det_tag, d_over, tag)
  -> restoring E3b = build_from_saved("E3b", {}, "E3b_r").
- pass 19 (unseen, E3c): DI 95/100 same 5 failures as E3b, 0 broken; DD 0/80; 191 identical (E3b 187).
- pass 20 (practice, E3c): DI 253 (fixes 125, breaks 323 'si era nascosto ... da lunedì'), DD 122 (141), 387 identical; new glitch 430 'piovere a dirotto'->'scendere a picco'. => tie with E3b, not adopted.
- run 62 (k58): Pf CV scores. p414/p464 ('You had been ...' -> 'Avevi ...') score ABOVE full-on point (in-sample ~0.9-1.0 > top) -> failure is push strength, not detection.
  Highest FIRSTNEG DD scores: p196 (judge accepts imperfetto anyway), p252 (already err), p341/p26 partial at L31 only.
- E3d = E3b with kappa_f 4.0 (pass 21): DI 252 (414/464 still unchanged) -> rejected.
- E3e = E3b with kappa_m 3.0 (pass 22): DI 255/260 (51 fixed, 0 broken; 125 OK, 198 fixed), DD 122 (141), PERF 60, DURP 59, 390 identical.
  pass 23 unseen: DI 95/100 (19/24 fixed, 0 broken), DD 0/80 tense changes, others 1 benign; 192 identical.
- E3f = kappa_m 2.5 (pass 24): DI 254 (1 'other') -> rejected (pre-registered rule: must be no worse on practice AND unseen).
- E3e restored (run 67). pass 25 HAB/COND: COND 40/40, HAB 60/60, 97 identical. pass 26 GEN: 184/195 identical, 11 changes all benign
  (5 DI-type fixes). pass 27 paragraphs: 34/40 identical (2 DI fixes, 4 neutral).
- run 68 (k63): firing rates of E3e detectors on stored activations: SYS max -0.97..-0.99; general-text output tokens active 0-0.74%;
  source tokens <=0.19%; DD decision rows 0-0.6%; DI decision rows 99-100%.
  !! Repurposed slots were NOT dead: orig max|act| 0.17-0.80; L31 8697 RMS contrib 5.7x median w/ near-constant offset; L34 2178/9458 offsets 7-9x median.
- runs 69-70 (k64/k65): no truly silent neurons at L31/L34 (min max-contrib 0.06 / 0.24). Least-used slots: L31 [6816,2945,9370,3243], L34 [13606,9301,1628,2389] (slots_min.json).
- E3g = E3e circuit in least-used slots (run 71). pass 28 GEN 181/195 identical (all benign; different near-tie word swaps);
  pass 29 practice DI 255 but DD 121 (extra flip = item 9 'arrivò'->'è arrivato', tense part identical; judge noise). Fails pre-registered DD>=122 -> not adopted.
- FINAL = E3e (restored run 72; pass 30 HAB/COND reproduces COND 40/40, HAB 60/60, 97 identical). Budget at end: 30/50 passes, 72/300 runs.
