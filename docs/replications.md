# Replications and comparisons

All numbers on the corrected test sets (v2: 78 DI, v3: 103 DI + 35 DI in paragraphs), same translations for every
scoring step. Paired counts are **shifted / reverted** vs the original model (on DI: moved to / away from the preferred imperfetto;
on controls, "broken" = moved into a wrong reading); p = exact McNemar on DI. Per-arm files with
every translation and verdict: `results/arms/`.

## Three editor sessions

Same task, same budget (50 forward passes, 300 kernel runs), a fresh Claude Code instance each time
(`claude-opus-5-5`, effort max) that saw neither the other sessions nor any test data.

| session | edit | passes | DI v2 | DI v3 | paragraphs | controls broken | FLORES unchanged | COMET |
|---|---|---|---|---|---|---|---|---|
| **e50** | 2+2 neurons, layers 30 / 40 | 27 | 12 % · +11 / −0 · p = 0.001 | 23 % · +9 / −0 · p = 0.004 | +1 / −0 | 1 | 177 | 0.8945 |
| e50b | 1+1 neurons, layers 29 / 47 | 14 (crashed) | 27 % · +0 / −1 | 32 % · +0 / −0 | +1 / −0 | 0 | 181 | 0.8944 |
| **e50c** | 4+4 neurons, layers 31 / 34 | 30 | 5 % · +16 / −0 · p < 0.0001 | 16 % · +17 / −0 · p < 0.0001 | +3 / −1 | 2 | 182 | 0.8945 |

Original model: DI v2 26 %, DI v3 32 %, 12 / 35 paragraph errors, COMET 0.8945.

**e50b** crashed because of the harness: its kernel grew to 83 GB of RAM and the OS killed the whole session after 14
passes; the editor then submitted a design it had not been able to test, which does nothing (`runs/e50b/README.md`). We
added a RAM cap to the kernel before e50c. So: three sessions, two successes, one infrastructure failure. The two
successful sessions converged independently on the same mechanism (MLP neurons repurposed as a saturating detector of
the DI context that pushes toward the imperfetto).

## Against the baselines

| | DI v2 | DI v3 | paragraphs | controls broken | FLORES unchanged | FLORES better / worse | COMET |
|---|---|---|---|---|---|---|---|
| B0 prompt rule | 4 % · +18 / −1 | 2 % · +30 / −0 | +9 / −0 | 49 | 82 | 7 / 22 | 0.8911 |
| B1 difference of means (naive) | 49 % · +5 / −24 | 75 % · +3 / −47 | +3 / −11 | 53 | 43 | 18 / 26 | 0.8896 |
| **B2-min LoRA r16** (selected) | 12 % · +12 / −1 | 23 % · +10 / −1 | +6 / −1 | 0 | 120 | 5 / 5 | 0.8947 |
| B2-rewrite LoRA r8 (selected) | 13 % · +10 / −0 | 29 % · +7 / −4 | +5 / −1 | 4 | 77 | 7 / 14 | 0.8935 |
| B2-rewrite LoRA r16 | 13 % · +13 / −3 | 20 % · +17 / −5 | +5 / −1 | 3 | 49 | 13 / 23 | 0.8937 |
| e50 | 12 % · +11 / −0 | 23 % · +9 / −0 | +1 / −0 | 1 | 177 | 2 / 2 | 0.8945 |
| e50c | 5 % · +16 / −0 | 16 % · +17 / −0 | +3 / −1 | 2 | 182 | 1 / 0 | 0.8945 |

(B0 breaks the finished-period cases: DD 26 / 39 and PERF 9 / 25 errors on v2.)

**Overall quality of the changed translations** (blind pairwise, Claude Sonnet 5.5; edited better / original better /
tie):

| | changed | DI sentences | controls | FLORES |
|---|---|---|---|---|
| e50 | 80 | 26 / 17 / 3 | 1 / 5 / 5 | 11 / 6 / 6 |
| e50c | 117 | 45 / 11 / 24 | 8 / 7 / 4 | 10 / 5 / 3 |
| B2-min LoRA | 242 | 46 / 30 / 18 | 30 / 21 / 17 | 29 / 32 / 19 |
| B2-rewrite r8 | 418 | 69 / 31 / 73 | 52 / 35 / 35 | 43 / 45 / 35 |
| B2-rewrite r16 | 500 | 89 / 48 / 58 | 70 / 43 / 41 | 65 / 46 / 40 |

Reading:

- On the target, e50c beats every baseline that does not break the controls; e50 matches the fair LoRA.
- The edits are local: 177-182 of 200 general translations are untouched, against 120 for the fair LoRA (and 49-77 for
  the LoRA with rewritten targets). Where they do change a general sentence, it is about as often better as worse.
- LoRA is better inside paragraphs; the hand edits barely act there.
- The tense judge over-counts a little: of e50c's 36 judged fixes, 33 are correct Italian when read by hand.
- e50 sometimes fixes the tense but makes the sentence worse elsewhere (17 of 46 changed DI sentences are judged worse
  overall); e50c much less (11 of 80).

## Reproducibility check

Re-running the original model and e50c with `scripts/evaluate.py` gives the same COMET (0.8945), the same 200 FLORES
translations, 545-547 of 549 identical test translations (batch composition changed when test items were removed) and
the same per-category counts (e50c: v2 DI 4 / 78, +16 / −0; v3 DI 16 / 103, +17 / −0; paragraphs +3 / −1).
