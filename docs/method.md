# Method

## Starting point

MIT CSAIL's *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, 2 Nov 2026) had GPT-6 Astra edit the
weights of Qwen3.5-0.8B by hand; MATH-500 went from 7.9 % to 48.1 % while the share of answers cut at the token limit
fell from 93 % to 33 %. The post itself concludes that the edit "could remove an obstacle that hid an ability, but could
not add an ability". Their published logs make this concrete: in Astra's own final confirmation of the 50-pass run
(1,152 held-out *practice* problems, `exp8_math/runs/p50c` in [their log repository](https://github.com/RikiyaT/lm_understand-logs)),
accuracy on the problems the original model already finished is 93.9 % before and after the edit, and all the gain
(1.4 % → 56.7 %) is on the problems where it looped.

We wanted the same experiment on something we actually use: not removing a failure, but shifting how a model we use
translates one construction. The question is the same — can a frontier model, reading weights and activations, change a
specific behaviour by editing weights directly — on a narrow target chosen by us.

## Target

Gemma 4 translates Italian tenses well (we first probed Gemma 4 31B on 255 sentences: tense, capitalisation, titles,
punctuation, all fine). What it does systematically is render the English past perfect with *for/since*, for a
situation still going on at a past moment (**DI**), with the trapassato + *per*: «Si era nascosto in soffitta per mesi
quando…». That is correct Italian. The imperfetto + *da* («Si nascondeva in soffitta da mesi quando…») is equally
correct and is the form the author, a native speaker, prefers in this context, because it states that the situation was
still going on, as the English does. Gemma 4 12B uses the trapassato in about a quarter of these sentences, Gemma 4 31B
in about a fifth.

So the task is a **targeted preference shift**, not an error fix: move DI to the imperfetto, and leave the trapassato
where the period was already over (**DD**), where the imperfetto would change the meaning. Our earlier documents and
the judge rubric call the trapassato form on DI an "error"; read it as "non-preferred form".

## Data

| file | items | written by |
|---|---|---|
| `data/practice.tsv` | 610, visible to the editor (DI 260, DD 130, PERF 60, DURP 60, HAB 60, COND 40) | Claude Sonnet 5.5 |
| `data/test_v2.tsv` | 257 (DI 78, DD 39, PERF 25, DURP 35, HAB 50, COND 30) | written for this project with Claude Opus 5.5, before any edit existed |
| `data/test_v3.tsv` | 292, mixed styles, incl. 2-3 sentence paragraphs (DI 103, PDI 35, DD 30, PDD 10, controls 26-30 each) | Claude Opus 5.5, a separate generation made after the first edit had been evaluated on v2 (it never saw any edit's output) |
| `data/flores200_en_it_200.jsonl` | 200 FLORES-200 devtest sentences, general-domain quality guard (COMET) | FLORES-200, CC BY-SA 4.0 |

Category meaning: **DI** ongoing past perfect (target), **DD** finished past perfect, **PERF** finished simple past /
present perfect, **DURP** present perfect for an ongoing state (presente + da), **HAB** habitual *would* (imperfetto),
**COND** conditional *would* (condizionale); **PDI/PDD** the same inside a paragraph.

Overlap: word-set Jaccard between practice and test items is at most 0.60 for v2 (33 v2 items are ≥ 0.5, so v2 is
close to the practice distribution) and below 0.5 for v3.

**Corrections made after review** (`data/test_changes.json`, originals in `data/original/`): two v2 sentences also
appear, almost verbatim, as worked examples in the editor's task prompt and were removed; a blind re-classification of
every test item by Claude Opus 5.5 led us to relabel 21 v3 items (mostly "DD" items that describe a situation still going
on) and drop 9 ambiguous items (8 in v3, 1 in v2). Numbers were recomputed on the same translations.

## Grading

- Translation: greedy, thinking off, frozen system prompt (`scripts/common.py: SYSTEM`); test sets in batches of 65 with
  at most 256 new tokens, FLORES in batches of 50 with at most 400.
- **Tense judge** (`scripts/judge.py`, Claude Sonnet 5.5): checks only the tense/aspect rule of the item's category and
  returns ok / err / other. Each (category, source, translation) triple is judged once and cached, so an unchanged output
  always gets the same verdict. Our verdicts were produced through the `claude` CLI (`JUDGE_BACKEND=cli`) and are shipped
  in `results/judge_cache_claude-sonnet-5-5.json`.
- **Second judge**: Claude Opus 5.5 re-graded the original model, B0, B1 and e50 on test v2 (before the corrections):
  97.5-100 % agreement per category, identical DI verdicts on 80/80 items for the original model and for e50.
- **Overall quality** (`internal/quality_pairwise.py`): every translation an arm changes (test and FLORES) is compared
  blind with the original by Claude Sonnet 5.5 (grammar, naturalness, faithfulness; A/B order randomised). And we read
  every DI item e50c's judge counted as fixed (33 of 36 are correct Italian).
- **General quality**: COMET (Unbabel/wmt22-comet-da) and identical / better / worse (> 0.02 COMET) FLORES segments.
- Paired statistics: fixed / broken items and an exact McNemar test, always against the same original-model outputs.

## The editor and its rules

`harness/editor_prompt.md` is the exact task text (the agent saw nothing else about the experiment). It got:

- **Kernel** (`harness/kernel.py`): persistent Python on CPU over the original weights; `W(name)`, `save_edit(name, t)`,
  `drop_edit`, `load_acts`; at most 300 runs. Edits to embeddings, final norm, output head, `layer_scalar`, or a whole
  tensor times a constant are refused. Autograd entry points, `torch.func` transforms and `torch.optim` are replaced by
  functions that raise. From e50c on, the kernel caps its own RAM (56 GB).
- **Forward tool** (`harness/fwd_server.py`), 50 passes: *generate* (up to 512 practice or free sentences, original /
  edited / both, judged automatically) or *read* (teacher-forced pass over up to 1,024 texts the editor writes, returning
  log-probs and saving residual / attention / MLP activations).
- **Rules stated in the prompt**: no gradient descent or training, no running the model in the kernel, no internet,
  nothing outside its folder and the kernel's work folder, no command other than `./wsp`.

What was **technically enforced** and what was only **stated**: the agent was a headless Claude Code instance in its
own folder outside the project, with no inherited settings or memory files, Bash allowed for `./wsp` (Claude Code also
auto-allows read-only commands such as `ls`/`head` inside the working folder, which e50 used), file reads denied outside
its folder, web tools disabled. The gradient ban and the "no model in the kernel" rule are patches inside the same
Python process, so a determined agent could bypass them; and the kernel could read files on the GPU server. We therefore
audited every session afterwards (`harness/audit_session.py`, plus reading the logs): no session used gradients,
optimisers or the model in the kernel, or read anything outside its folder.

One thing to disclose: in e50c the kernel ran out of memory under the 56 GB cap (allocator fragmentation). The agent
inspected the kernel's own globals to understand the limit, freed memory with `malloc_trim` via `ctypes`, and once
restarted the kernel process with `os.execv` (same interpreter and arguments; the run counter and the edits persisted).
It reported this itself in `runs/e50c/final_answer.md`. None of it touched the rules about gradients or the model.

During the sessions the test files were kept off the machines the agent could reach; evaluations ran on the GPU server
only between sessions.

## Baselines

- **B0, prompt**: the rule written into the system prompt (`baselines/b0_prompt.md`).
- **B1, difference of means** (`baselines/b1_diffmeans.py`): the standard no-frontier recipe — contrast residuals where
  the model picks the calque vs the correct form, remove that direction from the attention/MLP output matrices that
  write it most (corrected for Gemma's post-norm), choose layer / number of matrices / strength on half of the practice
  pairs with a fixed rule. Our sweep used strengths 2 and 4, which over-subtract the direction; this is a naive version,
  not a tuned one.
- **B2, LoRA** (`baselines/b2_lora.py`): adapters on all attention and MLP projections, trained on the editor's practice
  set (80 %), configuration chosen on the other 20 % with a rule fixed in advance. **B2-min** (the one in the README):
  targets are Gemma's own translations, where wrong rewritten by Claude Sonnet 5.5 changing only the tense
  (`baselines/b2_min_targets.py`); rule: fewest held-out errors (DI + controls), tie → smaller adapter → r = 16.
  **B2-rewrite** (reported for completeness): DI/DD targets are the correct member of Sonnet-written minimal pairs, so
  the wording also changes; that version drifted much more on general text.

## Note on the recorded sessions

The sessions in `runs/` were recorded with the same harness code, whose status and error messages were then in Italian
(e.g. `edit salvato` = *edit saved*, `VIETATO` = *forbidden*, `pronto` = *ready*); those strings appear verbatim in the
logs. For this release the messages were translated to English; the logic, limits and rules are unchanged except for the
RAM cap added after e50b. Private paths in the logs were replaced by placeholders (`<server>`, `<agent-dir>`, ...).
