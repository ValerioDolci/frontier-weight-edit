# B0 — telling the model the rule in the system prompt

Same frozen system prompt, with this sentence appended (`scripts/translate.py --system-extra "..."`):

> Grammar note: when English uses the past perfect (simple or continuous) with for or since for a state or activity that was still going on at a past moment (for example interrupted by a when-clause), Italian uses the imperfetto with da (Aspettavamo da due ore quando arrivò il treno), not the trapassato with per.

Result (`results/arms/B0_prompt.json`): test v2 DI errors 20/78 → 3/78, but DD 3/39 → 26/39 and PERF 0/25 → 9/25;
test v3 DI 33/103 → 2/103 with DD 1/30 → 12/30; COMET FLORES 0.8945 → 0.8911. The model applies the rule wherever it
sees a past perfect, finished or not. (We did not try a version of the rule that also states the exception.)
