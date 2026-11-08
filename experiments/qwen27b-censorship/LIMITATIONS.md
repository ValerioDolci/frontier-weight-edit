# Limitations

- **A seed, not "without knowing".** Everything starts from one topic (Taiwan) and from batteries we wrote. The map
  covers the basin of the sovereignty/official-script machinery; it does not claim to list *all* censored topics
  (Xinjiang and Falun Gong denial, for instance, are untouched by the 32 neurons).
- **Small numbers in the causal tests.** The held-out Taiwan half has 19 questions, the open questions 10, the
  Tiananmen held-out half 5–7. Exact tests are reported; effect sizes are large but intervals are wide.
- **LLM judges.** Labels come from Claude Sonnet. Agreement with a blind re-annotation is 92 % on PRC censorship but
  74 % on foreign-beneficiary cases (all disagreements: the judge says "censored", the reviewer "balanced", e.g. Catalonia,
  West Papua, Chechnya). Results that rest on foreign cases (C4) are the weakest. Second judge (Gemma): see FINDINGS C5.
- **"Chinese" questions embed the English entity name** (e.g. «关于the death penalty，…»): "language" here means the
  language of the template.
- **Two precisions.** Labels come from Q4_K_M answers, activations from bf16; C6 measures how many labels change.
  C7 (safety) and E11 (Pan & Xu) were run on Q4_K_M GGUFs of base / off32 / comb_t32 (llama.cpp), because the bf16
  hybrid path was too slow for ~2,000 generations in one night; rand64 and comb_t16 were not run there (deviation from
  the pre-registration: rand64 matched the base in every earlier measure; comb_t16 ⊂ comb_t32).
- **MMLU-Pro on 300 questions** has a ±5.5-point 95 % interval: "no cost" means "no drop larger than ~5 points".
- **Geometry.** Cosines between difference-of-means directions in the raw residual stream are dominated by a few
  high-variance components; we only use them together with a null. Earlier cosine-based claims were withdrawn (C4).
- **Side effects of the edit.** off32 makes the model deny the 2020 Hong Kong National Security Law and, in Chinese, Huawei's 2019 US indictment and Mengniu's role in the melamine scandal (E13); the larger edit
  that also removes the Tiananmen refusal (comb_t32) has a small but diffuse cost on generic text (C0) and, where the
  refusal goes away, what comes out is the official narrative, a denial or an invented event — and the unmodified model produces the same narrative and invention when its answer is pre-filled (E5): removing a refusal is not the same as recovering the facts.
- **Topic verdicts (E16) rest on one phrasing per topic.** Individual verdicts can flip with precision or wording
  (C6: ~8 % of labels; the base verdict agrees between bf16 and Q4 for 23 of 33 candidates); the counts (19 of 29 censored
  topics unlocked in bf16, 13 of 19 in Q4) are the robust figure, not any single row.
- **The probe finds the basin of the seed.** Topics on other machinery (Xinjiang camps, Tiananmen, Falun Gong, Mengniu)
  are found as censored but not unlocked; they need their own seeds.
- **Safety was measured on translated benchmarks** (StrongREJECT/XSTest in English and machine-translated Chinese, 40 Italian
  prompts), with one LLM judge and 64-token answers; J. Chang (X, 20 Oct 2026) reports a safety cost in Chinese for similar
  neurons in Qwen3-1.7B/8B, which we do not observe for the 32 neurons on the 27B (E18).
- **Not tested:** multi-turn, thinking mode, agentic use; other model sizes; other Chinese-origin models.
