# How this was made

- **People.** Valerio Dolci chose the question (find in the weights what blocks answers the model knows, starting
  from Taiwan; then map the topics and the direction of the censorship, without producing an "uncensored" model as
  the goal), reviewed the results and decided what to publish.
- **Agent.** Claude Opus 5.5 (Anthropic), running as a coding agent on a local workstation, wrote the code, designed
  and ran the experiments, wrote the pre-registrations before running them, and wrote these documents. Sessions:
  6 November 2026 (Taiwan case, censorship map) and the night of 6–7 November (the check round C0–C8, E5, E9, E11).
- **Judges.** Answer types (censored / balanced, beneficiary) were labelled by Claude Sonnet with the prompt in
  `censura/judge_base.py`. Reliability checks: a second judge from another family (Gemma 4 31B-it, local), and a blind
  re-annotation of 150 answers by a separate Claude Opus instance that saw only questions and answers and a rule written
  beforehand (`censura/checks/data/c5_manual_*`). Refusals on harmful/benign prompts (C7) were judged by Claude Sonnet.
- **Reviews.** Key analyses were attacked by a fresh Claude Opus instance acting as adversarial reviewer; the C4
  correction (foreign censorship is *not* "the same circuit") came out of that loop.
- **Limits of this setup.** Battery questions and several test sets were written by Claude models; the judges are
  LLMs. See LIMITATIONS.md.
