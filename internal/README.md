# The internal pipeline that produced the numbers

These are the scripts we actually ran, with private paths replaced by placeholders (`<server>`, `<ssh-to-gpu-server>`,
`<home>`, ...). They assume our two-machine setup — a Mac running the agent, the judge (`claude -p`, logged in with a
Claude subscription token) and orchestration, and a GPU server reached over ssh with a GPU lock — so they are here for
transparency, not to be run as they are. The portable equivalents are in `../scripts`, `../harness` and `../baselines`,
and `scripts/evaluate.py` reproduces the published numbers: re-running the original model and e50c with it gives the
same COMET, the same 200/200 FLORES translations, 545-547 of 549 identical test translations (batched greedy decoding
changes slightly when the test files change) and the same per-category counts.

| file | role |
|---|---|
| `run_editor.sh`, `wsp_session.sh`, `wsp_broker.py`, `wsp_cli.py` | one editor session: servers on the GPU box, broker + judge on the Mac, Claude Code headless |
| `judge_tense.py` | the tense judge (rubrics and prompt identical to `scripts/judge.py`), called through the `claude` CLI |
| `translate_set.py` | step 0: translations of the first probe sets through our llama.cpp endpoint (Gemma 4 31B) |
| `hf_translate.py`, `eval_edit.py`, `eval_edit_ws.sh`, `comet_score.py` | evaluation of an arm: translate test + FLORES on the GPU box, COMET, judge |
| `rescore.py` | re-score existing translations on the corrected test labels (no new generation) |
| `relabel.py` | blind re-classification of the test items (found the v3 label errors) |
| `quality_pairwise.py` | blind pairwise overall-quality check on every changed translation |
| `gen_practice.py`, `gen_test_v3.py`, `gen_pairs.py` | data generation (practice by Sonnet, test v3 by Opus, minimal pairs for B1/B2) |
| `b2_select.py` | configuration choice for the first LoRA baseline |
| `decision_neurons.py`, `few_pairs.py` | follow-up: gradient attribution on contrast pairs + held-out ablation (found layer 33 neuron 1640); how many pairs it needs |
| `scale_cols.py`, `ablate_one.py`, `make_gain_edit.py` | follow-up: the gain on one neuron (and the switched-off arms) as full tensors; the same edit as a public diff |
| `e1_q4_eval.py`, `e1_q4_ws.sh`, `e1_gguf_diff.py`, `rescore_q4.py` | follow-up: the published Q4_K_M file against a matched Q4 original, served with llama.cpp |
