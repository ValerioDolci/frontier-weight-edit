"""neuroaudit — neuron-level auditing of a language model's decisions.

Start from ONE behaviour you already know (a seed: e.g. a scripted answer), find the MLP neurons that carry it,
check that they are causal, then use them as a probe over a neutral battery of prompts to surface other behaviours
that share the same machinery, and verify each finding.

Modules
    model       load a Hugging Face causal LM (optionally split CPU/GPU), locate decoder layers and MLP down projections
    intervene   runtime neuron ablation / scaling, residual-direction edits, writing a neuron edit into the weights
    attribution activation × gradient attribution of a binary decision (two-token margin) to every MLP neuron
    locality    causal check: ablate the top-k neurons on held-out items, with a matched random control
    discover    use a neuron set as a probe over a neutral battery (activation and ablation-effect scores)
    capture     last-prompt-token activations of all MLP neurons and the residual stream, for many prompts
    direction   difference-of-means directions, cross-fitted AUC, neuron write scores, neuron-only probe scores, families
    judge       batched LLM judge (answer type, refusal, correctness against a reference) with a resumable cache
    stats       AUC, stratified AUC, exact McNemar, Wilson interval, permutation nulls, Cohen's d
"""
__version__ = "0.1.0"
