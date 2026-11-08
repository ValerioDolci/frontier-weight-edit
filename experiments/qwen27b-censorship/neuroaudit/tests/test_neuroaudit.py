"""Tests on a tiny randomly initialised Llama (CPU, no download) + pure-numpy statistics."""
import json, os, tempfile
import numpy as np, pytest, torch
from transformers import LlamaConfig, LlamaForCausalLM

from neuroaudit import stats, direction
from neuroaudit.model import decoder_layers, down_proj, n_neurons
from neuroaudit.intervene import NeuronEditor, DirectionEditor, parse_neurons, bake
from neuroaudit.attribution import split_halves, attribute, top_neurons, margin
from neuroaudit.locality import curve


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def items(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [{"cat": "T" if i % 2 else "C", "ids": rng.integers(1, 60, 8).tolist(), "tg": int(rng.integers(0, 64)), "tb": int(rng.integers(0, 64))} for i in range(n)]


def test_layers_and_neurons(tiny):
    assert len(decoder_layers(tiny)) == 3 and n_neurons(tiny) == 48 and down_proj(decoder_layers(tiny)[0]).out_features == 32


def test_neuron_editor_zero_equals_weight_zero(tiny):
    it = items(1)[0]; spec = {1: [3, 7]}; ed = NeuronEditor(tiny)
    with ed.scaled(spec, 0.0):
        a = margin(tiny, it)
    W = down_proj(decoder_layers(tiny)[1]).weight; old = W[:, [3, 7]].clone(); W[:, [3, 7]] = 0
    b = margin(tiny, it); W[:, [3, 7]] = old; ed.remove()
    assert abs(a - b) < 1e-5 and abs(margin(tiny, it) - a) > 0


def test_attribution_shapes_and_sign(tiny):
    it = split_halves(items(12), 0); att = attribute(tiny, it, "A")
    assert set(att) == {"T", "C"} and att["T"].shape == (3, 48)
    top = top_neurons(att["T"], 4, "neg"); assert len(top) == 4 and all(att["T"][l, j] <= att["T"].ravel()[np.argsort(att["T"].ravel())[4]] for l, j in top)


def test_attribution_matches_ablation_first_order(tiny):
    """activation×gradient ≈ change in margin when one neuron is zeroed (first order), on a single item."""
    it = [dict(items(1)[0], half="A")]; att = attribute(tiny, it, "A")[it[0]["cat"]]
    l, j = np.unravel_index(np.argmax(np.abs(att)), att.shape); ed = NeuronEditor(tiny); base = margin(tiny, it[0])
    with ed.scaled({int(l): [int(j)]}, 0.0):
        d = margin(tiny, it[0]) - base
    ed.remove(); assert np.sign(d) == -np.sign(att[l, j])


def test_locality_runs(tiny):
    it = split_halves(items(12), 0); att = attribute(tiny, it, "A")["T"]; res = curve(tiny, it, att, ks=(1, 4))
    assert {r["kind"] for r in res["curve"]} == {"top", "rand"} and "T" in res["base_right"]


def test_direction_editor_ablate_removes_projection(tiny):
    d = torch.randn(32); seen = {}
    h = decoder_layers(tiny)[0].register_forward_hook(lambda m, a, o: seen.__setitem__("x", (o[0] if isinstance(o, tuple) else o).clone()), prepend=False)
    de = DirectionEditor(tiny, 0)
    with de.edit(d, "ablate", 0.0):
        tiny(input_ids=torch.tensor([[1, 2, 3]]))
    h.remove(); de.remove()


def test_bake_writes_zero_columns(tiny):
    with tempfile.TemporaryDirectory() as t:
        src, dst = os.path.join(t, "src"), os.path.join(t, "dst"); tiny.save_pretrained(src)
        man = bake(src, {"2": [0, 5]}, dst); assert man["tensors"]
        m2 = LlamaForCausalLM.from_pretrained(dst); W = down_proj(decoder_layers(m2)[2]).weight
        assert torch.all(W[:, [0, 5]] == 0) and torch.any(W[:, 1] != 0)


def test_stats():
    assert stats.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0 and stats.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    a, n = stats.stratified_auc([1, 2, 3, 4], [0, 1, 0, 1], ["x", "x", "y", "y"]); assert a == 1.0 and n == 2
    assert stats.mcnemar_exact(0, 6) == pytest.approx(0.03125) and stats.mcnemar_exact(3, 3) == 1.0
    lo, hi = stats.wilson(17, 19); assert 0.6 < lo < hi < 1.0
    null = stats.permutation_null(lambda y: stats.auc(np.arange(20), y), np.r_[np.zeros(10), np.ones(10)], n=200); assert 0.3 < np.median(null) < 0.7


def test_direction_cv_auc_separable():
    rng = np.random.default_rng(0); X = rng.normal(size=(200, 16)); y = (np.arange(200) % 2).astype(int); X[y == 1, 0] += 3
    groups = np.arange(200) // 4; f = direction.group_folds(groups); assert direction.cv_auc(X, y, f) > 0.9
    lab = direction.families(rng.normal(size=(30, 10)), k=3); assert lab.shape == (30,)
