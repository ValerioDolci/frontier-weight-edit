"""From labelled answers to directions and neurons.

labels: 1 = the behaviour (e.g. censored answer), 0 = the reference (e.g. balanced answer); groups: an id per item
(e.g. the entity asked about) so that validation never trains and tests on the same group."""
from __future__ import annotations
import numpy as np
from .stats import auc


def group_folds(groups, seed: int = 0) -> np.ndarray:
    g = np.asarray(groups); u = sorted(set(g.tolist())); rng = np.random.default_rng(seed); rng.shuffle(u); half = set(u[: len(u) // 2])
    return np.array([0 if x in half else 1 for x in g])


def diff_means(X, y) -> np.ndarray:
    y = np.asarray(y).astype(bool); return X[y].mean(0) - X[~y].mean(0)


def cv_auc(X, y, folds, mask=None, project_out=None) -> float:
    """AUC of the difference-of-means direction trained on one fold of groups and tested on the other (mean of the two)."""
    y = np.asarray(y).astype(int); m = np.ones(len(y), bool) if mask is None else np.asarray(mask); a = []
    for f in (0, 1):
        tr, te = m & (folds != f), m & (folds == f)
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2 or y[te].sum() < 1 or (1 - y[te]).sum() < 1:
            continue
        d = diff_means(X[tr], y[tr])
        if project_out is not None:
            p = project_out / np.linalg.norm(project_out); d = d - p * (p @ d)
        a.append(auc(X[te] @ d, y[te]))
    return float(np.mean(a)) if a else float("nan")


def layer_scan(resid, y, folds) -> list[dict]:
    """resid: [n, n_layers, hidden] (memmap ok). Cross-fitted AUC per layer."""
    return [{"layer": L, "auc_cv": cv_auc(resid[:, L].astype(np.float32), y, folds)} for L in range(resid.shape[1])]


def write_scores(down_weights: list[np.ndarray], direction: np.ndarray) -> np.ndarray:
    """For each layer's down-projection weight W [hidden, n_neurons], the score w = d̂ᵀW: how much each neuron's output
    column writes along the direction. Returns [n_layers, n_neurons]."""
    d = direction / np.linalg.norm(direction); return np.stack([d @ W for W in down_weights])


def neuron_contrast(acts, W, y, train_mask) -> np.ndarray:
    """Mean contribution a·w (censored − reference) on the training rows, per neuron: the 'neurons that write the behaviour'."""
    y = np.asarray(y).astype(bool); out = np.zeros(W.shape, np.float64)
    for l in range(W.shape[0]):
        C = acts[:, l].astype(np.float32) * W[l][None]
        out[l] = C[train_mask & y].mean(0) - C[train_mask & ~y].mean(0)
    return out


def neuron_probe(acts, W, neurons: list[tuple[int, int]]) -> np.ndarray:
    """Score every item from the chosen neurons only: Σ a·w. No answer is needed — only a forward pass on the prompt."""
    S = np.zeros(acts.shape[0], np.float32)
    for l, j in neurons:
        S += acts[:, l, j].astype(np.float32) * W[l, j]
    return S


def families(contrib_profiles: np.ndarray, k: int = 8, seed: int = 0) -> np.ndarray:
    """k-means (numpy) on neuron contribution profiles [n_neurons, n_items] → family id per neuron."""
    X = contrib_profiles / (np.linalg.norm(contrib_profiles, axis=1, keepdims=True) + 1e-9); rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(100):
        lab = np.argmax(X @ C.T, 1); newC = np.stack([X[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
        newC /= np.linalg.norm(newC, axis=1, keepdims=True) + 1e-9
        if np.allclose(newC, C):
            break
        C = newC
    return lab
