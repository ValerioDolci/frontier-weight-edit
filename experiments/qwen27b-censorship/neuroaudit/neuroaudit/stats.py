"""Small, dependency-free statistics used throughout (numpy only)."""
from __future__ import annotations
import math
import numpy as np


def auc(scores, labels) -> float:
    """ROC AUC with ties handled by average rank; labels in {0,1}."""
    s = np.asarray(scores, float); y = np.asarray(labels).astype(bool); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):           # average ranks for ties
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_auc(scores, labels, strata) -> tuple[float, int]:
    """AUC computed only over (positive, negative) pairs that share a stratum; returns (auc, number of pairs)."""
    s = np.asarray(scores, float); y = np.asarray(labels).astype(bool); st = np.asarray(strata); num = den = 0.0
    for g in np.unique(st):
        m = st == g; p = s[m & y]; n = s[m & ~y]
        if len(p) and len(n):
            d = p[:, None] - n[None]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return (num / den if den else float("nan")), int(den)


def permutation_null(stat, labels, strata=None, n: int = 1000, seed: int = 0) -> np.ndarray:
    """Null distribution of stat(labels) under label permutation (within strata if given)."""
    rng = np.random.default_rng(seed); y = np.asarray(labels); out = []
    for _ in range(n):
        yp = y.copy()
        if strata is None:
            rng.shuffle(yp)
        else:
            st = np.asarray(strata)
            for g in np.unique(st):
                idx = np.where(st == g)[0]; yp[idx] = rng.permutation(yp[idx])
        out.append(stat(yp))
    return np.array(out)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cohen_d(a, b) -> np.ndarray:
    """Per-column Cohen's d between two sample matrices [n_a, k] and [n_b, k]."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2)) + 1e-9
    return (a.mean(0) - b.mean(0)) / sp
