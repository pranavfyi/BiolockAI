"""Ranking evaluation metrics: EF, Precision@K, Recall@K, ROC-AUC, PR-AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _break_ties(scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return ranks (0-indexed, lower = better) with random tie-breaking."""
    rng = np.random.RandomState(seed)
    noise = rng.uniform(0, 1e-12, size=len(scores))
    return np.argsort(-(scores + noise))


def enrichment_factor(y_true: np.ndarray, scores: np.ndarray,
                      frac: float, seed: int = 0) -> float:
    """Enrichment factor at a given fraction of the ranked list."""
    order = _break_ties(scores, seed)
    n = len(y_true)
    k = max(1, int(n * frac))
    top_k = y_true[order[:k]]
    ratio_in_top = top_k.sum() / k
    ratio_overall = y_true.sum() / n
    if ratio_overall == 0:
        return 0.0
    return float(ratio_in_top / ratio_overall)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray,
                   k: int, seed: int = 0) -> float:
    """Precision among the top-K ranked molecules."""
    order = _break_ties(scores, seed)
    k = min(k, len(y_true))
    return float(y_true[order[:k]].sum() / k)


def recall_at_k(y_true: np.ndarray, scores: np.ndarray,
                k: int, seed: int = 0) -> float:
    """Recall among the top-K ranked molecules."""
    order = _break_ties(scores, seed)
    k = min(k, len(y_true))
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    return float(y_true[order[:k]].sum() / total_pos)


def evaluate_ranking(
    y_true: np.ndarray,
    scores: np.ndarray,
    seed: int = 0,
) -> dict[str, float]:
    """Compute all ranking metrics. Returns a flat dict."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n = len(y)

    try:
        roc = roc_auc_score(y, s)
    except ValueError:
        roc = float('nan')
    try:
        pr = average_precision_score(y, s)
    except ValueError:
        pr = float('nan')

    k1 = max(1, int(n * 0.01))
    k5 = max(1, int(n * 0.05))

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "ef_1pct": enrichment_factor(y, s, 0.01, seed),
        "ef_5pct": enrichment_factor(y, s, 0.05, seed),
        "precision_at_1pct": precision_at_k(y, s, k1, seed),
        "precision_at_5pct": precision_at_k(y, s, k5, seed),
        "recall_at_1pct": recall_at_k(y, s, k1, seed),
        "recall_at_5pct": recall_at_k(y, s, k5, seed),
    }
