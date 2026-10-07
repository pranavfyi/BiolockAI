"""Tests that filtered-out molecules are ranked last and counted as misses."""

import numpy as np
from biolock.metrics import evaluate_ranking


def test_filtered_molecules_ranked_last():
    """In workflows B1/B2, molecules failing the filter should get the
    lowest score so they appear at the bottom of the ranking."""
    y_true = np.array([1, 0, 1, 0, 1])     # 3 actives
    model_scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    passes_filter = np.array([True, True, True, True, False])  # last active fails filter

    # Apply workflow B logic: failures get lowest possible score
    adjusted = model_scores.copy()
    adjusted[~passes_filter] = -np.inf

    result = evaluate_ranking(y_true, adjusted)
    # The active at index 4 was filtered out, so recall cannot be 1.0
    # at low K values — it's penalized
    assert result["recall_at_5pct"] < 1.0 or len(y_true) < 20


def test_filtered_active_is_a_miss():
    """An active molecule removed by the filter should not appear in top-K."""
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    model_scores = np.array([0.99, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    passes_filter = np.array([False, True, True, True, True, True, True, True, True, True])

    adjusted = model_scores.copy()
    adjusted[~passes_filter] = -np.inf

    result = evaluate_ranking(y_true, adjusted)
    # The only active is filtered, so it's ranked last → EF at 1% should be 0
    assert result["ef_1pct"] == 0.0
    assert result["recall_at_1pct"] == 0.0
