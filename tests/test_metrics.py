"""Tests for ranking metrics with hand-computed examples."""

import numpy as np
import pytest
from biolock.metrics import (
    enrichment_factor, precision_at_k, recall_at_k, evaluate_ranking,
)


FIXTURE_Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # 2 actives out of 10
PERFECT_SCORES = np.array([1.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class TestEnrichmentFactor:
    def test_perfect_ranking(self):
        ef = enrichment_factor(FIXTURE_Y, PERFECT_SCORES, frac=0.1)
        # top 1 molecule is active (index 0, score 1.0)
        # ratio in top = 1/1 = 1.0, ratio overall = 2/10 = 0.2
        assert ef == pytest.approx(5.0)

    def test_no_actives(self):
        y = np.zeros(10, dtype=int)
        s = np.random.RandomState(0).uniform(size=10)
        assert enrichment_factor(y, s, 0.1) == 0.0


class TestPrecisionAtK:
    def test_perfect_top2(self):
        p = precision_at_k(FIXTURE_Y, PERFECT_SCORES, k=2)
        assert p == pytest.approx(1.0)  # both actives are top-2

    def test_top5(self):
        p = precision_at_k(FIXTURE_Y, PERFECT_SCORES, k=5)
        assert p == pytest.approx(0.4)  # 2 out of 5


class TestRecallAtK:
    def test_perfect_top2(self):
        r = recall_at_k(FIXTURE_Y, PERFECT_SCORES, k=2)
        assert r == pytest.approx(1.0)  # found both actives

    def test_top1(self):
        r = recall_at_k(FIXTURE_Y, PERFECT_SCORES, k=1)
        assert r == pytest.approx(0.5)  # found 1 of 2


class TestEvaluateRanking:
    def test_returns_all_keys(self):
        result = evaluate_ranking(FIXTURE_Y, PERFECT_SCORES)
        expected = {"roc_auc", "pr_auc", "ef_1pct", "ef_5pct",
                    "precision_at_1pct", "precision_at_5pct",
                    "recall_at_1pct", "recall_at_5pct"}
        assert set(result.keys()) == expected

    def test_roc_auc_perfect(self):
        result = evaluate_ranking(FIXTURE_Y, PERFECT_SCORES)
        assert result["roc_auc"] == pytest.approx(1.0)

    def test_ties_handled(self):
        y = np.array([1, 0, 1, 0])
        tied = np.array([0.5, 0.5, 0.5, 0.5])
        result = evaluate_ranking(y, tied, seed=42)
        assert 0 <= result["roc_auc"] <= 1
