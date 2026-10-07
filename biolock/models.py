"""ML models, baselines, and workflow scoring."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from biolock.metrics import evaluate_ranking

logger = logging.getLogger(__name__)


def make_models(seed: int = 42) -> dict[str, Any]:
    """Return a dict of named sklearn classifiers."""
    return {
        "LogReg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        ),
        "RF": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGBT": HistGradientBoostingClassifier(
            max_iter=200,
            random_state=seed,
            early_stopping=False,
        ),
    }


def fit_predict_proba(
    model: Any,
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
) -> tuple[NDArray[np.float64], float, float]:
    """Fit a model and return probability scores for the given test molecules."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    probas = model.predict_proba(X_test)[:, 1]
    pred_time = time.perf_counter() - t0

    return probas, fit_time, pred_time


def max_tanimoto_scores(
    X_test: NDArray[np.uint8],
    X_train_actives: NDArray[np.uint8],
) -> NDArray[np.float64]:
    """For each test molecule, compute the maximum Tanimoto similarity to any active training molecule."""
    scores = np.zeros(X_test.shape[0], dtype=np.float64)
    if X_train_actives.shape[0] == 0:
        return scores

    X_t = X_test.astype(np.float64)
    X_a = X_train_actives.astype(np.float64)

    for i in range(X_t.shape[0]):
        intersection = np.minimum(X_t[i], X_a).sum(axis=1)
        union = np.maximum(X_t[i], X_a).sum(axis=1)
        tani = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        scores[i] = tani.max()
    return scores


def random_scores(n: int, seed: int = 42) -> NDArray[np.float64]:
    """Return uniform random scores in [0, 1]."""
    return np.random.RandomState(seed).uniform(0, 1, size=n)


def score_workflow(
    name: str,
    model: Any,
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    test_passes: np.ndarray | None = None,
    seed: int = 0,
    model_name: str = "model",
) -> dict[str, float | int | str]:
    """Score one workflow and return metrics plus runtime and counts."""
    if name in {"B1", "B2"}:
        mask = np.asarray(test_passes, dtype=bool) if test_passes is not None else np.ones(len(X_test), dtype=bool)
        scored_X = X_test[mask]
        if scored_X.size == 0:
            pred_probas = np.full(len(X_test), -1e12, dtype=np.float64)
            fit_t = 0.0
            pred_t = 0.0
            n_scored = 0
        else:
            pred_probas, fit_t, pred_t = fit_predict_proba(model, X_train, y_train, scored_X)
            full_scores = np.full(len(X_test), -1e12, dtype=np.float64)
            full_scores[mask] = pred_probas
            pred_probas = full_scores
            n_scored = int(mask.sum())
    else:
        pred_probas, fit_t, pred_t = fit_predict_proba(model, X_train, y_train, X_test)
        n_scored = len(X_test)

    if name in {"B1", "B2"} and test_passes is not None:
        pred_probas = pred_probas.copy()
        pred_probas[~np.asarray(test_passes, dtype=bool)] = -1e12

    metrics = evaluate_ranking(np.asarray(y_test, dtype=int), np.asarray(pred_probas, dtype=float), seed=seed)
    metrics.update(
        {
            "workflow": name,
            "model": model_name,
            "seed": seed,
            "fit_time_s": round(float(fit_t), 4),
            "predict_time_s": round(float(pred_t), 4),
            "predict_seconds": round(float(pred_t), 4),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_predicted": int(n_scored),
            "n_scored": int(n_scored),
            "test_actives_filtered": int(((np.asarray(y_test) == 1) & (~np.asarray(test_passes, dtype=bool))).sum()) if test_passes is not None else 0,
        }
    )
    return metrics


def score_baseline(
    name: str,
    baseline_name: str,
    scores: np.ndarray,
    y_test: np.ndarray,
    test_passes: np.ndarray | None,
    seed: int,
    n_train: int,
) -> dict[str, float | int | str]:
    """Evaluate a baseline score array for a workflow, keeping filtered-out molecules last."""
    adjusted = np.asarray(scores, dtype=float).copy()
    if name in {"B1", "B2"} and test_passes is not None:
        mask = np.asarray(test_passes, dtype=bool)
        adjusted[~mask] = -1e12

    metrics = evaluate_ranking(np.asarray(y_test, dtype=int), adjusted, seed=seed)
    metrics.update(
        {
            "workflow": name,
            "model": baseline_name,
            "seed": seed,
            "fit_time_s": 0.0,
            "predict_time_s": 0.0,
            "predict_seconds": 0.0,
            "n_train": n_train,
            "n_test": len(y_test),
            "n_predicted": int(np.asarray(test_passes, dtype=bool).sum()) if test_passes is not None else len(y_test),
            "n_scored": int(np.asarray(test_passes, dtype=bool).sum()) if test_passes is not None else len(y_test),
            "test_actives_filtered": int(((np.asarray(y_test) == 1) & (~np.asarray(test_passes, dtype=bool))).sum()) if test_passes is not None else 0,
        }
    )
    return metrics
