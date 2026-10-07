#!/usr/bin/env python
"""Train and evaluate ML models under three workflows (A, B1, B2)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from biolock.chem import morgan_matrix
from biolock.metrics import evaluate_ranking
from biolock.models import (
    make_models,
    max_tanimoto_scores,
    random_scores,
    score_baseline,
    score_workflow,
)
from biolock.split import scaffold_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

DEFAULT_INPUT = Path("data/prepared.csv")
DEFAULT_METRICS = Path("results/metrics.csv")
DEFAULT_MODEL_DIR = Path("models")


def train(
    input_path: Path = DEFAULT_INPUT,
    metrics_path: Path = DEFAULT_METRICS,
    model_dir: Path = DEFAULT_MODEL_DIR,
    n_seeds: int = 5,
    test_frac: float = 0.2,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    save_models: bool = True,
) -> pd.DataFrame:
    """Full training and evaluation pipeline. Returns metrics DataFrame."""
    df = pd.read_csv(input_path)
    logger.info("Loaded %d molecules (%d active, %d inactive)", len(df), (df["label"] == "active").sum(), (df["label"] == "inactive").sum())

    smiles = df["smiles"].tolist()
    y_all = (df["label"] == "active").astype(int).values
    passes_all = df["passes_filter"].values.astype(bool)

    logger.info("Computing Morgan fingerprints (radius=%d, bits=%d)...", fp_radius, fp_bits)
    X_all = morgan_matrix(smiles, radius=fp_radius, n_bits=fp_bits)

    all_rows: list[dict] = []
    seeds = list(range(n_seeds))
    skipped = 0

    for seed in seeds:
        logger.info("=== Seed %d ===", seed)
        train_idx, test_idx = scaffold_split(smiles, test_frac=test_frac, seed=seed)

        y_train_full = y_all[train_idx]
        y_test = y_all[test_idx]

        if y_train_full.sum() == 0 or y_test.sum() == 0:
            logger.warning("Seed %d: no actives in train or test, skipping", seed)
            skipped += 1
            continue

        X_train_full = X_all[train_idx]
        X_test = X_all[test_idx]
        test_passes = passes_all[test_idx]
        train_passes = passes_all[train_idx]

        X_train_A = X_train_full
        y_train_A = y_train_full

        b2_mask = train_passes
        X_train_B2 = X_train_full[b2_mask]
        y_train_B2 = y_train_full[b2_mask]

        if y_train_B2.sum() == 0:
            logger.warning("Seed %d: no actives in B2 training set, skipping B2", seed)

        rand_scores = random_scores(len(y_test), seed=seed)
        train_active_mask = y_train_full == 1
        tani_scores = max_tanimoto_scores(X_test, X_train_full[train_active_mask])

        for wf_name, X_tr, y_tr in [
            ("A", X_train_A, y_train_A),
            ("B1", X_train_A, y_train_A),
            ("B2", X_train_B2, y_train_B2),
        ]:
            if wf_name == "B2" and y_train_B2.sum() == 0:
                continue

            models = make_models(seed)
            for model_name, model in models.items():
                row = score_workflow(
                    wf_name,
                    model,
                    X_tr,
                    y_tr,
                    X_test,
                    y_test,
                    test_passes,
                    seed,
                    model_name,
                )
                all_rows.append(row)
                logger.info("  %s / %s: ROC-AUC=%.3f  PR-AUC=%.3f  EF@1%%=%.1f", wf_name, model_name, row["roc_auc"], row["pr_auc"], row["ef_1pct"])

            for bl_name, bl_scores in [("Random", rand_scores), ("MaxTanimoto", tani_scores)]:
                row = score_baseline(wf_name, bl_name, bl_scores, y_test, test_passes, seed, len(y_tr))
                all_rows.append(row)

        if save_models and seed == 0:
            model_dir.mkdir(parents=True, exist_ok=True)
            best_model = make_models(seed)["HistGBT"]
            best_model.fit(X_train_full, y_train_full)
            model_path = model_dir / "best_model.joblib"
            joblib.dump({"model": best_model, "fp_radius": fp_radius, "fp_bits": fp_bits}, model_path)
            logger.info("Saved model to %s", model_path)

    if skipped:
        logger.warning("Skipped %d/%d seeds (no actives in train or test)", skipped, len(seeds))

    results = pd.DataFrame(all_rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(metrics_path, index=False)
    logger.info("Saved %d result rows to %s", len(results), metrics_path)

    metric_cols = [
        "roc_auc",
        "pr_auc",
        "ef_1pct",
        "ef_5pct",
        "precision_at_1pct",
        "precision_at_5pct",
        "recall_at_1pct",
        "recall_at_5pct",
        "fit_time_s",
        "predict_time_s",
    ]
    summary = results.groupby(["workflow", "model"])[metric_cols].agg(["mean", "std"])
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY (mean ± std over seeds)")
    print("=" * 100)
    for (wf, mdl), row in summary.iterrows():
        print(f"\n  {wf} / {mdl}:")
        for col in metric_cols:
            m, s = row[(col, "mean")], row[(col, "std")]
            if pd.isna(s):
                s = 0.0
            print(f"    {col:>22s}: {m:.4f} ± {s:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models and evaluate ranking.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--no-save", action="store_true", help="Don't save model to disk.")
    args = parser.parse_args()

    train(
        input_path=args.input,
        metrics_path=args.metrics,
        model_dir=args.model_dir,
        n_seeds=args.seeds,
        test_frac=args.test_frac,
        fp_radius=args.fp_radius,
        fp_bits=args.fp_bits,
        save_models=not args.no_save,
    )


if __name__ == "__main__":
    main()
