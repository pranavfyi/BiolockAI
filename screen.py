#!/usr/bin/env python
"""Screen a library of molecules: validate, filter, predict, rank.

Reads  a library CSV (molecule_id, smiles) + a saved model (.joblib)
Writes a ranked CSV with scores, ranks, and molecular properties.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from biolock.chem import standardize, properties, lipinski_violations, passes_lipinski, morgan_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def screen(
    library_path: Path,
    model_path: Path,
    output_path: Path,
    max_violations: int = 0,
    sep: str = ",",
    col_id: str = "molecule_id",
    col_smiles: str = "smiles",
) -> pd.DataFrame:
    """Screen a molecule library and return a ranked DataFrame."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    fp_radius = bundle["fp_radius"]
    fp_bits = bundle["fp_bits"]
    logger.info("Loaded model from %s (radius=%d, bits=%d)", model_path, fp_radius, fp_bits)

    df = pd.read_csv(library_path, sep=sep)
    logger.info("Loaded %d molecules from %s", len(df), library_path)

    if col_id not in df.columns:
        df[col_id] = [f"MOL_{i:06d}" for i in range(len(df))]
    df = df.rename(columns={col_id: "molecule_id", col_smiles: "smiles"})

    # Standardize
    df["std_smiles"] = df["smiles"].apply(standardize)
    n_invalid = df["std_smiles"].isna().sum()
    logger.info("Dropped %d invalid SMILES", n_invalid)
    df = df.dropna(subset=["std_smiles"]).copy()
    df["smiles"] = df["std_smiles"]
    df.drop(columns=["std_smiles"], inplace=True)

    if len(df) == 0:
        logger.error("No valid molecules remaining.")
        return pd.DataFrame()

    # Properties and filter
    prop_rows = df["smiles"].apply(properties).tolist()
    prop_df = pd.DataFrame(prop_rows)
    for col in prop_df.columns:
        df[col] = prop_df[col].values

    df["n_violations"] = df.apply(
        lambda r: lipinski_violations({k: r[k] for k in ["MW", "LogP", "HBD", "HBA"]}), axis=1
    )
    df["passes_filter"] = df.apply(
        lambda r: passes_lipinski({k: r[k] for k in ["MW", "LogP", "HBD", "HBA"]}, max_violations), axis=1
    )

    n_pass = df["passes_filter"].sum()
    logger.info("Lipinski filter: %d pass, %d fail", n_pass, len(df) - n_pass)

    # Fingerprints and prediction
    X = morgan_matrix(df["smiles"].tolist(), radius=fp_radius, n_bits=fp_bits)
    probas = model.predict_proba(X)[:, 1]

    # Filtered-out molecules get lowest score
    adjusted = probas.copy()
    adjusted[~df["passes_filter"].values] = -np.inf

    # Rank (1 = best)
    order = np.argsort(-adjusted)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)

    df["predicted_score"] = np.round(probas, 6)
    df["rank"] = ranks

    out_cols = [
        "molecule_id", "smiles", "predicted_score", "rank",
        "MW", "LogP", "HBD", "HBA", "RotBonds", "TPSA",
        "n_violations", "passes_filter",
    ]
    df = df[out_cols].sort_values("rank").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved ranked output (%d molecules) to %s", len(df), output_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen a molecule library with a trained model.",
    )
    parser.add_argument("library", type=Path, help="Input CSV with molecule_id and smiles columns.")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.joblib"),
                        help="Path to saved model bundle (.joblib).")
    parser.add_argument("--output", type=Path, default=Path("results/screened.csv"),
                        help="Output ranked CSV path.")
    parser.add_argument("--max-violations", type=int, default=0)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--col-id", default="molecule_id")
    parser.add_argument("--col-smiles", default="smiles")
    args = parser.parse_args()

    screen(
        library_path=args.library,
        model_path=args.model,
        output_path=args.output,
        max_violations=args.max_violations,
        sep=args.sep,
        col_id=args.col_id,
        col_smiles=args.col_smiles,
    )


if __name__ == "__main__":
    main()
