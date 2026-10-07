#!/usr/bin/env python
"""Prepare raw ChEMBL data: validate, standardise, deduplicate, label,
compute molecular properties, and create the Lipinski filter flag.

Reads:
    data/raw.csv

Writes:
    data/prepared.csv
    data/prepare_log.json

The ChEMBL export is expected to contain:
    Smiles
    Standard Type
    Standard Relation
    Standard Value
    Standard Units

Only exact IC50 measurements reported in nM are retained.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from biolock.chem import (
    lipinski_violations,
    passes_lipinski,
    properties,
    standardize,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


DEFAULT_INPUT = Path("data/raw.csv")
DEFAULT_OUTPUT = Path("data/prepared.csv")
DEFAULT_LOG = Path("data/prepare_log.json")


def prepare(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    active_threshold: float = 1_000.0,
    inactive_threshold: float = 10_000.0,
    max_violations: int = 0,
    sep: str = ";",
    col_smiles: str = "smiles",
    col_ic50: str = "ic50_nM",
) -> pd.DataFrame:
    """Run the full preparation pipeline."""

    log: dict = {}

    # ============================================================
    # 1. LOAD
    # ============================================================

    df = pd.read_csv(input_path, sep=sep)

    logger.info("Loaded %d rows from %s", len(df), input_path)
    log["01_loaded"] = len(df)

    # ============================================================
    # 2. HANDLE CHemBL EXPORT
    # ============================================================

    # ChEMBL activity exports contain multiple activity types.
    # Keep only exact IC50 measurements reported in nM.

    if "Standard Type" in df.columns:

        df["Standard Type"] = (
            df["Standard Type"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df = df[
            df["Standard Type"] == "IC50"
        ].copy()

        logger.info(
            "After filtering Standard Type=IC50: %d rows",
            len(df),
        )

        log["02_ic50"] = len(df)

        # --------------------------------------------------------
        # Standard Relation
        # --------------------------------------------------------

        if "Standard Relation" in df.columns:

            relation = (
                df["Standard Relation"]
                .astype(str)
                .str.strip()
                .str.strip("'")
            )

            df = df[
                relation == "="
            ].copy()

            logger.info(
                "After filtering Standard Relation='=': %d rows",
                len(df),
            )

            log["03_exact_relation"] = len(df)

        # --------------------------------------------------------
        # Standard Units
        # --------------------------------------------------------

        if "Standard Units" in df.columns:

            units = (
                df["Standard Units"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            df = df[
                units == "nm"
            ].copy()

            logger.info(
                "After filtering Standard Units=nM: %d rows",
                len(df),
            )

            log["04_nM"] = len(df)

        # --------------------------------------------------------
        # Rename ChEMBL columns to internal project schema
        # --------------------------------------------------------

        rename_map = {
            "Smiles": "smiles",
            "Standard Value": "ic50_nM",
        }

        df = df.rename(columns=rename_map)

    # ============================================================
    # 3. VALIDATE REQUIRED COLUMNS
    # ============================================================

    if col_smiles not in df.columns or col_ic50 not in df.columns:

        raise ValueError(
            f"Expected columns '{col_smiles}' and '{col_ic50}' "
            f"in {input_path}. "
            f"Found: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            col_smiles: "smiles",
            col_ic50: "ic50_nM",
        }
    )

    # Convert IC50 to numeric

    df["ic50_nM"] = pd.to_numeric(
        df["ic50_nM"],
        errors="coerce",
    )

    # ============================================================
    # 4. DROP INVALID IC50 VALUES
    # ============================================================

    df = df.dropna(
        subset=[
            "smiles",
            "ic50_nM",
        ]
    )

    df = df[
        df["ic50_nM"] > 0
    ].copy()

    logger.info(
        "After dropping missing/non-positive IC50: %d",
        len(df),
    )

    log["05_valid_ic50"] = len(df)

    # ============================================================
    # 5. STANDARDISE SMILES
    # ============================================================

    logger.info("Standardising molecular structures...")

    df["std_smiles"] = df["smiles"].apply(
        standardize
    )

    n_invalid = int(
        df["std_smiles"].isna().sum()
    )

    df = df.dropna(
        subset=["std_smiles"]
    ).copy()

    logger.info(
        "Dropped %d invalid SMILES; remaining: %d",
        n_invalid,
        len(df),
    )

    log["06_standardised"] = len(df)
    log["06_invalid_smiles"] = n_invalid

    # ============================================================
    # 6. DEDUPLICATION
    # ============================================================

    grouped = (
        df.groupby("std_smiles")["ic50_nM"]
        .agg(
            [
                "median",
                "count",
            ]
        )
        .reset_index()
        .rename(
            columns={
                "std_smiles": "smiles",
                "median": "ic50_nM",
                "count": "n_measurements",
            }
        )
    )

    logger.info(
        "After deduplication (median IC50): %d unique molecules",
        len(grouped),
    )

    log["07_deduplicated"] = len(grouped)

    df = grouped

    # ============================================================
    # 7. LABEL ACTIVE / INACTIVE
    # ============================================================

    def _label(ic50: float) -> str | None:

        if ic50 <= active_threshold:
            return "active"

        if ic50 >= inactive_threshold:
            return "inactive"

        return None

    df["label"] = df["ic50_nM"].apply(
        _label
    )

    n_intermediate = int(
        df["label"].isna().sum()
    )

    df = df.dropna(
        subset=["label"]
    ).copy()

    n_active = int(
        (df["label"] == "active").sum()
    )

    n_inactive = int(
        (df["label"] == "inactive").sum()
    )

    logger.info(
        "Labelled: %d active (<= %.0f nM), "
        "%d inactive (>= %.0f nM), "
        "%d intermediate dropped",
        n_active,
        active_threshold,
        n_inactive,
        inactive_threshold,
        n_intermediate,
    )

    log["08_labelled"] = len(df)
    log["08_active"] = n_active
    log["08_inactive"] = n_inactive
    log["08_intermediate_dropped"] = n_intermediate

    # ============================================================
    # 8. DATASET SIZE CHECK
    # ============================================================

    min_class = 100

    if (
        n_active < min_class
        or n_inactive < min_class
    ):

        logger.warning(
            "INADEQUATE DATASET: need at least %d compounds "
            "in each class, but found %d active and %d inactive.",
            min_class,
            n_active,
            n_inactive,
        )

        log["08_inadequate_dataset"] = True
        log["08_min_class_size"] = min_class

    else:

        logger.info(
            "Dataset has sufficient class sizes."
        )

        log["08_inadequate_dataset"] = False
        log["08_min_class_size"] = min_class

    # ============================================================
    # 9. COMPUTE MOLECULAR PROPERTIES
    # ============================================================

    logger.info(
        "Computing molecular properties..."
    )

    prop_records = (
        df["smiles"]
        .apply(properties)
        .tolist()
    )

    prop_df = pd.DataFrame(
        prop_records
    )

    for col in prop_df.columns:
        df[col] = prop_df[col].values

    # ============================================================
    # 10. LIPINSKI FILTER
    # ============================================================

    df["n_violations"] = df.apply(
        lambda row: lipinski_violations(
            {
                "MW": row["MW"],
                "LogP": row["LogP"],
                "HBD": row["HBD"],
                "HBA": row["HBA"],
            }
        ),
        axis=1,
    )

    df["passes_filter"] = df.apply(
        lambda row: passes_lipinski(
            {
                "MW": row["MW"],
                "LogP": row["LogP"],
                "HBD": row["HBD"],
                "HBA": row["HBA"],
            },
            max_violations=max_violations,
        ),
        axis=1,
    )

    n_pass = int(
        df["passes_filter"].sum()
    )

    n_fail = int(
        len(df) - n_pass
    )

    actives_failing = int(
        (
            (df["label"] == "active")
            & (~df["passes_filter"])
        ).sum()
    )

    logger.info(
        "Lipinski filter (max_violations=%d): "
        "%d pass, %d fail, "
        "%d actives would be removed by filter",
        max_violations,
        n_pass,
        n_fail,
        actives_failing,
    )

    log["09_pass_filter"] = n_pass
    log["09_fail_filter"] = n_fail
    log["09_actives_failing_filter"] = actives_failing
    log["09_final_count"] = len(df)

    # ============================================================
    # 11. SELECT OUTPUT COLUMNS
    # ============================================================

    out_cols = [
        "smiles",
        "label",
        "ic50_nM",
        "n_measurements",
        "MW",
        "LogP",
        "HBD",
        "HBA",
        "RotBonds",
        "TPSA",
        "n_violations",
        "passes_filter",
    ]

    df = df[out_cols].copy()

    # ============================================================
    # 12. SAVE PREPARED DATA
    # ============================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved prepared data to %s (%d rows)",
        output_path,
        len(df),
    )

    # ============================================================
    # 13. SAVE LOG
    # ============================================================

    log["settings"] = {
        "active_threshold_nM": active_threshold,
        "inactive_threshold_nM": inactive_threshold,
        "max_violations": max_violations,
        "input_separator": sep,
        "activity_type": "IC50",
        "relation": "=",
        "units": "nM",
    }

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            log,
            f,
            indent=2,
        )

    logger.info(
        "Saved preparation log to %s",
        log_path,
    )

    return df


# ================================================================
# COMMAND LINE INTERFACE
# ================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Prepare raw ChEMBL data for ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to raw CSV (default: data/raw.csv).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for prepared CSV (default: data/prepared.csv).",
    )

    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="Path for JSON log (default: data/prepare_log.json).",
    )

    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1_000.0,
        help="IC50 <= this value -> active.",
    )

    parser.add_argument(
        "--inactive-threshold",
        type=float,
        default=10_000.0,
        help="IC50 >= this value -> inactive.",
    )

    parser.add_argument(
        "--max-violations",
        type=int,
        default=0,
        help="Maximum Lipinski violations allowed.",
    )

    parser.add_argument(
        "--sep",
        default=";",
        help="CSV separator (default: semicolon).",
    )

    parser.add_argument(
        "--col-smiles",
        default="smiles",
        help="Internal SMILES column name.",
    )

    parser.add_argument(
        "--col-ic50",
        default="ic50_nM",
        help="Internal IC50 column name.",
    )

    args = parser.parse_args()

    prepare(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        active_threshold=args.active_threshold,
        inactive_threshold=args.inactive_threshold,
        max_violations=args.max_violations,
        sep=args.sep,
        col_smiles=args.col_smiles,
        col_ic50=args.col_ic50,
    )


if __name__ == "__main__":
    main()