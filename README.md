# BioLock.AI

Computational virtual-screening and molecule-prioritization pipeline for evaluating small-molecule activity prediction workflows.

> **Scope note:** The original specification considered a SARS-CoV-2 Mpro target. The required target could not be verified from official ChEMBL target metadata under the project's strict selection rules. To complete real-data validation without fabricating target or activity data, the final experiment uses a verified non-pathogen human ChEMBL target. No SARS-CoV-2 experimental results are claimed.

This project uses ChEMBL IC50 data, standardizes and deduplicates molecules, computes molecular properties, applies a Lipinski pass/fail filter, and compares ML models against simple baselines using scaffold splits.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Fetch ChEMBL data

```bash
# Select a verified SARS-CoV-2 Mpro SINGLE PROTEIN target from official ChEMBL dumps
# and download qualifying IC50 records (IC50, nM, '=', assay type B)
python fetch_chembl.py select --output data/raw.csv --log data/target_selection.json

# Search official dumps by free-text (does not use /target/search.json)
python fetch_chembl.py search "SARS-CoV-2 3C-like proteinase"
```

Target discovery uses the official ChEMBL FTP release files (`chembl_uniprot_mapping.txt` and the target FASTA). It does **not** guess target IDs, and it will refuse to download if no target satisfies the organism / SINGLE PROTEIN / Mpro-name rules. Assay descriptions that mention 3C-like protease are not treated as target metadata.

The download script saves `data/raw.csv` with columns:
- `molecule_id`
- `smiles`
- `ic50_nM`

### 3. Prepare the dataset

```bash
python prepare.py --input data/raw.csv --output data/prepared.csv --log data/prepare_log.json
```

This produces:
- `data/prepared.csv` — deduplicated, labeled, filtered dataset with descriptors
- `data/prepare_log.json` — per-step preparation counts

CLI defaults currently used by the project:
- `--active-threshold 1000.0`
- `--inactive-threshold 10000.0`
- `--max-violations 0`

### 4. Train and evaluate models

```bash
python train.py --input data/prepared.csv --metrics results/metrics.csv --seeds 5
```

This runs workflows A, B1, and B2 and compares ML models with simple baselines.

### 5. Screen a library

```bash
python screen.py path/to/library.csv --model models/best_model.joblib --output results/screened.csv
```

## Real-data experiment

The final real-data experiment uses a verified non-pathogen human protein target from ChEMBL.

Dataset preparation produced:

- 4,488 IC50 records
- 2,346 records with exact relation, nM units, and valid structures
- 1,821 unique molecules after deduplication
- 361 active molecules
- 914 inactive molecules
- 546 intermediate molecules excluded
- 1,275 molecules remaining after the final Lipinski filtering stage

### Best configuration

Workflow A with Random Forest was the strongest overall configuration:

| Metric | Mean ± SD |
|---|---:|
| ROC-AUC | 0.7809 ± 0.0355 |
| PR-AUC | 0.6059 ± 0.0883 |
| EF@1% | 3.6971 ± 1.0445 |
| EF@5% | 2.6387 ± 0.7526 |
| Fit time | 0.1613 ± 0.0174 s |

### Workflow comparison

| Workflow | Best model | ROC-AUC | PR-AUC | EF@1% |
|---|---|---:|---:|---:|
| A | Random Forest | **0.7809** | **0.6059** | **3.6971** |
| B1 | Random Forest | 0.6244 | 0.4896 | 2.9054 |
| B2 | Logistic Regression | 0.6034 | 0.4530 | 3.0120 |

The experiment therefore found that, for this dataset and experimental configuration, the filter-first workflows B1 and B2 did not improve predictive performance over Workflow A. Random Forest also provided a substantially better performance/compute trade-off than HistGradientBoosting.

These findings are specific to the evaluated dataset, preprocessing choices, models, and scaffold-split design. They are not a universal claim that property filtering is harmful.

## Research question

Does applying a cheap property filter before ML change the enrichment of actives among top-ranked molecules and the computational workload, compared with the same ML without the filter?

## Project structure

```text
biolock-ai/
├── biolock/
│   ├── chem.py
│   ├── metrics.py
│   ├── models.py
│   └── split.py
├── app/
├── data/
├── fetch_chembl.py
├── prepare.py
├── screen.py
├── train.py
├── tests/
├── README.md
├── requirements.txt
├── .gitignore
├── results/
├── models/
└── notebooks/
```

## Report Figures

Run:

```bash
python make_figures.py
```

This generates:

- `results/figures/roc_auc_comparison.png`
- `results/figures/pr_auc_comparison.png`
- `results/figures/ef_1pct_comparison.png`
- `results/figures/training_time_comparison.png`

The figures are generated directly from `results/metrics.csv`.

## Validation

The synthetic validation suite completed successfully:

```text
35 passed in 4.04s
```

The five-seed real-data experiment is stored in:

```text
results/metrics.csv
```

## Notes

- Output folders are created automatically by scripts that write files.
- Synthetic data is used only for software/workflow validation; final experimental reporting uses the verified real-data dataset.
- Output wording should use “predicted activity score” rather than drug/discovery wording.
