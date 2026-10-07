import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
df = pd.read_csv("results/metrics.csv")

# Output directory
out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)

# Only ML models for the main workflow comparison
ml_models = ["LogReg", "RF", "HistGBT"]
workflows = ["A", "B1", "B2"]

ml = df[
    df["workflow"].isin(workflows) &
    df["model"].isin(ml_models)
].copy()

summary = (
    ml.groupby(["workflow", "model"])
    [["roc_auc", "pr_auc", "ef_1pct", "ef_5pct", "fit_time_s"]]
    .agg(["mean", "std"])
)

def grouped_bar(metric, ylabel, filename, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(workflows))
    width = 0.24

    for i, model in enumerate(ml_models):
        means = [
            summary.loc[(w, model), (metric, "mean")]
            for w in workflows
        ]
        stds = [
            summary.loc[(w, model), (metric, "std")]
            for w in workflows
        ]

        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=model
        )

    ax.set_xlabel("Workflow")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(workflows)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created: {out / filename}")


# 1. ROC-AUC
grouped_bar(
    "roc_auc",
    "ROC-AUC",
    "roc_auc_comparison.png",
    "ROC-AUC Comparison Across Workflows"
)

# 2. PR-AUC
grouped_bar(
    "pr_auc",
    "PR-AUC",
    "pr_auc_comparison.png",
    "PR-AUC Comparison Across Workflows"
)

# 3. Enrichment Factor @ 1%
grouped_bar(
    "ef_1pct",
    "Enrichment Factor @ 1%",
    "ef_1pct_comparison.png",
    "Enrichment Factor @ 1% Across Workflows"
)

# 4. Training time
grouped_bar(
    "fit_time_s",
    "Training Time (seconds)",
    "training_time_comparison.png",
    "Training Time Comparison Across Workflows"
)

print("\nAll figures generated successfully.")