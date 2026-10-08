from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
FIGURES = {
    "classification": {
        "datasets": ["adult_income", "bank_marketing", "cmc", "diabetes_pima", "satimage"],
        "metric_column": "validation_macro_f1",
        "metric_label": "Macro-F1",
        "maximize": True,
        "prefix": "rankup_cls_",
        "title": "Optuna convergence · Classification (validation Macro-F1)",
    },
    "ordinal_regression": {
        "datasets": ["melbourne_airbnb", "student_performance", "nhanes", "esl", "lev"],
        "metric_column": "validation_mae",
        "metric_label": "MAE",
        "maximize": False,
        "prefix": "rankup_",
        "title": "Optuna convergence · Ordinal regression (validation MAE)",
    },
}
DATASET_LABELS = {
    "adult_income": "Adult Income",
    "bank_marketing": "Bank Marketing",
    "cmc": "CMC",
    "diabetes_pima": "Diabetes Pima",
    "satimage": "Satimage",
    "melbourne_airbnb": "Melbourne Airbnb",
    "student_performance": "Student Performance",
    "nhanes": "NHANES",
    "esl": "ESL",
    "lev": "LEV",
}
AUGMENTATIONS = ["gaussian", "scarf", "range_limited", "tabcutmix", "tabcutmix_plus", "zero_out"]
AUGMENTATION_LABELS = {
    "gaussian": "Gaussian",
    "scarf": "SCARF",
    "range_limited": "Range-limited",
    "tabcutmix": "TabCutMix",
    "tabcutmix_plus": "TabCutMix+",
    "zero_out": "Zero-out",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_run(task_type: str, dataset: str, augmentation: str, metric: str) -> tuple[np.ndarray, np.ndarray, int]:
    config = FIGURES[task_type]
    prefix = config["prefix"]
    path = ROOT / dataset / f"{prefix}{augmentation}_optuna_trials.csv"
    rows = read_csv(path)
    points = []
    for row in rows:
        raw_score = row.get(metric, "")
        if row.get("state", "COMPLETE") != "COMPLETE" or not raw_score:
            continue
        points.append((int(float(row["trial"])), float(raw_score)))
    points.sort(key=lambda item: item[0])
    if not points:
        raise ValueError(f"No completed trials with {metric} in {path}")
    trials = np.array([point[0] for point in points])
    scores = np.array([point[1] for point in points])
    return trials, scores, len(rows)


def plot_task(task_type: str) -> None:
    config = FIGURES[task_type]
    datasets = config["datasets"]
    metric = config["metric_column"]
    maximize = config["maximize"]
    fig, axes = plt.subplots(6, 5, figsize=(19, 14.8), sharex=False, sharey=False)

    for row_index, augmentation in enumerate(AUGMENTATIONS):
        for column_index, dataset in enumerate(datasets):
            ax = axes[row_index, column_index]
            trials, scores, trial_count = load_run(task_type, dataset, augmentation, metric)
            running_best = np.maximum.accumulate(scores) if maximize else np.minimum.accumulate(scores)
            best_index = int(np.argmax(scores) if maximize else np.argmin(scores))

            ax.plot(trials, scores, color="#79add0", linewidth=0.9, alpha=0.72, zorder=2)
            ax.step(trials, running_best, where="post", color="#df4b5b", linewidth=1.5, zorder=3)
            ax.scatter([trials[best_index]], [scores[best_index]], s=18, color="#171717", zorder=4)
            ax.set_title(f"{DATASET_LABELS[dataset]}\nn={trial_count}", fontsize=8.5, pad=4)
            ax.grid(True, color="#d4d4d4", linewidth=0.65)
            ax.tick_params(axis="both", labelsize=7, length=2, pad=2)
            ax.set_xlim(min(trials) - 1, max(trials) + 1)
            lower, upper = float(scores.min()), float(scores.max())
            padding = max((upper - lower) * 0.1, abs(lower) * 0.002, 0.0005)
            ax.set_ylim(lower - padding, upper + padding)
            if column_index == 0:
                ax.set_ylabel(f"{AUGMENTATION_LABELS[augmentation]}\nValidation {config['metric_label']}", fontsize=8, labelpad=6)
            if row_index == len(AUGMENTATIONS) - 1:
                ax.set_xlabel("Trial", fontsize=8, labelpad=3)

    fig.suptitle(config["title"], fontsize=15, fontweight="bold", x=0.055, ha="left", y=0.995)
    legend_handles = [
        Line2D([0], [0], color="#79add0", lw=1.3, label="Skor validasi per trial"),
        Line2D([0], [0], color="#df4b5b", lw=1.7, label="Best-so-far"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#171717", markersize=5, label="Trial terbaik"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 0.966),
               ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.065, hspace=0.48, wspace=0.28)
    output = ROOT / f"optuna_convergence_{task_type}"
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for task_type in FIGURES:
        plot_task(task_type)


if __name__ == "__main__":
    main()
