from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "experiment_summary.csv"
OUT = RESULTS / "visualizations"

DATASETS = {
    "classification": [
        ("diabetes_pima", "Diabetes"), ("cmc", "CMC"), ("satimage", "Satimage"),
        ("bank_marketing", "Bank"), ("adult_income", "Adult"),
    ],
    "ordinal_regression": [
        ("melbourne_airbnb", "Melbourne"), ("student_performance", "Student"),
        ("nhanes", "NHANES"), ("esl", "ESL"), ("lev", "LEV"),
    ],
}
METHODS = [
    ("RankUp", "RankUp + gaussian", "RankUp-CLS + gaussian"),
    ("RankUp + RLA", "RankUp + range_limited", "RankUp-CLS + range_limited"),
    ("RankUp + SCARF", "RankUp + scarf", "RankUp-CLS + scarf"),
    ("RankUp + TabCutMix", "RankUp + tabcutmix", "RankUp-CLS + tabcutmix"),
    ("RankUp + TabCutMixPlus", "RankUp + tabcutmix_plus", "RankUp-CLS + tabcutmix_plus"),
    ("RankUp + Zero-Out", "RankUp + zero_out", "RankUp-CLS + zero_out"),
    ("GFTab", "GFTab (authors' GKSMT, GFK loss and tree embedding; ordered classes)",
     "GFTab (authors' GKSMT, GFK loss and tree embedding)"),
    ("SSLAE", "SSLAE (authors' implementation; ordinal labels treated as ordered classes)",
     "SSLAE (authors' implementation)"),
    ("STUNT", "STUNT (self-generated tasks, prototypical few-shot evaluation; ordered classes)",
     "STUNT (self-generated tasks, prototypical few-shot evaluation)"),
    ("VIME", "VIME (author corruption/loss, TensorFlow 2 adapter; ordered classes)",
     "VIME (author corruption/loss, TensorFlow 2 adapter)"),
]
METRICS = {
    "classification": [
        ("accuracy", "Acc", True),
        ("macro_precision", "Prec", True),
        ("macro_recall", "Recall", True),
        ("macro_f1", "F1", True),
    ],
    "ordinal_regression": [
        ("mae", "MAE", False),
        ("rmse", "RMSE", False),
        ("r2", "R²", True),
        ("spearman", "ρ", True),
        ("qwk", "QWK", True),
    ],
}


def matrix_for(df: pd.DataFrame, task: str, metric: str) -> tuple[np.ndarray, list[str]]:
    datasets = DATASETS[task]
    values = np.full((len(METHODS), len(datasets)), np.nan)
    labels = [label for _, label in datasets]
    for r, (short, ordinal_name, class_name) in enumerate(METHODS):
        method = class_name if task == "classification" else ordinal_name
        for c, (dataset, _) in enumerate(datasets):
            match = df[(df.task_type == task) & (df.dataset == dataset) & (df.method == method)]
            if len(match) != 1:
                raise ValueError(f"Expected one result for {dataset}/{method}, found {len(match)}")
            values[r, c] = float(match.iloc[0][metric])
    return values, labels


def plot_metric(df: pd.DataFrame, task: str, metric: str, metric_label: str,
                higher_is_better: bool) -> None:
    values, dataset_labels = matrix_for(df, task, metric)
    methods = [method[0] for method in METHODS]
    finite = values[np.isfinite(values)]
    vmin, vmax = float(finite.min()), float(finite.max())
    if task == "classification":
        vmin, vmax = 0.0, 1.0
    elif np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    fig, ax = plt.subplots(figsize=(10.2, 7.5))
    direction = "lebih tinggi lebih baik" if higher_is_better else "lebih rendah lebih baik"
    fig.suptitle(f"{metric_label} — {task.replace('_', ' ')}",
                 x=0.055, y=0.965, ha="left", fontsize=16, fontweight="bold")
    cmap = "viridis" if higher_is_better else "viridis_r"
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(dataset_labels)), dataset_labels, fontsize=9)
    ax.set_yticks(range(len(methods)), methods, fontsize=9)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_xticks(np.arange(-0.5, len(dataset_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            fraction = (value - vmin) / (vmax - vmin)
            if not higher_is_better:
                fraction = 1 - fraction
            color = "#14212b" if fraction > 0.65 else "white"
            ax.text(c, r, f"{value:.3f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.axhline(5.5, color="#34434b", linewidth=1.4, zorder=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Skor", fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.12, 0.055,
             f"Nilai test set; {direction}. Warna memakai skala {vmin:.3f}–{vmax:.3f}.",
             fontsize=9, color="#374151")
    fig.subplots_adjust(left=0.25, right=0.88, top=0.87, bottom=0.13)

    folder = "classification" if task == "classification" else "ordinal"
    output_dir = OUT / "heatmaps" / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{metric}_heatmap.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{metric}_heatmap.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_combined_task(df: pd.DataFrame, task: str, title: str) -> None:
    metric_specs = METRICS[task]
    method_labels = ["RankUp", "+ RLA", "+ SCARF", "+ TabCutMix", "+ TabCutMixPlus",
                     "+ Zero-Out", "GFTab", "SSLAE", "STUNT", "VIME"]
    if task == "classification":
        fig, axes = plt.subplots(2, 2, figsize=(18, 11), squeeze=False)
        panel_axes = list(axes.flat)
    else:
        fig = plt.figure(figsize=(20, 14))
        grid = fig.add_gridspec(3, 4)
        panel_axes = [
            fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 0:2]), fig.add_subplot(grid[1, 2:4]),
            fig.add_subplot(grid[2, 1:3]),
        ]
    fig.suptitle(title, x=0.045, y=0.985, ha="left", fontsize=18, fontweight="bold")
    for i, (metric, metric_label, higher_is_better) in enumerate(metric_specs):
        ax = panel_axes[i]
        values, dataset_labels = matrix_for(df, task, metric)
        values = values.T
        finite = values[np.isfinite(values)]
        vmin, vmax = float(finite.min()), float(finite.max())
        if task == "classification":
            vmin, vmax = 0.0, 1.0
        elif np.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        cmap = "viridis" if higher_is_better else "viridis_r"
        im = ax.imshow(values, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        direction = "↑" if higher_is_better else "↓"
        ax.set_title(f"{metric_label} {direction}", fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xticks(range(len(method_labels)), method_labels, rotation=38, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(dataset_labels)), dataset_labels, fontsize=8.5)
        ax.tick_params(axis="both", length=0, pad=5)
        ax.set_xticks(np.arange(-0.5, len(method_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(dataset_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axvline(5.5, color="#34434b", linewidth=1.2, zorder=4)

        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                value = values[r, c]
                fraction = (value - vmin) / (vmax - vmin)
                if not higher_is_better:
                    fraction = 1 - fraction
                color = "#14212b" if fraction > 0.65 else "white"
                ax.text(c, r, f"{value:.3f}", ha="center", va="center",
                        fontsize=7.2, color=color)
        ax.set_ylim(values.shape[0] - 0.5, -0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025, format="%.2f")
        colorbar.ax.tick_params(labelsize=7.5)

    fig.text(0.045, 0.018,
             "Nilai test set. Sel berbentuk kotak; tiap panel memakai skala metriknya sendiri. Panah menunjukkan arah yang lebih baik; garis memisahkan RankUp dari metode pembanding.",
             fontsize=8.8, color="#374151")
    if task == "classification":
        fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.94), w_pad=1.8, h_pad=2.0)
    else:
        fig.subplots_adjust(left=0.065, right=0.98, top=0.91, bottom=0.09,
                            wspace=0.36, hspace=0.42)
    filename = "classification_all_metrics_heatmap" if task == "classification" else "ordinal_all_metrics_heatmap"
    fig.savefig(OUT / f"{filename}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(SUMMARY)
    for task, metric_specs in METRICS.items():
        for metric, label, higher_is_better in metric_specs:
            plot_metric(df, task, metric, label, higher_is_better)
    plot_combined_task(df, "classification", "Metrik klasifikasi — seluruh metode dan dataset")
    plot_combined_task(df, "ordinal_regression", "Metrik ordinal — seluruh metode dan dataset")


if __name__ == "__main__":
    main()
