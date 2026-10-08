from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "experiment_summary.csv"
OUT = ROOT / "results" / "visualizations"

METHODS = {
    "RankUp": ("RankUp + gaussian", "RankUp-CLS + gaussian"),
    "RankUp + RLA": ("RankUp + range_limited", "RankUp-CLS + range_limited"),
    "RankUp + SCARF": ("RankUp + scarf", "RankUp-CLS + scarf"),
    "RankUp + TabCutMix": ("RankUp + tabcutmix", "RankUp-CLS + tabcutmix"),
    "RankUp + TabCutMixPlus": ("RankUp + tabcutmix_plus", "RankUp-CLS + tabcutmix_plus"),
    "RankUp + Zero-Out": ("RankUp + zero_out", "RankUp-CLS + zero_out"),
    "GFTab": (
        "GFTab (authors' GKSMT, GFK loss and tree embedding; ordered classes)",
        "GFTab (authors' GKSMT, GFK loss and tree embedding)",
    ),
    "SSLAE": (
        "SSLAE (authors' implementation; ordinal labels treated as ordered classes)",
        "SSLAE (authors' implementation)",
    ),
    "STUNT": (
        "STUNT (self-generated tasks, prototypical few-shot evaluation; ordered classes)",
        "STUNT (self-generated tasks, prototypical few-shot evaluation)",
    ),
    "VIME": (
        "VIME (author corruption/loss, TensorFlow 2 adapter; ordered classes)",
        "VIME (author corruption/loss, TensorFlow 2 adapter)",
    ),
}
METRICS = {
    "classification": [
        ("accuracy", "Accuracy", True),
        ("macro_precision", "Macro Precision", True),
        ("macro_recall", "Macro Recall", True),
        ("macro_f1", "Macro F1", True),
    ],
    "ordinal_regression": [
        ("mae", "MAE", False),
        ("rmse", "RMSE", False),
        ("r2", "R²", True),
        ("spearman", "Spearman ρ", True),
        ("qwk", "QWK", True),
    ],
}


def metric_values(df: pd.DataFrame, task: str, metric: str) -> tuple[list[str], np.ndarray]:
    task_rows = df[df.task_type == task]
    ordinal_col = 0 if task == "ordinal_regression" else 1
    means = []
    names = []
    for label, method_names in METHODS.items():
        method = method_names[ordinal_col]
        selected = task_rows[task_rows.method == method]
        if len(selected) != 5:
            raise ValueError(f"Expected 5 dataset results for {method}, found {len(selected)}")
        names.append(label)
        means.append(float(selected[metric].mean()))
    return names, np.asarray(means)


def validation_values(df: pd.DataFrame, task: str) -> tuple[list[str], np.ndarray]:
    task_rows = df[df.task_type == task]
    field = "best_validation_metric" if task == "classification" else "best_validation_mae"
    names, means = [], []
    for label, method_names in METHODS.items():
        method = method_names[0 if task == "ordinal_regression" else 1]
        selected = task_rows[(task_rows.method == method) & task_rows[field].notna()]
        if len(selected) == 5:
            names.append(label)
            means.append(float(selected[field].mean()))
    return names, np.asarray(means)


def draw_metric(ax, names: list[str], scores: np.ndarray, title: str,
                higher_is_better: bool) -> None:
    order = np.argsort(-scores if higher_is_better else scores)
    names_ordered = [names[i] for i in order]
    scores_ordered = scores[order]
    low, high = float(scores.min()), float(scores.max())
    span = max(high - low, abs(high) * 0.02, 0.01)
    axis_min = low - span * 0.12
    axis_max = high + span * 0.22
    goodness = (scores_ordered - low) / (high - low) if high > low else np.full_like(scores_ordered, 0.5)
    if not higher_is_better:
        goodness = 1 - goodness
    colors = plt.get_cmap("viridis")(goodness)

    y = np.arange(len(names_ordered))
    ax.barh(y, scores_ordered - axis_min, left=axis_min, height=0.72,
            color=colors, edgecolor="none", zorder=3)
    ax.set_yticks(y, names_ordered, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(axis_min, axis_max)
    ax.set_title(f"{title} {'↑' if higher_is_better else '↓'}", fontsize=11.5, fontweight="bold", pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d6dbe0", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9aa4ad")
    for yi, value in enumerate(scores_ordered):
        ax.text(value + span * 0.025, yi, f"{value:.3f}", ha="left", va="center",
                fontsize=8, color="#202a31", clip_on=False)


def render_task(df: pd.DataFrame, task: str) -> None:
    metrics = METRICS[task]
    validation_label = "Validation Macro F1" if task == "classification" else "Validation MAE"
    validation_direction = task == "classification"
    n_panels = len(metrics) + 1
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    figsize = (16, 13) if task == "classification" else (17, 13)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    title = "Rata-rata metrik klasifikasi antar 5 dataset" if task == "classification" else "Rata-rata metrik ordinal antar 5 dataset"
    fig.suptitle(title, x=0.04, y=0.98, ha="left", fontsize=17, fontweight="bold")

    for i, (metric, label, higher_is_better) in enumerate(metrics):
        names, scores = metric_values(df, task, metric)
        draw_metric(axes.flat[i], names, scores, label, higher_is_better)
    val_names, val_scores = validation_values(df, task)
    draw_metric(axes.flat[len(metrics)], val_names, val_scores, validation_label, validation_direction)
    if task == "classification":
        val_note = "Validation Macro F1 hanya tersedia untuk enam varian RankUp; skor validasi pembanding tidak tersimpan."
    else:
        val_note = "Validation MAE tersedia untuk sembilan metode; skor SSLAE tidak tersimpan."
    for i in range(n_panels, nrows * ncols):
        axes.flat[i].axis("off")

    note = "Test: rata-rata aritmetik skor per dataset. Panel validasi memakai skor validasi utama yang tersimpan. Metode diurutkan terbaik ke bawah; sumbu dimulai dekat skor minimum. " + val_note
    if task == "ordinal_regression":
        note += " MAE/RMSE dirata-ratakan pada skala level asli tiap dataset."
    fig.text(0.04, 0.018, note, fontsize=8.5, color="#374151")
    fig.tight_layout(rect=(0.025, 0.055, 0.99, 0.94), h_pad=1.7, w_pad=1.8)
    name = "average_classification_metrics_bar" if task == "classification" else "average_ordinal_metrics_bar"
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(SUMMARY)
    for task in METRICS:
        render_task(df, task)


if __name__ == "__main__":
    main()
