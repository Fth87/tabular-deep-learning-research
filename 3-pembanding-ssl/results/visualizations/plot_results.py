from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "experiment_summary.csv"

DATASET_LABELS = {
    "adult_income": "Adult Income",
    "bank_marketing": "Bank Marketing",
    "cmc": "CMC",
    "diabetes_pima": "Diabetes Pima",
    "satimage": "Satimage",
    "esl": "ESL",
    "lev": "LEV",
    "melbourne_airbnb": "Melbourne Airbnb",
    "nhanes": "NHANES",
    "student_performance": "Student Performance",
}

METHOD_LABELS = {
    "RankUp-CLS + tabcutmix": "RankUp-CLS · TabCutMix",
    "RankUp-CLS + tabcutmix_plus": "RankUp-CLS · TabCutMix+",
    "RankUp-CLS + scarf": "RankUp-CLS · SCARF",
    "RankUp-CLS + gaussian": "RankUp-CLS · Gaussian",
    "RankUp-CLS + zero_out": "RankUp-CLS · Zero-out",
    "RankUp-CLS + range_limited": "RankUp-CLS · Range-limited",
    "RankUp + tabcutmix": "RankUp · TabCutMix",
    "RankUp + tabcutmix_plus": "RankUp · TabCutMix+",
    "RankUp + scarf": "RankUp · SCARF",
    "RankUp + gaussian": "RankUp · Gaussian",
    "RankUp + zero_out": "RankUp · Zero-out",
    "RankUp + range_limited": "RankUp · Range-limited",
    "GFTab (authors' GKSMT, GFK loss and tree embedding)": "GFTab",
    "GFTab (authors' GKSMT, GFK loss and tree embedding; ordered classes)": "GFTab",
    "SSLAE (authors' implementation)": "SSLAE",
    "SSLAE (authors' implementation; ordinal labels treated as ordered classes)": "SSLAE",
    "STUNT (self-generated tasks, prototypical few-shot evaluation)": "STUNT",
    "STUNT (self-generated tasks, prototypical few-shot evaluation; ordered classes)": "STUNT",
    "VIME (author corruption/loss, TensorFlow 2 adapter)": "VIME",
    "VIME (author corruption/loss, TensorFlow 2 adapter; ordered classes)": "VIME",
}

METHOD_ORDER = [
    "RankUp-CLS + tabcutmix", "RankUp-CLS + tabcutmix_plus",
    "RankUp-CLS + scarf", "RankUp-CLS + gaussian",
    "RankUp-CLS + zero_out", "RankUp-CLS + range_limited",
    "GFTab (authors' GKSMT, GFK loss and tree embedding)",
    "SSLAE (authors' implementation)",
    "STUNT (self-generated tasks, prototypical few-shot evaluation)",
    "VIME (author corruption/loss, TensorFlow 2 adapter)",
]
ORDINAL_METHOD_ORDER = [
    "RankUp + tabcutmix", "RankUp + tabcutmix_plus",
    "RankUp + scarf", "RankUp + gaussian",
    "RankUp + zero_out", "RankUp + range_limited",
    "GFTab (authors' GKSMT, GFK loss and tree embedding; ordered classes)",
    "SSLAE (authors' implementation; ordinal labels treated as ordered classes)",
    "STUNT (self-generated tasks, prototypical few-shot evaluation; ordered classes)",
    "VIME (author corruption/loss, TensorFlow 2 adapter; ordered classes)",
]


def plot_heatmap(rows: list[dict[str, str]], task_type: str, metric: str,
                 method_order: list[str], datasets: list[str], title: str,
                 color_map: str, output_name: str) -> None:
    selected = [row for row in rows if row["task_type"] == task_type]
    values = np.full((len(datasets), len(method_order)), np.nan)
    for row in selected:
        if row["method"] in method_order and row[metric].strip():
            values[datasets.index(row["dataset"]), method_order.index(row["method"])] = float(row[metric])

    if np.isnan(values).any():
        missing = np.argwhere(np.isnan(values))
        raise ValueError(f"Missing {metric} values for {task_type}: {missing.tolist()}")

    low, high = float(values.min()), float(values.max())
    fig, ax = plt.subplots(figsize=(16.2, 5.8), layout="constrained")
    im = ax.imshow(values, cmap=color_map, vmin=low, vmax=high, aspect="auto")
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels([METHOD_LABELS[name] for name in method_order], rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS[name] for name in datasets])
    ax.set_title(title, loc="left", fontsize=15, fontweight="semibold", pad=14)
    ax.set_xlabel("Model / augmentation", labelpad=10)
    ax.set_ylabel("Dataset", labelpad=10)
    ax.set_xticks(np.arange(-0.5, len(method_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0, pad=8)
    ax.axvline(5.5, color="#374151", linewidth=1.4)

    spread = high - low or 1
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            normalized = (values[y, x] - low) / spread
            color = "white" if normalized < 0.16 or normalized > 0.82 else "#17212b"
            ax.text(x, y, f"{values[y, x]:.3f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, pad=0.018, shrink=0.91)
    cbar.set_label(metric.replace("_", " ").title() + (" · lebih tinggi lebih baik" if metric == "macro_f1" else " · lebih rendah lebih baik"), labelpad=12)
    cbar.set_ticks(np.linspace(low, high, 4))
    cbar.ax.set_yticklabels([f"{v:.3f}" for v in np.linspace(low, high, 4)])
    fig.text(0.01, -0.025, f"Skala warna mencakup seluruh rentang hasil teramati: {low:.3f}–{high:.3f}. Setiap sel menampilkan skor test.", fontsize=9, color="#4b5563")

    for extension in ("png", "pdf"):
        fig.savefig(HERE / f"{output_name}.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    with SUMMARY.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))

    classification_datasets = ["adult_income", "bank_marketing", "cmc", "diabetes_pima", "satimage"]
    ordinal_datasets = ["esl", "lev", "melbourne_airbnb", "nhanes", "student_performance"]
    plot_heatmap(
        rows, "classification", "macro_f1", METHOD_ORDER, classification_datasets,
        "Classification · Macro-F1 (test)", "RdYlGn", "classification_macro_f1",
    )
    plot_heatmap(
        rows, "ordinal_regression", "mae", ORDINAL_METHOD_ORDER, ordinal_datasets,
        "Ordinal regression · MAE (test)", "RdYlGn_r", "ordinal_mae",
    )


if __name__ == "__main__":
    main()
