from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "experiment_summary.csv"
OUT = ROOT / "results" / "visualizations"

ORDINAL_METHODS = {
    "RankUp + Gaussian": "RankUp + gaussian",
    "RankUp + RLA": "RankUp + range_limited",
    "RankUp + SCARF": "RankUp + scarf",
    "RankUp + TabCutMix": "RankUp + tabcutmix",
    "RankUp + TabCutMixPlus": "RankUp + tabcutmix_plus",
    "RankUp + Zero-Out": "RankUp + zero_out",
    "GFTab": "GFTab (authors' GKSMT, GFK loss and tree embedding; ordered classes)",
    "SSLAE": "SSLAE (authors' implementation; ordinal labels treated as ordered classes)",
    "STUNT": "STUNT (self-generated tasks, prototypical few-shot evaluation; ordered classes)",
    "VIME": "VIME (author corruption/loss, TensorFlow 2 adapter; ordered classes)",
}

CLASSIFICATION_METHODS = {
    "RankUp + Gaussian": "RankUp-CLS + gaussian",
    "RankUp + RLA": "RankUp-CLS + range_limited",
    "RankUp + SCARF": "RankUp-CLS + scarf",
    "RankUp + TabCutMix": "RankUp-CLS + tabcutmix",
    "RankUp + TabCutMixPlus": "RankUp-CLS + tabcutmix_plus",
    "RankUp + Zero-Out": "RankUp-CLS + zero_out",
}

DATASET_ORDER = {
    "classification": ["adult_income", "bank_marketing", "cmc", "diabetes_pima", "satimage"],
    "ordinal_regression": ["melbourne_airbnb", "student_performance", "nhanes", "esl", "lev"],
}

VAL_COLOR = "#31688e"
TEST_COLOR = "#fde725"


def draw_comparison(ax, rows: pd.DataFrame, methods: dict[str, str],
                    val_column: str, test_column: str, title: str,
                    lower_is_better: bool) -> None:
    values = []
    for label, method in methods.items():
        row = rows[rows.method == method]
        if row.empty:
            continue
        val = row.iloc[0][val_column]
        test = row.iloc[0][test_column]
        if pd.notna(val) and pd.notna(test):
            values.append((label, float(val), float(test)))

    if not values:
        ax.set_title(f"{title} — data validasi tidak tersedia")
        ax.axis("off")
        return

    values.sort(key=lambda item: item[2], reverse=not lower_is_better)
    names = [item[0] for item in values]
    val_scores = np.array([item[1] for item in values])
    test_scores = np.array([item[2] for item in values])
    low = min(val_scores.min(), test_scores.min())
    high = max(val_scores.max(), test_scores.max())
    span = max(high - low, abs(high) * 0.04, 0.01)
    axis_min = low - span * 0.12
    axis_max = high + span * 0.55
    y = np.arange(len(names))
    height = 0.31 if len(names) > 1 else 0.26

    ax.barh(y - height / 2, val_scores - axis_min, left=axis_min, height=height,
            color=VAL_COLOR, label="Validasi", zorder=3)
    ax.barh(y + height / 2, test_scores - axis_min, left=axis_min, height=height,
            color=TEST_COLOR, label="Test", zorder=3)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(axis_min, axis_max)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Skor", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d6dbe0", linewidth=0.7, zorder=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9aa4ad")

    for i, (label, val, test) in enumerate(values):
        ax.text(val + span * 0.012, y[i] - height / 2, f"{val:.3f}",
                va="center", fontsize=7.3, color="#263238", clip_on=False)
        ax.text(test + span * 0.012, y[i] + height / 2, f"{test:.3f}",
                va="center", fontsize=7.3, color="#263238", clip_on=False)
        delta = test - val
        ax.text(axis_max - span * 0.025, y[i], f"Δ {delta:+.3f}", ha="right",
                va="center", fontsize=7.3, color="#374151")


def render_classification(df: pd.DataFrame) -> None:
    task_rows = df[df.task_type == "classification"]
    datasets = DATASET_ORDER["classification"]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(15, 18), squeeze=False)
    fig.suptitle("Perbandingan skor validasi dan test — klasifikasi", x=0.04, y=0.995,
                 ha="left", fontsize=17, fontweight="bold")

    for i, dataset in enumerate(datasets):
        rows = task_rows[task_rows.dataset == dataset]
        draw_comparison(axes[i, 0], rows, CLASSIFICATION_METHODS,
                        "best_validation_metric", "macro_f1",
                        f"{dataset} · Macro F1", lower_is_better=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.995),
               ncol=2, frameon=False, fontsize=9)
    fig.text(0.04, 0.008,
             "Δ = Macro F1 test − Macro F1 validasi. Grafik memuat enam varian RankUp, "
             "satu-satunya metode dengan skor validasi Macro F1 yang tersimpan. "
             "Skor validasi Macro F1 untuk GFTab, SSLAE, STUNT, dan VIME belum tersedia.",
             fontsize=8.5, color="#374151")
    fig.tight_layout(rect=(0.025, 0.03, 0.985, 0.975), h_pad=1.4)
    fig.savefig(OUT / "validation_test_classification_comparison.png", dpi=240,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_ordinal(df: pd.DataFrame) -> None:
    task_rows = df[df.task_type == "ordinal_regression"]
    datasets = DATASET_ORDER["ordinal_regression"]
    fig, axes = plt.subplots(3, 2, figsize=(17, 16), squeeze=False)
    fig.suptitle("Perbandingan skor validasi dan test — ordinal", x=0.04, y=0.99,
                 ha="left", fontsize=17, fontweight="bold")

    for i, dataset in enumerate(datasets):
        draw_comparison(axes.flat[i], task_rows[task_rows.dataset == dataset],
                        ORDINAL_METHODS, "best_validation_mae", "mae",
                        f"{dataset} · MAE", lower_is_better=True)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99),
               ncol=2, frameon=False, fontsize=9)
    fig.text(0.04, 0.012,
             "Δ = MAE test − MAE validasi; nilai positif berarti MAE test lebih tinggi. "
             "MAE validasi/test tersimpan untuk sembilan metode; SSLAE tidak memiliki "
             "MAE validasi di file hasil.",
             fontsize=8.5, color="#374151")
    fig.tight_layout(rect=(0.025, 0.04, 0.985, 0.955), h_pad=1.5, w_pad=1.8)
    fig.savefig(OUT / "validation_test_ordinal_comparison.png", dpi=240,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    render_classification(df)
    render_ordinal(df)


if __name__ == "__main__":
    main()
