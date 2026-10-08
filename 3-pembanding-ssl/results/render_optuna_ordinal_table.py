from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATASETS = [
    ("melbourne_airbnb", "melbourneAirbnb", 48),
    ("student_performance", "studentPerformance", 43),
    ("nhanes", "nhanes", 30),
    ("esl", "esl", 4),
    ("lev", "lev", 4),
]
METHODS = [
    ("gaussian", "Gaussian (σ)"),
    ("scarf", "SCARF (fitur)"),
    ("range_limited", "Range-Limited (p)"),
    ("tabcutmix", "TabCutMix (p)"),
]


def read_best(dataset: str, method: str) -> dict:
    path = RESULTS / dataset / f"rankup_{method}_best_params.json"
    return json.loads(path.read_text(encoding="utf-8"))


def values(dataset: str, method: str, feature_count: int) -> tuple[str, str]:
    result = read_best(dataset, method)
    params = result["parameters"]
    if method == "scarf":
        weak = params["weak_q"]
        strong = params.get("strong_q")
        if strong is None:
            strong = round(result["strong_strength"] * feature_count)
        return str(weak), str(strong)
    if method == "gaussian":
        return f'{result["weak_strength"]:.3f}', f'{result["strong_strength"]:.3f}'
    return f'{result["weak_strength"]:.3f}', f'{result["strong_strength"]:.3f}'


def render() -> None:
    fig, ax = plt.subplots(figsize=(15, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x0, table_width = 0.025, 0.95
    y_top, table_height = 0.86, 0.69
    group_h, sub_h = 0.15, 0.13
    row_h = (table_height - group_h - sub_h) / len(DATASETS)
    dataset_w = 0.19
    value_w = (1 - dataset_w) / 8
    dark, mid = "#24566f", "#4d8399"
    edge = "#2f3437"

    # Dataset header spans both header rows.
    ax.add_patch(Rectangle((x0, y_top - group_h - sub_h), table_width * dataset_w,
                           group_h + sub_h, facecolor=dark, edgecolor=edge, linewidth=1.0))
    ax.text(x0 + table_width * dataset_w / 2, y_top - (group_h + sub_h) / 2,
            "Dataset", ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    for mi, (_, title) in enumerate(METHODS):
        gx = x0 + table_width * (dataset_w + mi * 2 * value_w)
        gw = table_width * 2 * value_w
        ax.add_patch(Rectangle((gx, y_top - group_h), gw, group_h,
                               facecolor=dark, edgecolor=edge, linewidth=1.0))
        ax.text(gx + gw / 2, y_top - group_h / 2, title, ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
        for si, sub in enumerate(("Weak", "Strong")):
            sx = gx + si * table_width * value_w
            sw = table_width * value_w
            ax.add_patch(Rectangle((sx, y_top - group_h - sub_h), sw, sub_h,
                                   facecolor=mid, edgecolor=edge, linewidth=1.0))
            ax.text(sx + sw / 2, y_top - group_h - sub_h / 2, sub,
                    ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    for ri, (dataset, label, feature_count) in enumerate(DATASETS):
        y = y_top - group_h - sub_h - (ri + 1) * row_h
        fill = "#e3eaed" if ri % 2 == 0 else "#f4f4f5"
        ax.add_patch(Rectangle((x0, y), table_width * dataset_w, row_h,
                               facecolor=fill, edgecolor=edge, linewidth=0.9))
        ax.text(x0 + table_width * dataset_w / 2, y + row_h / 2, label,
                ha="center", va="center", fontsize=10)
        col = 0
        for method, _ in METHODS:
            weak, strong = values(dataset, method, feature_count)
            for value in (weak, strong):
                x = x0 + table_width * (dataset_w + col * value_w)
                w = table_width * value_w
                ax.add_patch(Rectangle((x, y), w, row_h, facecolor=fill,
                                       edgecolor=edge, linewidth=0.9))
                ax.text(x + w / 2, y + row_h / 2, value, ha="center", va="center", fontsize=10)
                col += 1

    fig.suptitle("Best parameter Optuna — RankUp SSL (ordinal)",
                 x=0.025, y=0.97, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.025, 0.075,
             "Nilai weak/strong diambil dari best trial Optuna pada validation. SCARF ditampilkan sebagai jumlah fitur yang dikorupsi (q).",
             fontsize=8.5, color="#374151")
    fig.text(0.025, 0.035,
             "Gaussian: σ; Range-Limited dan TabCutMix: p. Nilai merupakan parameter augmentasi, bukan skor model.",
             fontsize=8.5, color="#374151")
    fig.savefig(RESULTS / "optuna_best_parameters_ordinal.png", dpi=260,
                bbox_inches="tight", facecolor="white")
    fig.savefig(RESULTS / "optuna_best_parameters_ordinal.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render()
