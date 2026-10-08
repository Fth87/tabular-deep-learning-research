from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATASETS = [
    ("diabetes_pima", "Diabetes Pima", "Klasifikasi"),
    ("cmc", "CMC", "Klasifikasi"),
    ("satimage", "Satimage", "Klasifikasi"),
    ("bank_marketing", "Bank Marketing", "Klasifikasi"),
    ("adult_income", "Adult Income", "Klasifikasi"),
    ("melbourne_airbnb", "Melbourne Airbnb", "Ordinal"),
    ("student_performance", "Student Performance", "Ordinal"),
    ("nhanes", "NHANES", "Ordinal"),
    ("esl", "ESL", "Ordinal"),
    ("lev", "LEV", "Ordinal"),
]
METHODS = [
    ("gaussian", "Gaussian (σ)"),
    ("scarf", "SCARF (fitur)"),
    ("range_limited", "Range-Limited (p)"),
    ("tabcutmix", "TabCutMix (p)"),
    ("tabcutmix_plus", "TabCutMixPlus (p)"),
    ("zero_out", "Zero-Out (p)"),
]


def read_params(dataset: str, task: str, method: str) -> dict:
    prefix = "rankup_cls" if task == "Klasifikasi" else "rankup"
    path = RESULTS / dataset / f"{prefix}_{method}_best_params.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows() -> list[list[str]]:
    rows = []
    for dataset, name, task in DATASETS:
        row = [f"{name} ({task})"]
        for method, _ in METHODS:
            result = read_params(dataset, task, method)
            if method == "scarf":
                params = result["parameters"]
                weak_q = int(params["weak_q"])
                n_features = round(weak_q / result["weak_strength"])
                strong_q = round(result["strong_strength"] * n_features)
                row.extend((str(weak_q), str(strong_q)))
            else:
                row.extend((f'{result["weak_strength"]:.3f}', f'{result["strong_strength"]:.3f}'))
        rows.append(row)
    return rows


def render_table(rows: list[list[str]]) -> None:
    fig, ax = plt.subplots(figsize=(22, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x0, total_w, y_top = 0.018, 0.964, 0.87
    group_h, sub_h = 0.115, 0.09
    row_h = 0.061
    dataset_w = 0.16
    value_w = (1 - dataset_w) / (len(METHODS) * 2)
    dark, mid, edge = "#24566f", "#4d8399", "#30383c"

    ax.add_patch(Rectangle((x0, y_top - group_h - sub_h), total_w * dataset_w,
                           group_h + sub_h, facecolor=dark, edgecolor=edge, linewidth=0.85))
    ax.text(x0 + total_w * dataset_w / 2, y_top - (group_h + sub_h) / 2,
            "Dataset", ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")

    for mi, (_, label) in enumerate(METHODS):
        x = x0 + total_w * (dataset_w + mi * 2 * value_w)
        w = total_w * 2 * value_w
        ax.add_patch(Rectangle((x, y_top - group_h), w, group_h,
                               facecolor=dark, edgecolor=edge, linewidth=0.85))
        ax.text(x + w / 2, y_top - group_h / 2, label, ha="center", va="center",
                color="white", fontsize=9.1, fontweight="bold")
        for si, sub in enumerate(("Weak", "Strong")):
            sx = x + total_w * si * value_w
            sw = total_w * value_w
            ax.add_patch(Rectangle((sx, y_top - group_h - sub_h), sw, sub_h,
                                   facecolor=mid, edgecolor=edge, linewidth=0.85))
            ax.text(sx + sw / 2, y_top - group_h - sub_h / 2, sub,
                    ha="center", va="center", color="white", fontsize=8.8, fontweight="bold")

    for ri, row in enumerate(rows):
        y = y_top - group_h - sub_h - (ri + 1) * row_h
        fill = "#e3eaed" if ri % 2 == 0 else "#f4f4f5"
        ax.add_patch(Rectangle((x0, y), total_w * dataset_w, row_h,
                               facecolor=fill, edgecolor=edge, linewidth=0.75))
        ax.text(x0 + total_w * dataset_w / 2, y + row_h / 2, row[0],
                ha="center", va="center", fontsize=8.5)
        for ci, value in enumerate(row[1:]):
            x = x0 + total_w * (dataset_w + ci * value_w)
            w = total_w * value_w
            ax.add_patch(Rectangle((x, y), w, row_h, facecolor=fill,
                                   edgecolor=edge, linewidth=0.75))
            ax.text(x + w / 2, y + row_h / 2, value, ha="center", va="center", fontsize=8.6)

    fig.suptitle("Best parameter Optuna — seluruh eksperimen RankUp",
                 x=0.02, y=0.965, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.07,
             "Weak/strong diambil dari best trial Optuna berdasarkan validation. SCARF ditampilkan sebagai jumlah fitur yang dikorupsi (q).",
             fontsize=8.5, color="#374151")
    fig.text(0.02, 0.035,
             "60 hasil tuning: 6 augmentasi × 10 dataset. Nilai pada tabel adalah parameter augmentasi, bukan skor model.",
             fontsize=8.5, color="#374151")
    fig.savefig(RESULTS / "optuna_best_parameters_all.png", dpi=260,
                bbox_inches="tight", facecolor="white")
    fig.savefig(RESULTS / "optuna_best_parameters_all.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_tsv(rows: list[list[str]]) -> None:
    headers = ["Dataset"]
    for _, label in METHODS:
        headers.extend((f"{label} Weak", f"{label} Strong"))
    with (RESULTS / "optuna_best_parameters_all.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    all_rows = build_rows()
    render_table(all_rows)
    save_tsv(all_rows)
