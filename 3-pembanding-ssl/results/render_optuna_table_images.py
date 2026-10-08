from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "optuna_best_params.csv"


def render(task_type: str, output_name: str, title: str, metric: str) -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["task_type"] == task_type]

    data = []
    for row in rows:
        parameters = json.loads(row["parameters"])
        parameter_text = " · ".join(f"{name}={value:.6g}" if isinstance(value, (int, float)) else f"{name}={value}"
                                    for name, value in parameters.items())
        data.append([
            row["dataset"], row["augmentation"], row["n_trials"], row["best_trial"],
            f"{float(row['best_validation_metric']):.6f}", parameter_text,
        ])

    fig, ax = plt.subplots(figsize=(19, 12.4))
    ax.axis("off")
    fig.suptitle(title, x=0.045, y=0.985, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.045, 0.956,
             f"Parameter terpilih dari validation {metric}; trial terbaik memakai indeks mulai 0. Total {len(data)} kombinasi.",
             ha="left", fontsize=10, color="#4b5563")

    table = ax.table(
        cellText=data,
        colLabels=["Dataset", "Augmentasi", "Jumlah trial", "Best trial", f"{metric} validasi", "Best parameters"],
        colWidths=[0.15, 0.17, 0.09, 0.09, 0.14, 0.36],
        cellLoc="left",
        bbox=[0.02, 0.025, 0.96, 0.91],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d6dce3")
        cell.set_linewidth(0.45)
        if row_index == 0:
            cell.set_facecolor("#243b53")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(9.2)
        else:
            dataset_group = (row_index - 1) // 6
            cell.set_facecolor("#f1f5f9" if dataset_group % 2 == 0 else "white")
            if col_index in (2, 3, 4):
                cell.get_text().set_ha("center")
            else:
                cell.get_text().set_ha("left")
            if col_index == 0 and (row_index - 1) % 6 == 0:
                cell.get_text().set_weight("bold")

    fig.savefig(ROOT / f"{output_name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / f"{output_name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    render("classification", "optuna_best_params_classification", "Best parameter Optuna · Klasifikasi", "Macro-F1")
    render("ordinal_regression", "optuna_best_params_ordinal", "Best parameter Optuna · Ordinal regression", "MAE")


if __name__ == "__main__":
    main()
