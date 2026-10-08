from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
TASK3 = ROOT
CONFIG = TASK3 / "config" / "datasets.json"
OUTPUT = TASK3 / "results" / "visualizations"

LABELS = {
    "adult_income": "Adult Income",
    "bank_marketing": "Bank Marketing",
    "cmc": "CMC",
    "diabetes_pima": "Diabetes Pima",
    "satimage": "Satimage",
    "lev": "LEV",
    "nhanes": "NHANES",
    "esl": "ESL",
    "student_performance": "Student Performance",
    "melbourne_airbnb": "Melbourne Airbnb",
}
COLORS = {"classification": "#3266A8", "ordinal_regression": "#D9822B"}


def profile_datasets() -> list[dict[str, object]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    profiles = []
    for entry in config["datasets"]:
        path = TASK3 / entry["csv"]
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            target_values = {row[entry["target"]] for row in rows if row[entry["target"]] not in ("", None)}
            features = len(reader.fieldnames or []) - 1 - len(entry.get("drop_columns", []))
        profiles.append({
            "name": entry["name"],
            "label": LABELS[entry["name"]],
            "task_type": entry["task_type"],
            "samples": len(rows),
            "features": features,
            "targets": len(target_values),
            "target_label": "kelas" if entry["task_type"] == "classification" else "level ordinal",
        })
    return profiles


def main() -> None:
    profiles = profile_datasets()
    task_order = ["classification", "ordinal_regression"]
    ordered = []
    for task in task_order:
        ordered.extend(sorted((p for p in profiles if p["task_type"] == task), key=lambda p: p["samples"], reverse=True))

    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(14.5, 8.3))
    ax.set_xscale("log")
    ax.set_xlim(350, 70000)
    ax.set_ylim(len(ordered) - 0.35, -0.65)

    for index, item in enumerate(ordered):
        color = COLORS[str(item["task_type"])]
        ax.scatter(item["samples"], index, s=56, color=color, zorder=3)
        ax.text(item["samples"] * 1.075, index, f"{item['samples']:,}", va="center", ha="left", fontsize=9.5, color="#263238")

    tick_values = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(["500", "1k", "2k", "5k", "10k", "20k", "50k"])
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{item['label']}   ·   {item['features']} fitur   ·   {item['targets']} {item['target_label']}"
        for item in ordered
    ], fontsize=10)
    ax.grid(axis="x", which="major", color="#d8dee7", linewidth=0.8)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#aeb7c2")
    ax.set_xlabel("Jumlah sampel (baris; skala logaritmik)", labelpad=11)
    ax.set_title("Profil 10 dataset eksperimen", loc="left", fontsize=17, fontweight="bold", pad=22)
    ax.text(0, 1.012, "Sampel, jumlah fitur, dan banyaknya kelas atau level target",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color="#4b5563")

    split_y = 4.5
    ax.axhline(split_y, color="#c5ccd5", linewidth=1.0)
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["classification"],
               markeredgecolor="none", markersize=7, label="Klasifikasi (5 dataset)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["ordinal_regression"],
               markeredgecolor="none", markersize=7, label="Ordinal regression (5 dataset)"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1, 1.095), fontsize=9)
    fig.text(0.02, 0.018, "Eksperimen memakai split train/validation/test 70/15/15; 10% train pool berlabel.",
             fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.34, right=0.92, top=0.88, bottom=0.1)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"dataset_profile.{extension}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
