from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable


HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "experiment_summary.csv"
OUTPUT = HERE / "bar_charts"

DATASETS = {
    "classification": ["adult_income", "bank_marketing", "cmc", "diabetes_pima", "satimage"],
    "ordinal_regression": ["esl", "lev", "melbourne_airbnb", "nhanes", "student_performance"],
}
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

CLASSIFICATION_METHODS = [
    "RankUp-CLS + tabcutmix", "RankUp-CLS + tabcutmix_plus",
    "RankUp-CLS + scarf", "RankUp-CLS + gaussian",
    "RankUp-CLS + zero_out", "RankUp-CLS + range_limited",
    "GFTab (authors' GKSMT, GFK loss and tree embedding)",
    "SSLAE (authors' implementation)",
    "STUNT (self-generated tasks, prototypical few-shot evaluation)",
    "VIME (author corruption/loss, TensorFlow 2 adapter)",
]
ORDINAL_METHODS = [
    "RankUp + tabcutmix", "RankUp + tabcutmix_plus",
    "RankUp + scarf", "RankUp + gaussian",
    "RankUp + zero_out", "RankUp + range_limited",
    "GFTab (authors' GKSMT, GFK loss and tree embedding; ordered classes)",
    "SSLAE (authors' implementation; ordinal labels treated as ordered classes)",
    "STUNT (self-generated tasks, prototypical few-shot evaluation; ordered classes)",
    "VIME (author corruption/loss, TensorFlow 2 adapter; ordered classes)",
]
METHOD_LABELS = {
    "RankUp-CLS + tabcutmix": "RankUp · TabCutMix",
    "RankUp-CLS + tabcutmix_plus": "RankUp · TabCutMix+",
    "RankUp-CLS + scarf": "RankUp · SCARF",
    "RankUp-CLS + gaussian": "RankUp · Gaussian",
    "RankUp-CLS + zero_out": "RankUp · Zero-out",
    "RankUp-CLS + range_limited": "RankUp · Range-limited",
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


def plot_dataset(rows: list[dict[str, str]], task_type: str, dataset: str,
                 metric: str, methods: list[str], higher_is_better: bool) -> Path:
    entries = [row for row in rows if row["task_type"] == task_type and row["dataset"] == dataset]
    lookup = {row["method"]: float(row[metric]) for row in entries if row[metric].strip()}
    ranked = sorted(((method, lookup[method]) for method in methods),
                    key=lambda item: item[1], reverse=higher_is_better)
    labels = [METHOD_LABELS[method] for method, _ in ranked]
    values = np.array([value for _, value in ranked])
    low, high = float(values.min()), float(values.max())
    span = high - low or 1.0
    cmap = plt.get_cmap("viridis" if higher_is_better else "viridis_r")
    norm = plt.Normalize(vmin=low, vmax=high)

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    y = np.arange(len(ranked))
    ax.barh(y, values - low, left=low, color=cmap(norm(values)), height=0.7, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(low, high)
    ax.set_xticks(np.linspace(low, high, 5))
    ax.set_xticklabels([f"{value:.3f}" for value in np.linspace(low, high, 5)])
    ax.grid(axis="x", color="#d9dee5", linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0, labelsize=10, pad=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.axhline(5.5, color="#59636e", linewidth=1.1)
    for yi, value in enumerate(values):
        rgba = cmap(norm(value))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        if value >= high - span * 0.14:
            ax.text(value - span * 0.012, yi, f"{value:.3f}", va="center", ha="right",
                    fontsize=9, color="#17212b" if luminance > 0.5 else "white")
        else:
            ax.text(value + span * 0.012, yi, f"{value:.3f}", va="center", ha="left", fontsize=9, color="#263238")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#aeb7c2")

    better = "lebih tinggi lebih baik" if higher_is_better else "lebih rendah lebih baik"
    title_metric = "Macro-F1" if metric == "macro_f1" else "MAE"
    fig.suptitle(f"{DATASET_LABELS[dataset]} · {title_metric} test", x=0.02, y=0.98,
                 ha="left", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"{title_metric} · {better}", labelpad=10)
    cax = fig.add_axes([0.92, 0.23, 0.018, 0.58])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(f"{title_metric} · {better}", labelpad=10)
    fig.subplots_adjust(left=0.28, right=0.87, top=0.90, bottom=0.13)
    fig.text(0.02, 0.025,
             f"Skala sumbu dan warna khusus untuk dataset ini: {low:.3f}–{high:.3f}. Urutan: hasil terbaik ke terendah.",
             fontsize=9, color="#4b5563")

    subdir = OUTPUT / ("classification" if task_type == "classification" else "ordinal_regression")
    subdir.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}_{metric}_bar"
    for extension in ("png", "pdf"):
        fig.savefig(subdir / f"{filename}.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return subdir / f"{filename}.png"


def main() -> None:
    with SUMMARY.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    links: list[str] = [
        "# Bar chart per dataset", "",
        "Setiap dataset memiliki grafik sendiri dengan skala sumbu dan warna yang disesuaikan pada rentang nilainya. Batang diurutkan dari performa terbaik. Untuk klasifikasi, Macro-F1 terbesar di atas; untuk ordinal, MAE terkecil di atas.",
        "",
        "## Klasifikasi · Macro-F1", "",
    ]
    for dataset in DATASETS["classification"]:
        path = plot_dataset(rows, "classification", dataset, "macro_f1", CLASSIFICATION_METHODS, True)
        links.append(f"- {DATASET_LABELS[dataset]}: [PNG]({path.relative_to(OUTPUT).as_posix()}) · [PDF]({path.with_suffix('.pdf').relative_to(OUTPUT).as_posix()})")
    links.extend(["", "## Ordinal · MAE", ""])
    for dataset in DATASETS["ordinal_regression"]:
        path = plot_dataset(rows, "ordinal_regression", dataset, "mae", ORDINAL_METHODS, False)
        links.append(f"- {DATASET_LABELS[dataset]}: [PNG]({path.relative_to(OUTPUT).as_posix()}) · [PDF]({path.with_suffix('.pdf').relative_to(OUTPUT).as_posix()})")
    (OUTPUT / "README.md").write_text("\n".join(links) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
