from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIG = ROOT / "config" / "datasets.json"
OPENML = ROOT / "dataset" / "openml_manifest.json"

ORDER = {
    "classification": ["diabetes_pima", "cmc", "satimage", "bank_marketing", "adult_income"],
    "ordinal_regression": ["melbourne_airbnb", "student_performance", "nhanes", "esl", "lev"],
}
LABELS = {
    "diabetes_pima": "Diabetes Pima",
    "cmc": "CMC",
    "satimage": "Satimage",
    "bank_marketing": "Bank Marketing",
    "adult_income": "Adult Income",
    "melbourne_airbnb": "Melbourne Airbnb",
    "student_performance": "Student Performance",
    "nhanes": "NHANES",
    "esl": "ESL",
    "lev": "LEV",
}
PREDICTED = {
    "diabetes_pima": "Status diabetes (negatif/positif)",
    "cmc": "Metode kontrasepsi yang digunakan",
    "satimage": "Kelas tutupan lahan dari citra satelit",
    "bank_marketing": "Langganan deposito berjangka",
    "adult_income": "Kelompok pendapatan tahunan",
    "melbourne_airbnb": "Kategori harga listing Airbnb",
    "student_performance": "Nilai akhir siswa",
    "nhanes": "Status kesehatan umum yang dilaporkan sendiri",
    "esl": "Kecocokan kandidat terhadap pekerjaan",
    "lev": "Penilaian performa dosen",
}
SPLITS = {
    "diabetes_pima": (53, 483, 116, 116),
    "cmc": (103, 928, 221, 221),
    "satimage": (450, 4050, 965, 965),
    "bank_marketing": (3164, 28483, 6782, 6782),
    "adult_income": (3418, 30770, 7327, 7327),
    "melbourne_airbnb": (1402, 12622, 3006, 3006),
    "student_performance": (46, 416, 100, 100),
    "nhanes": (365, 3290, 784, 784),
    "esl": (34, 306, 74, 74),
    "lev": (70, 630, 150, 150),
}


def dataset_records() -> dict[str, dict[str, object]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    openml = {row["slug"]: row for row in json.loads(OPENML.read_text(encoding="utf-8"))}
    records = {}
    for item in config["datasets"]:
        path = ROOT / item["csv"]
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            raw = list(reader)
            target_counts: dict[str, int] = {}
            for row in raw:
                label = row[item["target"]]
                target_counts[label] = target_counts.get(label, 0) + 1
            feature_count = len(reader.fieldnames or []) - 1 - len(item.get("drop_columns", []))

        if item["task_type"] == "classification":
            metadata = openml[item["name"]]
            numeric = len(metadata["numeric_features"])
            categorical = len(metadata["categorical_features"])
            labels = list(metadata["target_counts"])
        else:
            numeric, categorical = feature_count, 0
            labels = sorted(target_counts, key=float)

        if item["name"] == "diabetes_pima":
            dist = f"Negatif {target_counts['tested_negative'] / len(raw) * 100:.1f}%; positif {target_counts['tested_positive'] / len(raw) * 100:.1f}%"
        elif item["name"] == "adult_income":
            dist = f"≤50K {target_counts['<=50K'] / len(raw) * 100:.1f}%; >50K {target_counts['>50K'] / len(raw) * 100:.1f}%"
        elif item["task_type"] == "classification":
            label_order = sorted(labels, key=lambda x: float(x.rstrip('.')))
            parts = [f"{label.rstrip('.')} {target_counts[label] / len(raw) * 100:.1f}%" for label in label_order]
            dist = "; ".join("; ".join(parts[i:i + 3]) for i in range(0, len(parts), 3))
        else:
            percents = [target_counts[label] / len(raw) * 100 for label in labels]
            chunks = [", ".join(f"{p:.1f}%" for p in percents[i:i + 5]) for i in range(0, len(percents), 5)]
            dist = f"Level 0–{len(labels)-1}: " + "\n".join(chunks)

        records[item["name"]] = {
            "name": LABELS[item["name"]],
            "task": item["task_type"],
            "samples": len(raw),
            "features": feature_count,
            "numeric": numeric,
            "categorical": categorical,
            "classes": len(target_counts),
            "distribution": dist,
            "predicted": PREDICTED[item["name"]],
            "split": SPLITS[item["name"]],
        }
    return records


def render(task: str, filename: str, title: str) -> None:
    records = dataset_records()
    names = ORDER[task]
    rows = []
    for i, key in enumerate(names, start=1):
        d = records[key]
        labeled, unlabeled, validation, test = d["split"]
        train_pool = labeled + unlabeled
        rows.append([
            str(i), d["name"], f"{train_pool:,}\n({labeled:,} L / {unlabeled:,} U)",
            f"{validation:,}", f"{test:,}", f"{d['samples']:,}",
            f"{d['features']}\n({d['numeric']} N / {d['categorical']} K)",
            str(d["classes"]), d["distribution"], d["predicted"],
        ])

    fig, ax = plt.subplots(figsize=(22, 7.8))
    ax.axis("off")
    fig.suptitle(title, x=0.035, y=0.975, ha="left", fontsize=18, fontweight="bold")
    table = ax.table(
        cellText=rows,
        colLabels=["No", "Dataset", "Train pool\n(L/U)", "Val", "Test", "Total", "Fitur\n(total; N/K)", "Kelas /\nlevel", "Distribusi target (%)", "Yang diprediksi"],
        colWidths=[0.035, 0.12, 0.11, 0.065, 0.065, 0.075, 0.105, 0.065, 0.19, 0.17],
        cellLoc="center",
        bbox=[0.015, 0.105, 0.97, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#4b5563")
        cell.set_linewidth(0.65 if row == 0 else 0.45)
        if row == 0:
            cell.set_facecolor("#bfd0eb")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(9.2)
        else:
            cell.set_facecolor("#f3f5f8" if row % 2 else "white")
            if col in (1, 8, 9):
                cell.get_text().set_ha("left")
            cell.get_text().set_wrap(True)
    fig.text(0.02, 0.055,
             "Train pool = data berlabel (L) + tanpa label (U). Split: 70% train / 15% validation / 15% test; 10% dari train pool berlabel.",
             fontsize=9.3, color="#374151")
    fig.text(0.02, 0.022,
             "Fitur N/K = numerik/kategorikal. Distribusi ordinal mengikuti level 0 sampai level tertinggi; persentase mengikuti urutan tersebut.",
             fontsize=8.8, color="#4b5563")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)
    for extension in ("png", "pdf"):
        fig.savefig(RESULTS / f"dataset_profile_{filename}.{extension}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    render("classification", "classification", "Dataset klasifikasi")
    render("ordinal_regression", "ordinal", "Dataset ordinal regression")


if __name__ == "__main__":
    main()
