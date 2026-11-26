"""Ringkas JSONL dan diagnostik tahap 8 di Kaggle atau dari hasil unduhan."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_konteks import (
    CORE, DATASETS, LATENT_DISTILL, MATCH_VARIANTS, METRIC, PERCENTS, SEEDS, SSL_AUGMENTED, SSL_IDENTITY, SUPERVISED, VARIANTS,
    eligible, experiment_names, read_jsonl,
)

KEY = ["dataset", "label_percent", "seed", "name"]
COSINE_BINS = np.linspace(-1, 1, 41)
FAMILIES = {
    "Supervised / label-only": SUPERVISED,
    "SSL tanpa augmentasi eksplisit": SSL_IDENTITY,
    "SSL dengan augmentasi": SSL_AUGMENTED,
    "SSL pilihan validation": (*SSL_IDENTITY, *SSL_AUGMENTED),
}
EFFECTS = (
    ("Distilasi SSL", "head_distill", "head_sup"),
    ("Self-training EMA", "head_match", "head_sup"),
    ("Distilasi vs EMA, view identik", "head_distill", "head_match"),
    ("ICS pada distilasi", "ics_distill_70", "head_distill"),
    ("ICS pada EMA", "ics_match_70", "head_match"),
    ("Gaussian pada EMA", "head_match_gaussian", "head_match"),
    ("Distilasi vs EMA, keduanya Gaussian", "head_gaussian", "head_match_gaussian"),
    ("ICS vs Gaussian pada EMA", "ics_match_70", "head_match_gaussian"),
    ("Distilasi vs EMA, keduanya ICS", "ics_distill_70", "ics_match_70"),
    ("ICS pada supervised", "ics_sup", "head_sup"),
    ("Tambah SSL pada ICS supervised", "ics_both_70", "ics_sup"),
    ("Tambah ICS labeled pada ICS-distill", "ics_both_70", "ics_distill_70"),
    ("ICS vs Gaussian pada distilasi", "ics_distill_70", "head_gaussian"),
    ("ICS vs angular pada distilasi", "ics_distill_70", "head_angular"),
    ("ICS vs SCARF pada distilasi", "ics_distill_70", "head_scarf"),
)


def table(columns, rows):
    yield "| " + " | ".join(columns) + " |"
    yield "|" + "|".join("---" for _ in columns) + "|"
    for row in rows:
        yield "| " + " | ".join(map(str, row)) + " |"


def verify(profile, rows, expected, allowed):
    observed = {tuple(row[key] for key in KEY) for row in rows}
    if expected - observed or observed - allowed:
        raise ValueError(f"Grid tidak sesuai: hilang={sorted(expected - observed)}, ekstra={sorted(observed - allowed)}")
    for row in rows:
        if row["task"] == "classification" and row["name"] in MATCH_VARIANTS:
            for key in ("unlabeled_accept_rate", "unlabeled_accept_rate_through_checkpoint"):
                if row.get(key) is None or not 0 <= row[key] <= 1:
                    raise ValueError(f"Acceptance rate EMA tidak valid: {row['predictions']}")
        with np.load(profile / row["predictions"], allow_pickle=False) as arrays:
            for split in ("validation", "test"):
                if len(arrays[f"pred_{split}"]) != len(arrays[f"y_{split}"]):
                    raise ValueError(f"Shape prediksi tidak cocok: {row['predictions']}")
                if row["task"] == "ordinal_regression" and f"pred_{split}_calibrated" not in arrays:
                    raise ValueError(f"QWK calibrated tidak tersedia: {row['predictions']}")
        if row["diagnostic"]:
            with np.load(profile / row["diagnostic"], allow_pickle=False) as diag:
                if not np.allclose(diag["cosine_ics"], diag["cosine_angular"], atol=1e-4, rtol=0, equal_nan=True):
                    raise ValueError(f"Cosine pembanding tidak cocok: {row['diagnostic']}")


def summarize_diagnostic(profile, rows):
    path_groups = {row["diagnostic"]: (row["dataset"], row["label_percent"], row["seed"]) for row in rows if row["diagnostic"]}
    bins = []
    group_means = []
    for path, group in sorted(path_groups.items()):
        dataset, percent, seed = group
        rho = Path(path).stem.removeprefix("ics_")
        with np.load(profile / path, allow_pickle=False) as data:
            keep = np.isfinite(data["cosine_ics"])
            cos = data["cosine_ics"][keep]
            drift = {key: data[f"drift_{key}"][keep] for key in ("ics", "angular", "teacher")}
        if not len(cos):
            continue
        group_means.append((dataset, percent, seed, rho, len(cos), cos.mean(),
                            *(arr.mean() for arr in drift.values())))
        idx = np.clip(np.searchsorted(COSINE_BINS, cos, side="right") - 1, 0, len(COSINE_BINS) - 2)
        for i in np.unique(idx):
            mask = idx == i
            bins.append((dataset, percent, seed, rho, COSINE_BINS[i], COSINE_BINS[i + 1], int(mask.sum()),
                         cos[mask].mean(), *(arr[mask].mean() for arr in drift.values())))
    if not group_means:
        return []
    cols = ["dataset", "label_percent", "seed", "rho"]
    metrics = ["n", "cosine", "ics", "angular", "teacher"]
    group_df = pd.DataFrame(group_means, columns=[*cols, *metrics])
    group_df.to_csv(profile / "diagnostik_per_grup.csv", index=False)
    bin_df = pd.DataFrame(bins, columns=[*cols, "cos_min", "cos_max", *metrics])
    bin_df.to_csv(profile / "diagnostik_cosine.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = bin_df[bin_df.rho == "70"]
    if not primary.empty:
        fig, axes = plt.subplots(3, 3, figsize=(11, 8), squeeze=False, layout="constrained")
        for i, dataset in enumerate(DATASETS):
            for j, percent in enumerate(PERCENTS):
                ax = axes[i, j]
                subset = primary[(primary.dataset == dataset) & (primary.label_percent == percent)]
                agg = subset.groupby("cos_min")[["cosine", "ics", "angular"]].mean()
                ax.set_title(f"{dataset} · {percent}%")
                if j == 0:
                    ax.set_ylabel("Fraksi kelas berubah" if dataset == "pendigits" else "Rerata |Δ prediksi|")
                if i == 2:
                    ax.set_xlabel("Cosine ke weak")
                if agg.empty:
                    ax.set_axis_off()
                    continue
                for method, label in (("ics", "ICS"), ("angular", "Angular")):
                    ax.plot(agg.cosine, agg[method], marker="o", markersize=3, label=label)
                ax.legend()
        fig.savefig(profile / "diagnostik_cosine.png", dpi=160)
        plt.close(fig)

    lines = [
        "## Diagnostik H1 pada ρ = 0,7", "",
        "Rerata tiap seed diberi bobot sama. Kolom ICS dan angular memakai probe yang sama. "
        "Drift teacher berasal dari prediksi TabPFN penuh dan subset, sehingga dilaporkan sebagai pengukuran terpisah.",
        "",
    ]
    grouped = group_df[group_df.rho == "70"].groupby(["dataset", "label_percent"])
    pairs = [[dataset, percent, len(block), *(f"{block[key].mean():.5f}" for key in ("cosine", "ics", "angular", "teacher"))]
             for (dataset, percent), block in grouped]
    lines.extend(table(["Dataset", "Label %", "Seed", "Cosine", "Drift ICS", "Drift angular", "Drift teacher"], pairs))
    lines += ["", "Data per seed dan bin cosine ada di `diagnostik_per_grup.csv` dan `diagnostik_cosine.csv`.", ""]
    return lines


def validation_selection(frame):
    result = []
    for (dataset, percent), cell in frame.groupby(["dataset", "label_percent"], sort=False):
        metric, sign = METRIC[dataset]
        column = f"validation_{metric}"
        blocks = {name: rows for name, rows in cell.groupby("name")}
        if not all(name in blocks and tuple(sorted(blocks[name].seed)) == SEEDS
                   and blocks[name][column].notna().all() for name in CORE):
            continue
        teacher_score = blocks["tabpfn"][column].mean()
        for family, methods in FAMILIES.items():
            winner = max(methods, key=lambda name: sign * blocks[name][column].mean())
            rows = blocks[winner]
            validation = rows[column].mean()
            result.append({
                "dataset": dataset, "label_percent": percent, "family": family, "name": winner, "metric": metric,
                "validation": validation, "test": rows[f"test_{metric}"].mean(),
                "validation_gain_vs_tabpfn": sign * (validation - teacher_score),
            })
    return result


def compare_history(profile, frame, selected):
    history_path = Path(__file__).with_name("stage8_history.json")
    config_path = profile / "config.json"
    lines = ["## Pembanding historis", ""]
    if not history_path.is_file() or not config_path.is_file():
        return lines + ["Metadata referensi atau konfigurasi tidak tersedia.", ""]
    history = json.loads(history_path.read_text())
    if json.loads(config_path.read_text())["data_sha256"] != history["data_sha256"]:
        return lines + ["Fingerprint split berbeda dari referensi repo. Tabel historis dilewati.", ""]
    rows = history["rows"]
    bridge = []
    for winner in selected:
        dataset, percent, name = (winner[key] for key in ("dataset", "label_percent", "name"))
        metric = "qwk" if dataset == "nhanes" else METRIC[dataset][0]
        current = frame[(frame.dataset == dataset) & (frame.label_percent == percent) & (frame.name == name)]
        references = [r for r in rows if (r["dataset"], r["label_percent"]) == (dataset, percent)]
        for reference in references:
            bridge.append({
                "dataset": dataset, "label_percent": percent, "metric": metric, "family": winner["family"],
                "stage8_name": name, "stage8_validation": current[f"validation_{metric}"].mean(),
                "stage8_test": current[f"test_{metric}"].mean(), "reference_series": reference["series"],
                "reference_name": reference["name"], "reference_validation": reference["validation"],
                "reference_test": reference["test"], "reference_picked_by_validation": reference["picked_by_validation"],
            })
    pd.DataFrame(bridge).to_csv(profile / "pembanding_historis.csv", index=False)
    lines += [
        "CSV memuat supervised dan RankUp tahap 5, RankUp dan SSR-RCUS tahap 6, serta teacher dan distill tahap 7 versi 2.5/3.5. "
        "Varian historis terpilih memakai rerata validation pada setiap sel. Arsitektur dan protokol checkpoint mengikuti tahap asal. "
        "Untuk NHANES, CSV memakai QWK mentah pada kedua sisi. Pemilihan tahap 8 tetap memakai QWK terkalibrasi.",
        "",
    ]
    compact = []
    for dataset in DATASETS:
        for percent in PERCENTS:
            matches = [r for r in bridge if (r["dataset"], r["label_percent"], r["family"])
                       == (dataset, percent, "SSL pilihan validation")]
            if not matches:
                continue
            stage8 = matches[0]
            anchors = ["Tahap 5 · TabM", "Tahap 6 · TabM", "Tahap 7 · 2.5", "Tahap 7 · 3.5"]
            refs = {r["reference_series"]: f"{r['reference_name']} ({r['reference_test']:.4f})"
                    for r in matches if r["reference_picked_by_validation"]}
            compact.append([dataset, percent, stage8["metric"], f"{stage8['stage8_name']} ({stage8['stage8_test']:.4f})",
                            *(refs.get(source, "—") for source in anchors)])
    lines.extend(table(
        ["Dataset", "Label %", "Metrik test", "SSL tahap 8", "SSL tahap 5", "RCUS tahap 6",
         "Augmentasi tahap 7 · 2.5", "Augmentasi tahap 7 · 3.5"], compact,
    ))
    lines.append("")
    return lines


def report(profile, rows):
    wide_rows = []
    for row in rows:
        flattened = {k: row[k] for k in KEY}
        for key in ("ssl_method", "unlabeled_accept_rate", "unlabeled_accept_rate_through_checkpoint"):
            flattened[key] = row.get(key)
        for split in ("validation", "test"):
            flattened.update({f"{split}_{k}": v for k, v in row[split].items()})
        wide_rows.append(flattened)
    frame = pd.DataFrame(wide_rows)
    frame.to_csv(profile / "skor_per_run.csv", index=False)
    lines = ["# Ringkasan augmentasi konteks", "", f"{len(rows)} run selesai.", "",
             "NHANES memakai QWK terkalibrasi untuk pemilihan checkpoint. Batas kelas hanya dari validation. "
             "QWK mentah, Spearman dan MSE tetap dicatat. Angka test hanya dilaporkan.", ""]
    chosen = validation_selection(frame)
    if chosen:
        pd.DataFrame(chosen).to_csv(profile / "pilihan_validation.csv", index=False)
        lines += ["## Pilihan per dataset dan fraksi label", "",
                  "Setiap sel memerlukan grid utama lengkap pada ketiga seed. Metode dipilih dari rerata validation "
                  "untuk masing-masing keluarga. Δ positif berarti membaik terhadap teacher TabPFN di validation.", ""]
        lines.extend(table(
            ["Dataset", "Label %", "Keluarga", "Pilihan", "Validation", "Test", "Δ validation vs teacher"],
            ([r["dataset"], r["label_percent"], r["family"], r["name"], f"{r['validation']:.4f}",
              f"{r['test']:.4f}", f"{r['validation_gain_vs_tabpfn']:+.4f}"] for r in chosen),
        ))
        lines.append("")
        lines.extend(compare_history(profile, frame, chosen))
    for dataset, (metric, sign) in METRIC.items():
        data = frame[frame.dataset == dataset]
        if data.empty:
            continue
        lines += [f"## {dataset} ({metric})", ""]
        summaries = []
        for name in VARIANTS:
            subset = data[data.name == name]
            if subset.empty:
                continue
            for percent, group in subset.groupby("label_percent"):
                def mean_std(column):
                    vals = group[column].dropna()
                    return f"{vals.mean():.4f} ± {vals.std():.4f}" if len(vals) > 1 else f"{vals.mean():.4f}"
                summaries.append([name, percent, len(group), mean_std(f"validation_{metric}"), mean_std(f"test_{metric}")])
        lines.extend(table(["Varian", "Label %", "Seed", "Validation", "Test"], summaries))
        lines += [""]
        val = data.pivot(index=["label_percent", "seed"], columns="name", values=f"validation_{metric}")
        comparisons = []
        pairs = [*EFFECTS, *((f"Efek {name}", name, "head_distill") for name in
                            (*LATENT_DISTILL, "ics_distill_90", "ics_distill_50", "ics_distill_ta"))]
        for effect, variant, base in pairs:
            if variant not in val.columns or base not in val.columns:
                continue
            for percent, group in val.groupby(level="label_percent"):
                delta = (sign * (group[variant] - group[base])).dropna()
                if len(delta):
                    comparisons.append([effect, variant, base, percent, f"{(delta > 0).sum()}/{len(delta)}", f"{delta.mean():+.4f}"])
        if comparisons:
            lines.extend(table(["Pertanyaan", "Varian", "Pembanding", "Label %", "Menang validation", "Δ rerata searah perbaikan"], comparisons))
            lines.append("")
    lines.extend(summarize_diagnostic(profile, rows))
    (profile / "ringkasan.md").write_text("\n".join(lines) + "\n")
    print(f"{profile}: {len(rows)} run, ringkasan.md dan skor_per_run.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--expect", choices=("smoke", "core", "all"))
    args = parser.parse_args()
    unique = {}
    for path in sorted(args.profile.glob("results_gpu*.jsonl")):
        for row in read_jsonl(path):
            key = tuple(row[name] for name in KEY)
            previous = unique.get(key)
            if previous and previous["data_and_config"] != row["data_and_config"]:
                raise ValueError(f"Konfigurasi campur untuk {key}")
            unique[key] = row
    rows = list(unique.values())
    if not rows:
        raise SystemExit(f"Belum ada hasil di {args.profile}")
    if args.expect:
        groups = [("nhanes", 10, 42), ("pendigits", 1, 42), ("superconductivity", 30, 42)] if args.expect == "smoke" \
            else [(d, p, s) for d in DATASETS for p in PERCENTS for s in SEEDS]
        names = CORE if args.expect == "smoke" else experiment_names(args.expect)
        expected = {(d, p, s, name) for d, p, s in groups for name in eligible(names, p)}
        allowed = ({(d, p, s, name) for d, p, s in groups for name in eligible(VARIANTS, p)}
                   if args.expect == "core" else expected)
        verify(args.profile, rows, expected, allowed)
    report(args.profile, rows)


if __name__ == "__main__":
    main()
