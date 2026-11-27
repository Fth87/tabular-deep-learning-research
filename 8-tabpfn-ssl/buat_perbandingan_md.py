"""Buat laporan Markdown dengan grafik dari CSV peringkat yang sudah ada."""

from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from visualisasi_peringkat import DATASETS, INK, MUTED, PALETTE, PERCENTS, RULE, heading_metric, is_minimize

OUTPUT = Path(__file__).resolve().parent / "output"
FIGURES = OUTPUT / "grafik_validation_3_seed"
SEEDS = "42,123,456"
PER_GRAPH = 34

operator = {
    "angular": "rotasi angular", "dropout": "dropout", "fnoise": "noise fitur (F-noise)",
    "gaussian": "Gaussian", "range_limited": "acak dalam rentang fitur", "scarf": "SCARF",
    "tabcutmix": "TabCutMix", "tabcutmix_plus": "TabCutMix+", "zero_out": "masking ke nol",
}
gibbs = {
    "cal05": "target jarak 0,05",
    "cal05_lab": "target jarak 0,05 + augmentasi data berlabel",
    "cal05_lr": "jarak 0,05, LR 0,0005",
    "cal05_lr_uw1": "jarak 0,05, LR 0,0005, bobot SSL 1",
    "cal05_lr_uw1_c80": "jarak 0,05, LR 0,0005, bobot SSL 1, cutoff 0,8",
    "cal05_sat": "target jarak 0,05, threshold adaptif",
    "cal05_sat_uw1": "target jarak 0,05, threshold adaptif, bobot SSL 1",
    "cal05_uw1_c80": "target jarak 0,05, bobot SSL 1, cutoff 0,8",
    "cal15_lab": "target jarak 0,15 + augmentasi data berlabel",
    "gibbs_k20_s30": "20 tetangga, 30% fitur",
    "gibbs_k20_s50": "20 tetangga, 50% fitur",
    "gibbs_k50": "50 tetangga, 50% fitur",
    "gibbs_k5_s30": "5 tetangga, 30% fitur",
    "gibbs_k5_s50_lab": "5 tetangga, 50% fitur + augmentasi data berlabel",
    "gibbs_k5_w10_s50": "5 tetangga, weak 10%, strong 50%",
    "gibbs_rho20": "target jarak 0,20 untuk langkah Gibbs",
    "gibbs_t3": "3 langkah Gibbs",
}
head = {
    "tabpfn": "prediksi langsung",
    "head_sup": "MLP supervised",
    "head_distill": "distilasi tanpa augmentasi",
    "head_match": "self-training EMA",
    "head_match_gaussian": "self-training EMA + Gaussian",
    "ics_sup": "MLP supervised + ICS 70% pada data berlabel",
    "ics_match_70": "self-training EMA + ICS 70%",
    "ics_both_70": "distilasi + ICS 70% pada data berlabel dan tanpa label",
    "ics_distill_ta": "distilasi + ICS acak (50–99%)",
    **{f"ics_distill_{p}": f"distilasi + ICS {p}%" for p in (50, 70, 90)},
    **{f"head_{op}": f"distilasi + {label}" for op, label in operator.items()},
}

def nama_metode(row):
    stage, method = row.stage, row.method
    rankup = "RankUp-CLS" if row.dataset == "pendigits" else "RankUp"
    if stage in ("T7/2.5", "T7/3.5", "T7+8", "T8"):
        version = "2.5" if stage == "T7/2.5" else "3.5"
        protokol = {"T7/3.5": " · protokol latent", "T8": " · protokol konteks"}
        shared_head = method.startswith("head_") and method not in (
            "head_zero_out", "head_match", "head_match_gaussian",
        )
        suffix = protokol.get(stage, "") if shared_head else ""
        return f"TabPFN {version} · {head[method]}{suffix}"
    if stage == "T6":
        rcus = "SSR-RCUS-CLS" if row.dataset == "pendigits" else "SSR-RCUS"
        if method == "rankup":
            return f"{rankup} · kontrol SSR-RCUS"
        if method == "rcus":
            return rcus
        if method.startswith("rcus_input_"):
            return f"{rcus} + {operator[method.removeprefix('rcus_input_')]} di input"
        view = "weak + strong" if method.endswith("_weak") else "strong"
        return f"{rcus} + rotasi angular ({view})"
    if stage == "T5":
        if method == "supervised":
            return "TabM supervised"
        if method == "rankup_noaug":
            return f"{rankup} / TabM · tanpa augmentasi"
        if method.startswith("angular"):
            strong = "0,1–0,3" if method.endswith("_s10") else "0,5–0,7" if method.endswith("_s50") else "0,3–0,5"
            weak = ", weak 0,9–0,99" if "_weak90" in method else ", weak 0,7–0,95" if "_weak" in method else ""
            position = ", sebelum ReLU" if "_prerelu" in method else ""
            return f"{rankup} + rotasi angular (cos strong {strong}{weak}{position})"
        position, suffix = method.split("_", 1)
        if position == "matched":
            return f"{rankup} + {operator[suffix]} di latent (cos ≈ 0,4)"
        view = " · weak + strong" if suffix.endswith("_weak") else ""
        place = {"input": "input", "latent": "latent", "both": "input + latent"}[position]
        return f"{rankup} + {operator[suffix.removesuffix('_weak')]} di {place}{view}"
    if method.startswith("Block-Gibbs / "):
        return f"{rankup} + Block-Gibbs · {gibbs[method.split(' / ')[1]]}"
    if method == "RankUp + Block-Gibbs (usulan)":
        return f"{rankup} + Block-Gibbs · 5 tetangga, 50% fitur"
    if method.startswith("RankUp + "):
        op, tuning = method.removeprefix("RankUp + ").split(" (", 1)
        return f"{rankup} + {operator[op]} di input ({tuning}"
    return {
        "TabM tanpa SSL": "TabM supervised · benchmark input",
        "HGB (supervised)": "Histogram Gradient Boosting · supervised",
        "Self-training HGB": "Histogram Gradient Boosting · self-training",
        "Ridge (supervised)": "Regresi ridge · supervised",
        "Logistic (supervised)": "Regresi logistik · supervised",
        "GFTAB": "GFTab",
    }.get(method, method)


def ranked(group):
    return group.sort_values(["rank_validation", "stage", "nama"])


def validation_ranks(group):
    metric = group["metric"].iloc[0]
    if not group["metric"].eq(metric).all():
        raise ValueError("Peringkat harus memakai satu metrik.")
    return group.assign(rank_validation=group["mean_validation"].round(9).rank(
        method="min", ascending=is_minimize(metric),
    ))


def rank_conditions(rows):
    return pd.concat([
        validation_ranks(group)
        for _, group in rows.groupby(["scope", "dataset", "label_percent"], sort=False)
    ])


def mean_rank(rows, conditions):
    identities = ["stage", "method"]
    complete = rows.groupby(identities)["rank_validation"].transform("size").eq(len(conditions))
    pool = rank_conditions(rows[complete])
    means = (pool.groupby(identities)["rank_validation"].mean().rename("mean_validation").reset_index()
             .assign(metric="rank", dataset="3 dataset utama", sd_validation=float("nan")))
    return validation_ranks(means)


def score(value, metric):
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}" if metric in ("mse", "rank") else f"{value:.4f}"


def chart(lines, filename, title, group, bounds_from=None):
    """Gambar skor validation dan peringkat dalam kondisi yang sama."""
    metric = group["metric"].iloc[0]
    reference = group if bounds_from is None else bounds_from
    spread = reference["sd_validation"].fillna(0)
    values = reference["mean_validation"]
    low, high = min(0, (values - spread).min()), max(0, (values + spread).max())
    span = (high - low) or 1.0
    graph_bounds = low - span * 0.025, high + span * 0.035

    labels = [fill(name, width=72, break_long_words=False, break_on_hyphens=False) for name in group["nama"]]
    row_count = len(labels)
    height = 1.15 + 0.39 * row_count
    fig = plt.figure(figsize=(14.5, height), facecolor="white")
    fig.text(.033, 1 - .40 / height, title, size=15, weight="bold", color=INK)
    fig.text(.033, 1 - .65 / height, heading_metric(metric), size=9.5, color=MUTED)

    y0, h = .44 / height, (height - 1.43) / height
    column_header = 1 + .08 / (height - 1.43)
    left = fig.add_axes((.033, y0, .378, h), frameon=False)
    left.set(xlim=(0, 1), ylim=(row_count - .5, -.5), xticks=[], yticks=[])
    left.text(0, column_header, "METODE", transform=left.transAxes, size=8.2, weight="bold", color=MUTED)
    for i, label in enumerate(labels):
        left.text(0, i, label, va="center", size=9, color=INK, linespacing=1.04)

    ax = fig.add_axes((.43, y0, .45, h))
    ax.set(xlim=graph_bounds, ylim=(row_count - .5, -.5), yticks=[])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if metric in ("qwk", "qwk_calibrated", "macro_f1"):
        ax.set_xticks([v for v in ax.get_xticks() if graph_bounds[0] <= v <= min(1, graph_bounds[1])])
    ax.tick_params(axis="x", length=0, pad=4, labelsize=8, colors=MUTED)
    ax.grid(axis="x", color=RULE, linewidth=.7)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#80919a", lw=.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0, column_header, "VALIDATION", transform=ax.transAxes, size=8.2, weight="bold", color=MUTED)
    ax.text(1.025, column_header, "SKOR / RANK", transform=ax.transAxes, size=8.2, weight="bold", color=MUTED)
    for i, row in enumerate(group.itertuples(index=False)):
        if i % 2 == 0:
            ax.axhspan(i - .48, i + .48, color="#f2f5f6", zorder=0)
        ax.barh(i, row.mean_validation, height=.61, color=PALETTE[row.stage], zorder=2)
        if pd.notna(row.sd_validation):
            ax.errorbar(row.mean_validation, i, xerr=row.sd_validation, color=INK,
                        fmt="none", capsize=1.8, lw=.7, zorder=3)
        display = f"{score(row.mean_validation, metric)}  #{int(row.rank_validation)}"
        ax.text(1.025, i, display, transform=ax.get_yaxis_transform(), va="center", clip_on=False,
                family="DejaVu Sans Mono", size=8.5, color=INK)
    rel = f"{FIGURES.name}/{filename}.png"
    fig.savefig(OUTPUT / rel, dpi=145, facecolor="white")
    plt.close(fig)
    lines.extend([f"![{title}. Skor dan rank validation.]({rel})", ""])


def paged_charts(lines, prefix, title, group):
    total = len(group)
    pages = (total + PER_GRAPH - 1) // PER_GRAPH
    for page in range(pages):
        lo, hi = page * total // pages, (page + 1) * total // pages
        description = f"{title} · metode {lo + 1}–{hi} dari {total}"
        chart(lines, f"{prefix}_{page + 1:02}", description, group.iloc[lo:hi], bounds_from=group)


def winners_table(lines, data, conditions):
    lines.extend([
        "| Kondisi | Metode rank #1 | Skor validation |",
        "|---|---|---:|",
    ])
    for dataset, pct in conditions:
        g = data[(data["dataset"] == dataset) & (data["label_percent"] == pct)]
        validation = next(ranked(g).itertuples(index=False))
        val_ties = int((g["rank_validation"] == 1).sum())
        val_tied = f" (+{val_ties - 1} seri)" if val_ties > 1 else ""
        lines.append(f"| {DATASETS[dataset][0]} {pct}% | {validation.nama}{val_tied} | "
                     f"{score(validation.mean_validation, validation.metric)} |")
    lines.append("")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(OUTPUT / "peringkat_ssl_dan_augmentasi.csv")
    data = rank_conditions(data[
        data.scope.isin(["utama", "nhanes_terkalibrasi"])
        & data.n_validation.eq(3)
        & data.seeds_validation.eq(SEEDS)
        & data.mean_validation.notna()
        & ~data.method.str.contains("Optuna", case=False, regex=False)
        & data.stage.ne("T7/2.5")
    ])
    data["nama"] = data.apply(nama_metode, axis=1)
    main = data[data.scope == "utama"]
    cal = data[data.scope == "nhanes_terkalibrasi"]
    conditions = [(dataset, pct) for dataset in DATASETS for pct in PERCENTS]
    overall = mean_rank(main, conditions)
    overall["nama"] = overall.apply(nama_metode, axis=1)
    overall = ranked(overall)
    best_overall = next(overall.itertuples(index=False))
    best_t8 = next(overall[overall.stage == "T8"].itertuples(index=False))
    lines = [
        "# Perbandingan validation metode SSL",
        "",
        "**Tahap 4 sampai 8**, validation dengan seed **42, 123, dan 456** pada split yang sama. "
        "Varian TabPFN memakai versi 3.5. Peringkat dihitung ulang untuk varian non-Optuna.",
        "",
        "[NHANES](#nhanes) · [Pen Digits](#pen-digits) · [Superconductivity](#superconductivity) · "
        "[QWK terkalibrasi](#qwk-terkalibrasi) · [Rerata peringkat](#rerata-peringkat) · "
        "[Seluruh metode](#seluruh-metode)",
        "",
        f"**{best_overall.nama}** berada di rank validation gabungan **#{int(best_overall.rank_validation)}**. "
        f"Varian terbaik dari eksperimen konteks, **{best_t8.nama}**, berada di **#{int(best_t8.rank_validation)}**.",
        "",
        "## Perbandingan hasil",
        "",
        "**NHANES** QWK mentah ↑ · **Pen Digits** macro-F1 ↑ · **Superconductivity** MSE ↓",
        "",
    ]
    winners_table(lines, main, conditions)
    lines.extend([
        "**Grafik** menampilkan rerata dan ±1 simpangan baku dari tiga seed. Hijau menandai eksperimen konteks.",
        "",
    ])
    details = ["## Seluruh metode", ""]
    for dataset, (name, _) in DATASETS.items():
        lines.extend([f"## {name}", ""])
        for pct in PERCENTS:
            group = ranked(main[(main["dataset"] == dataset) & (main["label_percent"] == pct)])
            lines.extend([f"### {name} dengan {pct}% label", ""])
            finalists = group.drop_duplicates("stage")
            chart(lines, f"{dataset}_{pct:02}_ringkas", f"{name} · {pct}% label · ringkasan", finalists)
            details.extend([f"### {name} · {pct}% label ({len(group)} metode)", ""])
            paged_charts(details, f"{dataset}_{pct:02}", f"{name} · {pct}% label", group)
    lines.extend([
        "## QWK terkalibrasi",
        "",
        "Batas kelas berasal dari validation. Grafik memuat varian eksperimen konteks dan prediksi langsung TabPFN 3.5.",
        "",
    ])
    winners_table(lines, cal, [("nhanes", pct) for pct in PERCENTS])
    for pct in PERCENTS:
        chart(lines, f"nhanes_{pct:02}_terkalibrasi", f"NHANES · {pct}% label · QWK terkalibrasi",
              ranked(cal[cal["label_percent"] == pct]))
    lines.extend([
        "## Rerata peringkat",
        "",
        f"{len(overall)} metode lengkap pada sembilan kondisi. Rerata rank lebih kecil lebih baik.",
        "",
    ])
    paged_charts(lines, "rerata_rank", "Rerata peringkat pada sembilan kondisi", overall)
    lines.extend([
        "## Sumber",
        "",
        "[CSV sumber sebelum penyaringan](peringkat_ssl_dan_augmentasi.csv)",
        "",
    ])
    lines.extend(details)
    dest = OUTPUT / "perbandingan_metode_ssl.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    count = sum(line.startswith("![") for line in lines)
    print(f"{dest}\n{count} gambar, {len(data)} baris sumber.")


if __name__ == "__main__":
    main()
