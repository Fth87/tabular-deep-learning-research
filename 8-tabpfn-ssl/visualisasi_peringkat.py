"""Buat atlas barchart dan CSV peringkat dari hasil eksperimen yang tersimpan."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
STAGE8 = ROOT / "8-tabpfn-ssl"
OUTPUT = STAGE8 / "output"
DATA = ROOT / "5-augmentasi-latent/data"
STAGE4 = ROOT / "4-augmentasi-input/eksperimen/hasil"
DATASETS = {"nhanes": ("NHANES", "qwk"), "pendigits": ("Pen Digits", "macro_f1"),
            "superconductivity": ("Superconductivity", "mse")}
METRICS = {"qwk": "QWK mentah", "qwk_calibrated": "QWK terkalibrasi", "macro_f1": "Macro-F1",
           "mse": "MSE", "mae": "MAE", "rank": "Rerata peringkat"}
PERCENTS = (1, 10, 30)
SEEDS = {42, 123, 456}
SCOPE = "utama"
CAL = "nhanes_terkalibrasi"
COMMON = "rerata_rank_3_seed"
EARLY2 = "tahap_2_fraksi_label"
EARLY3 = "tahap_3_ssl"
PALETTE = {"T2": "#586474", "T3": "#586474", "T4": "#997023", "T5": "#5275B4",
           "T6": "#765A98", "T7/2.5": "#647D4B", "T7/3.5": "#4F6272",
           "T7+8": "#254B5E", "T8": "#008676"}
INK, MUTED, PAPER, RULE = "#142732", "#455B67", "#FFFFFF", "#DFE8EA"
MISSING = "n/a"


def read_jsonl(paths):
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            if line.strip():
                yield path, json.loads(line)


def value(n):
    return float(n) if n not in (None, "") and np.isfinite(float(n)) else None


def add(records, scope, dataset, pct, stage, method, family, metric, seed, val, test, path):
    records.append({
        "scope": scope, "dataset": dataset, "label_percent": pct, "stage": stage, "method": method,
        "family": family, "metric": metric, "seed": seed, "validation": value(val), "test": value(test),
        "source": str(path.relative_to(ROOT)),
    })


def verify_splits():
    for dataset in DATASETS:
        for pct in PERCENTS:
            rel = f"label_{pct:02}/{dataset}"
            for split in ("train_labeled", "train_unlabeled", "validation", "test"):
                file = f"{split}.npz"
                with np.load(DATA / rel / file) as a, np.load(STAGE4 / "data-5seed" / rel / file) as b:
                    if set(a.files) != set(b.files) or any(
                        not np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind in "fc") for key in a
                    ):
                        raise ValueError(f"Split berbeda: {rel}/{file}")


def stage4_records(records):
    e1 = STAGE4 / "ta-5seed-40trial-20261105"
    optuna_seed42 = {}
    for file in sorted(e1.glob("label_*/seed_*/*/*_metrics.json")):
        r = json.loads(file.read_text())
        pct = int(file.parents[2].name.split("_")[1])
        dataset, seed = r["dataset"], r["seed"]
        metric = DATASETS[dataset][1]
        name = r["method"].replace("RankUp-CLS", "RankUp") + " (Optuna)"
        add(records, SCOPE, dataset, pct, "T4", name, "SSL", metric, seed,
            r[f"validation_{metric}"], r[metric], file)
        if seed == 42:
            optuna_seed42[dataset, pct, name] = r[metric]
    # E6 rolls up E1 (seed 42), E2, and the external baselines.
    e6 = STAGE4 / "pembanding/semua.jsonl"
    gibbs = {}
    supervised = "TabM tanpa SSL"
    for file, r in read_jsonl([e6]):
        d, p, seed, name = (r[k] for k in ("dataset", "label_percent", "seed", "name"))
        metric = DATASETS[d][1]
        if name.endswith("(Optuna)"):
            if not np.isclose(r["test"][metric], optuna_seed42[d, p, name], atol=1e-10, rtol=0):
                raise ValueError(f"E1 dan E6 berbeda: {d}, {p}, {name}")
            continue
        family = "Label-only" if "(supervised)" in name or name == supervised else "SSL"
        if name in (supervised, "RankUp + Block-Gibbs (usulan)"):
            gibbs[d, p, seed, name] = (r["validation"].get(metric), r["test"].get(metric))
            continue
        add(records, SCOPE, d, p, "T4", name, family, metric, seed,
            r["validation"].get(metric), r["test"].get(metric), file)
    seen = {}
    from_e7 = [STAGE4 / path for path in ("usulan-protokol/results.jsonl", "usulan-screening/results.jsonl", "usulan-semua/results.jsonl")]
    for file, r in read_jsonl(from_e7):
        d, p, seed, name = (r[k] for k in ("dataset", "label_percent", "seed", "name"))
        if d not in DATASETS or name.startswith("def_") or name == "scarf_w0_s50":
            continue
        metric = DATASETS[d][1]
        name = {"none": supervised, "gibbs_k5_s50": "RankUp + Block-Gibbs (usulan)"}.get(name, f"Block-Gibbs / {name}")
        key = d, p, seed, name
        scores = r["validation"].get(metric), r["test"].get(metric)
        if key in gibbs and not np.allclose(scores, gibbs.pop(key), atol=1e-10, rtol=0):
            raise ValueError(f"Skor E6 dan E7 berbeda: {key}")
        if key in seen:
            if not np.allclose(scores, seen[key], atol=1e-10, rtol=0):
                raise ValueError(f"Duplikat E7 dengan skor berbeda: {key}")
            continue
        seen[key] = scores
        add(records, SCOPE, d, p, "T4", name, "Label-only" if name == supervised else "SSL", metric, seed,
            *scores, file)
    if gibbs:
        raise ValueError("Salinan hasil E6 untuk Block-Gibbs belum cocok semua.")


def family(stage, name, r):
    if stage == "T8":
        return "Label-only" if r["ssl_method"] == "label_only" else "SSL"
    if name in ("supervised", "tabpfn", "head_sup"):
        return "Label-only"
    return "SSL"


def calibrate_teacher(pct, seed, cache):
    root = DATA / f"label_{pct:02}/nhanes"
    with np.load(cache) as source, np.load(root / "validation.npz") as val, np.load(root / "test.npz") as test:
        levels = np.asarray(json.loads((root / "manifest.json").read_text())["target_values"])
        scores = source["t_val"], source["t_test"]
        cutoffs = np.quantile(scores[0], np.cumsum([(val["y"] == level).mean() for level in levels])[:-1])
        ys = val["y"], test["y"]
        raw = tuple(cohen_kappa_score(y, levels[np.abs(p[:, None] - levels).argmin(axis=1)], labels=levels,
                                    weights="quadratic") for y, p in zip(ys, scores, strict=True))
        calibrated = tuple(cohen_kappa_score(y, levels[np.searchsorted(cutoffs, p, side="right")], labels=levels,
                                           weights="quadratic") for y, p in zip(ys, scores, strict=True))
    return raw, calibrated


def main_records():
    verify_splits()
    records = []
    stage4_records(records)
    s7 = ROOT / "7-tabpfn-latent"
    s8 = STAGE8 / "output/hasil_tabpfn_konteks/full"
    sources = {
        "T5": [ROOT / "5-augmentasi-latent/hasil/results.jsonl"],
        "T6": [ROOT / "6-ssr-rcus/hasil/results.jsonl"],
        "T7/2.5": (s7 / "versi 2.5/kaggle_data/hasil").glob("results_gpu*.jsonl"),
        "T7/3.5": (s7 / "versi 3.5/kaggle_data/hasil").glob("results_gpu*.jsonl"),
        "T8": s8.glob("results_gpu*.jsonl"),
    }
    teacher35 = {}
    for stage, paths in sources.items():
        for path, r in read_jsonl(paths):
            d, p, seed, name = (r[key] for key in ("dataset", "label_percent", "seed", "name"))
            metric = DATASETS[d][1]
            val, test = (r[split][metric] for split in ("validation", "test"))
            key = d, p, seed
            if stage == "T7/3.5" and name == "tabpfn":
                teacher35[key] = val, test
                continue
            group = "T7+8" if stage == "T8" and name == "tabpfn" else stage
            if group == "T7+8" and (val, test) != teacher35.pop(key):
                raise ValueError(f"Baseline TabPFN 3.5 tahap 7 dan 8 berbeda: {key}")
            kind = family(stage, name, r)
            add(records, SCOPE, d, p, group, name, kind, metric, seed, val, test, path)
            if d == "nhanes" and stage == "T8":
                add(records, CAL, d, p, group, name, kind, "qwk_calibrated", seed,
                    *(r[s]["qwk_calibrated"] for s in ("validation", "test")), path)
            if d == "nhanes" and stage == "T7/2.5" and name == "tabpfn":
                cache = s7 / f"versi 2.5/kaggle_data/cache/nhanes_{p:02}_s{seed}.npz"
                raw, calibrated = calibrate_teacher(p, seed, cache)
                if not np.allclose(raw, (val, test), atol=1e-10, rtol=0):
                    raise ValueError(f"QWK mentah berbeda dengan cache: {cache}")
                add(records, CAL, d, p, group, name, kind, "qwk_calibrated", seed, *calibrated, cache)
    if teacher35 or not any(r["stage"] == "T8" for r in records):
        raise ValueError("Teacher 3.5 atau hasil tahap 8 belum lengkap.")
    return records


def early_records():
    records = []
    path = ROOT / "2-rankup-ordinal/hasil-tabm/results/rankup_tabm_fraction_results/metrics.csv"
    with path.open() as f:
        for r in csv.DictReader(f):
            name = r["experiment"]
            add(records, EARLY2, r["dataset"], round(100 * float(r["labeled_fraction"])), "T2", name,
                "SSL" if name.startswith("RankUp") else "Label-only", "qwk", int(r["seed"]), None, r["QWK"], path)
    path = ROOT / "3-pembanding-ssl/results/test_validation_scores.csv"
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            metric = "mae" if r["task_type"] == "ordinal_regression" else "macro_f1"
            # The held-out metric is aligned across all methods; validation objectives differ by runner.
            add(records, EARLY3, r["dataset"], 10, "T3", r["method"], "SSL", metric, 42, None, r[f"test_{metric}"], path)
    return records


def aggregate(records):
    groups = defaultdict(list)
    seen = set()
    keys = ("scope", "dataset", "label_percent", "stage", "method", "family", "metric")
    for row in records:
        key = tuple(row[k] for k in keys)
        seed_key = (*key, row["seed"])
        if seed_key in seen:
            raise ValueError(f"Run ganda: {seed_key}")
        seen.add(seed_key)
        groups[key].append(row)
    result = []
    for key, rows in groups.items():
        entry = dict(zip(keys, key, strict=True))
        entry["source"] = rows[0]["source"] + (f" (+{len({r['source'] for r in rows}) - 1})"
                                               if len({r["source"] for r in rows}) > 1 else "")
        for split in ("validation", "test"):
            outcomes = [(r["seed"], r[split]) for r in rows if r[split] is not None]
            vals = [v for _, v in outcomes]
            entry[f"mean_{split}"] = mean(vals) if vals else None
            entry[f"sd_{split}"] = stdev(vals) if len(vals) > 1 else None
            entry[f"n_{split}"] = len(vals)
            entry[f"seeds_{split}"] = ",".join(str(s) for s, _ in sorted(outcomes))
        result.append(entry)
    return result


def is_minimize(metric):
    return metric in {"mse", "mae", "rank"}


def rank(rows):
    if not rows:
        return rows
    reverse = not is_minimize(rows[0]["metric"])
    for split in ("validation", "test"):
        name = f"mean_{split}"
        if any(x["metric"] != rows[0]["metric"] for x in rows):
            raise ValueError("Metrik campur dalam satu ranking")
        values = sorted((round(r[name], 9) for r in rows if r[name] is not None), reverse=reverse)
        first = {}
        for pos, val in enumerate(values, 1):
            first.setdefault(val, pos)
        for row in rows:
            row[f"rank_{split}"] = first[round(row[name], 9)] if row[name] is not None else None
    return sorted(rows, key=lambda r: (r["rank_validation"] is None, r["rank_validation"] or r["rank_test"] or 999,
                                       r["stage"], r["method"]))


def index_conditions(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["scope"], r["dataset"], r["label_percent"]].append(r)
    return {key: rank(items) for key, items in groups.items()}


def rank_common(records):
    cells = aggregate([r for r in records if r["scope"] == SCOPE and r["seed"] in SEEDS])
    by_model = defaultdict(list)
    for r in cells:
        if r["n_validation"] == r["n_test"] == len(SEEDS):
            by_model[r["stage"], r["method"]].append(r)
    expected = set((d, p) for d in DATASETS for p in PERCENTS)
    methods = {key for key, rows in by_model.items() if {(r["dataset"], r["label_percent"]) for r in rows} == expected}
    cells = index_conditions([r for key in methods for r in by_model[key]])
    by_model.clear()
    for rows in cells.values():
        for r in rows:
            by_model[r["stage"], r["method"]].append(r)
    result = []
    for (stage, method), rows in by_model.items():
        record = {"scope": COMMON, "dataset": "3 dataset utama", "label_percent": "", "stage": stage,
                  "method": method, "family": rows[0]["family"], "metric": "rank",
                  "source": "Rerata rank dari 9 kondisi dengan kandidat yang sama"}
        for split in ("validation", "test"):
            vals = [r[f"rank_{split}"] for r in rows]
            record[f"mean_{split}"], record[f"sd_{split}"] = mean(vals), None
            record[f"n_{split}"], record[f"seeds_{split}"] = len(SEEDS), "42,123,456"
        result.append(record)
    return rank(result)


def limits(rows):
    values = [(r[f"mean_{s}"], r[f"sd_{s}"] or 0) for r in rows for s in ("validation", "test")
              if r[f"mean_{s}"] is not None]
    lo = min(0, *(a - b for a, b in values))
    hi = max(0, *(a + b for a, b in values))
    span = hi - lo or 1
    return lo - span * 0.02, hi + span * 0.035


def formatted(x, metric):
    if x is None:
        return MISSING
    return f"{x:.1f}" if metric == "mse" else f"{x:.2f}" if metric == "rank" else f"{x:.4f}"


def legend(fig):
    labels = ["T4 input & benchmark", "T5 RankUp / TabM", "T6 SSR-RCUS", "T7 TabPFN 2.5",
              "T7 TabPFN 3.5", "T7+T8 teacher 3.5", "T8 konteks"]
    keys = ["T4", "T5", "T6", "T7/2.5", "T7/3.5", "T7+8", "T8"]
    for i, (key, label) in enumerate(zip(keys, labels, strict=True)):
        x = 0.042 + i * 0.135
        fig.text(x, 0.042, "■", color=PALETTE[key], size=8)
        fig.text(x + 0.012, 0.042, label, color=MUTED, size=8)


def frame(title, subtitle, page, footer=True):
    fig = plt.figure(figsize=(13.8, 9.6), facecolor=PAPER)
    fig.text(0.041, 0.944, title, fontsize=23, fontweight="bold", color=INK)
    fig.text(0.042, 0.906, subtitle, fontsize=10.2, color=MUTED)
    fig.text(0.96, 0.024, f"{page:02d}", ha="right", fontsize=9, color=MUTED)
    if footer:
        legend(fig)
    return fig


def render(fig, rows, y0, height, xlim, show_validation=True, ranks_only=False):
    metric = rows[0]["metric"]
    n = len(rows)
    region = (0.04, y0, 0.92, height)
    labels = fig.add_axes(region, frameon=False)
    labels.set(xlim=(0, 1), ylim=(n - 0.48, -0.52), xticks=[], yticks=[])
    labels.set_zorder(3)
    labels.patch.set_visible(False)
    labels.text(0.012, 1.028, "TAHAP / VARIAN", transform=labels.transAxes, size=8, weight="bold", color=MUTED)
    labels.text(0.302, 1.028, "SEED", transform=labels.transAxes, size=8, weight="bold", color=MUTED, ha="right")
    for i, r in enumerate(rows):
        color = PALETTE[r["stage"]]
        labels.text(0.012, i, r["stage"], va="center", size=7.6, color=color, weight="bold")
        labels.text(0.061, i, r["method"], va="center", size=8.2, color=INK)
        labels.text(0.302, i, str(r["n_test"] or r["n_validation"]), va="center", ha="right", size=7.8, color=MUTED)
    splits = (("validation", 0.325, 0.215, "VALIDATION"), ("test", 0.655, 0.215, "TEST")) if show_validation         else (("test", 0.325, 0.55, "TEST"),)
    for split, left, width, title in splits:
        ax = fig.add_axes((left, y0, width, height))
        ax.set(xlim=xlim, ylim=(n - 0.48, -0.52), yticks=[])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="x", length=0, pad=3, labelsize=7.5, colors=MUTED)
        ax.grid(axis="x", color=RULE, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvline(0, color="#92A2A8", lw=0.65)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0, 1.028, title, transform=ax.transAxes, fontsize=8, weight="bold", color=MUTED)
        ax.text(1.035, 1.028, "NILAI    RANK", transform=ax.transAxes, fontsize=8, weight="bold", color=MUTED)
        for i, r in enumerate(rows):
            if i % 2 == 0:
                ax.axhspan(i - 0.48, i + 0.48, color="#F0F5F6", zorder=0)
            score, sd, pos = (r[f"{k}_{split}"] for k in ("mean", "sd", "rank"))
            if score is not None:
                ax.barh(i, score, height=0.66, color=PALETTE[r["stage"]], alpha=0.9, zorder=2)
                if sd is not None and not ranks_only:
                    ax.errorbar(score, i, xerr=sd, color=INK, fmt="none", capsize=2, lw=0.65, zorder=3)
            text = formatted(score, metric) + (f"  #{pos:02d}" if pos is not None else "")
            ax.text(1.035, i, text, transform=ax.get_yaxis_transform(), va="center", size=7.8, color=INK,
                    fontfamily="DejaVu Sans Mono", clip_on=False)


def nice_dataset(d):
    return DATASETS[d][0] if d in DATASETS else d.replace("_", " ").title()


def heading_metric(metric):
    return METRICS[metric] + (" · lebih kecil lebih baik" if is_minimize(metric) else " · lebih besar lebih baik")


def pdf_report(path, groups, overall):
    page = 0
    detail_pages = {
        dataset: sum((len(groups[SCOPE, dataset, p]) + 33) // 34 for p in PERCENTS) for dataset in DATASETS
    }
    calibrated_page = 5 + sum(detail_pages.values())
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.unicode_minus": False})

    with PdfPages(path, metadata={"Title": "Peringkat SSL dan augmentasi tabular", "Author": "Riset deep learning"}) as pdf:
        def save(fig):
            nonlocal page
            pdf.savefig(fig)
            plt.close(fig)
            page += 1

        cover = frame("Peringkat SSL dan augmentasi tabular", "Perbandingan validation dan test dari hasil eksperimen yang tersimpan", page + 1)
        cover.text(0.042, 0.853, f"Ringkas 2–4     Atlas 5–{calibrated_page - 1}     "
                   f"QWK terkalibrasi {calibrated_page}–{calibrated_page + 2}     "
                   f"Rerata rank mulai {calibrated_page + 3}", size=10.2, weight="bold", color="#346879")
        cover.text(0.042, 0.795, "Cara membaca", size=15, weight="bold", color=INK)
        instructions = [
            "Mulai dari tiga halaman ringkas untuk melihat varian terpilih di validation per tahap dan kondisi.",
            "Buka atlas lengkap untuk melihat semua baris, termasuk baseline klasik, SSL lain, dan varian ablasi.",
            "Rank #01 adalah rerata skor terbaik dalam kondisi yang sama. Nilai seri berbagi rank.",
            "Urutan baris mengikuti validation. Rank test dilaporkan sebagai hasil pada split terpisah.",
            "Garis pada batang menunjukkan ±1 simpangan baku antar-seed. Kolom SEED menunjukkan jumlah run test.",
        ]
        for i, line in enumerate(instructions):
            cover.text(0.043, 0.749 - i * 0.045, "•  " + line, size=10.5, color=INK)
        cover.text(0.042, 0.480, "Cakupan", size=15, weight="bold", color=INK)
        for i, line in enumerate([
            "Atlas utama memakai QWK mentah (NHANES), macro-F1 (Pen Digits), dan MSE (Superconductivity).",
            "Konfigurasi tahap 4 memakai 1–5 seed. Rank gabungan memakai seed 42/123/456 dan kandidat lengkap di 9 kondisi.",
            "QWK terkalibrasi memakai prediksi yang dapat diperiksa dari tahap 7 dan 8. Guru 3.5 yang identik muncul sekali (T7+8).",
            "Lampiran memuat sweep fraksi label tahap 2 dan benchmark tahap 3 dengan split dan protokol asalnya.",
            "Cakupan tahap 4 meliputi E1, E2, E6, dan E7 pada tiga dataset utama. Pilot validation-only E8 di luar atlas.",
        ]):
            cover.text(0.043, 0.433 - i * 0.040, "•  " + line, size=10.2, color=INK)
        cover.text(0.042, 0.176, "Interpretasi lintas tahap", size=13, weight="bold", color=INK)
        cover.text(0.042, 0.138, "Split utama sama. Backbone, tuning, aturan checkpoint, dan jumlah seed mengikuti eksperimen masing-masing.",
                   size=10.2, color=MUTED)
        cover.text(0.042, 0.112, "Tahap 8 memilih checkpoint NHANES dengan QWK terkalibrasi. Atlas utama memakai QWK mentah untuk perbandingan.",
                   size=10.2, color=MUTED)
        save(cover)

        for dataset, (name, metric) in DATASETS.items():
            fig = frame(f"{name} · Ringkas lintas tahap", heading_metric(metric) +
                        " · Pilihan validation per tahap. * menandai rank test #1 pada atlas lengkap.", page + 1)
            for i, p in enumerate(PERCENTS):
                rows = groups[SCOPE, dataset, p]
                picks = {}
                for row in rows:
                    if row["rank_validation"] is not None:
                        picks.setdefault(row["stage"], row)
                shortlist = [{**row, "method": row["method"] + " *"} if row["rank_test"] == 1 else row
                             for row in picks.values()]
                shortlist.extend({**r, "method": r["method"] + " *"} for r in rows
                                 if r["rank_test"] == 1 and r not in picks.values())
                y0 = 0.64 - i * 0.268
                fig.text(0.042, y0 + 0.183, f"{p}% label", size=11, weight="bold", color=INK)
                render(fig, shortlist, y0, 0.155, limits(rows))
            save(fig)

        for dataset, (name, metric) in DATASETS.items():
            for p in PERCENTS:
                rows = groups[SCOPE, dataset, p]
                bounds = limits(rows)
                for offset in range(0, len(rows), 34):
                    part = rows[offset:offset + 34]
                    fig = frame(f"{name} · {p}% label", f"{heading_metric(metric)} · Varian {offset + 1}–{offset + len(part)}"
                                f" dari {len(rows)} · Rank dihitung atas semua varian pada kondisi ini.", page + 1)
                    height = min(0.77, 0.77 * len(part) / 22)
                    render(fig, part, 0.85 - height, height, bounds)
                    save(fig)

        for p in PERCENTS:
            rows = groups[CAL, "nhanes", p]
            fig = frame(f"NHANES · {p}% label · QWK terkalibrasi",
                        "Batas kelas dari validation · TabPFN 2.5 langsung, teacher 3.5 bersama, dan semua varian tahap 8.",
                        page + 1)
            render(fig, rows, 0.12, 0.71, limits(rows))
            save(fig)

        bounds = limits(overall)
        for offset in range(0, len(overall), 34):
            part = overall[offset:offset + 34]
            fig = frame("Rerata peringkat di 9 kondisi",
                        f"Seed 42/123/456 · {len(overall)} kandidat lengkap · Peringkat kecil lebih baik · "
                        f"Varian {offset + 1}–{offset + len(part)}.", page + 1)
            height = min(0.77, 0.77 * len(part) / 22)
            render(fig, part, 0.85 - height, height, bounds, ranks_only=True)
            save(fig)

        # Earlier tasks used their own datasets and protocols.
        datasets = sorted({d for scope, d, _ in groups if scope == EARLY2})
        for d in datasets:
            ref = groups[EARLY2, d, 100][0]
            fig = frame(f"Tahap 2 · {d}", f"TabM, sweep fraksi label · QWK test · Referensi 100% label {ref['mean_test']:.4f} "
                        f"(±{ref['sd_test']:.4f}, {ref['n_test']} seed).", page + 1, footer=False)
            for i, p in enumerate(PERCENTS):
                rows = groups[EARLY2, d, p]
                y0 = 0.64 - i * 0.268
                fig.text(0.042, y0 + 0.183, f"{p}% label", size=11, weight="bold", color=INK)
                render(fig, rows, y0, 0.155, limits(rows), show_validation=False)
            save(fig)
        datasets = sorted({d for scope, d, _ in groups if scope == EARLY3})
        for offset in range(0, len(datasets), 2):
            fig = frame("Tahap 3 · Benchmark SSL", "10% label pada split tahap 3 · Rank test dihitung per dataset dengan metrik yang sama.",
                        page + 1, footer=False)
            for i, d in enumerate(datasets[offset:offset + 2]):
                rows = groups[EARLY3, d, 10]
                y0 = 0.48 - i * 0.375
                fig.text(0.042, y0 + 0.316, f"{nice_dataset(d)} · {heading_metric(rows[0]['metric'])}",
                         size=11, weight="bold", color=INK)
                render(fig, rows, y0, 0.285, limits(rows), show_validation=False)
            save(fig)
    return page


def main():
    main_runs = main_records()
    groups = index_conditions(aggregate([*main_runs, *early_records()]))
    overall = rank_common(main_runs)
    csv_path = OUTPUT / "peringkat_ssl_dan_augmentasi.csv"
    pdf_path = OUTPUT / "pdf/peringkat_ssl_dan_augmentasi.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = [r for group in groups.values() for r in group] + overall
    columns = ["scope", "dataset", "label_percent", "stage", "method", "family", "metric",
               "rank_validation", "mean_validation", "sd_validation", "n_validation", "seeds_validation",
               "rank_test", "mean_test", "sd_test", "n_test", "seeds_test", "source"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ranked)
    pages = pdf_report(pdf_path, groups, overall)
    print(f"{pages} halaman, {len(ranked)} baris peringkat\n{pdf_path}\n{csv_path}")


if __name__ == "__main__":
    main()
