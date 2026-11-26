"""Buat ulang ringkasan tahap 7 dari grid final, prediksi teacher, dan hasil tahap 5–6."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "5-augmentasi-latent/data"
METRICS = {"nhanes": ("qwk", 1), "pendigits": ("macro_f1", 1), "superconductivity": ("mse", -1)}
PERCENTS = (1, 10, 30)
SEEDS = (42, 123, 456)
KEY = ["dataset", "label_percent", "seed", "name"]


def read(paths):
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows).set_index(KEY).sort_index()
    if not frame.index.is_unique:
        raise ValueError("Ada duplikat pada hasil final")
    return frame


def scores(frame, dataset, pct, variant, split, key):
    values = frame.xs((dataset, pct, variant), level=("dataset", "label_percent", "name"))[split]
    return values.map(lambda row: row[key]).reindex(SEEDS).to_numpy()


def fmt(values, key):
    digits = 1 if key == "mse" and np.nanmean(values) > 10 else 3
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def header(cols):
    return "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|"


def line(values):
    return "| " + " | ".join(map(str, values)) + " |"


def calibrated(cache, p, seed, baseline):
    folder = DATA / f"label_{p:02d}/nhanes"
    with np.load(folder / "validation.npz") as val, np.load(folder / "test.npz") as test:
        yv, yt = val["y"], test["y"]
    levels = np.asarray(json.loads((folder / "manifest.json").read_text())["target_values"])
    with np.load(cache / f"nhanes_{p:02d}_s{seed}.npz") as npz:
        pv, pt = npz["t_val"], npz["t_test"]
    cutoffs = np.quantile(pv, np.cumsum([(yv == level).mean() for level in levels])[:-1])
    result = {}
    for split, y, pred in (("validation", yv, pv), ("test", yt, pt)):
        nearest = levels[np.abs(pred[:, None] - levels).argmin(axis=1)]
        raw = cohen_kappa_score(y, nearest, labels=levels, weights="quadratic")
        if not np.isclose(raw, baseline[split]["qwk"]):
            raise ValueError(f"Cache {cache}, {p}%, seed {seed} berbeda dari hasil final {split}")
        idx = np.searchsorted(cutoffs, pred, side="right")
        result[split] = cohen_kappa_score(y, levels[idx], labels=levels, weights="quadratic")
    return result


def main():
    roots = {v: HERE / f"versi {v}/kaggle_data" for v in ("2.5", "3.5")}
    grids = {v: read(sorted(path.glob("hasil/results_gpu*.jsonl"))) for v, path in roots.items()}
    old, new = grids.values()
    expected = pd.MultiIndex.from_product([METRICS, PERCENTS, SEEDS, old.index.unique("name")], names=KEY)
    if len(old) != 270 or not expected.sort_values().equals(old.index) or not old.index.equals(new.index):
        raise ValueError("Kedua grid harus lengkap dan berpasangan")
    s5 = read([ROOT / "5-augmentasi-latent/hasil/results.jsonl"])
    s6 = read([ROOT / "6-ssr-rcus/hasil/results.jsonl"])
    variants = ["tabpfn", "head_sup", "head_distill", "head_gaussian", "head_dropout", "head_fnoise",
                "head_angular", "head_tabcutmix", "head_zero_out", "head_scarf"]
    out = [
        "# Hasil TabPFN 2.5 dan 3.5", "",
        "Masing-masing versi punya 270 run final yang berpasangan di 3 dataset × 3 fraksi label × 3 seed × 10 varian. "
        "Ringkasan memakai `results_gpu0.jsonl` dan `results_gpu1.jsonl`. Run smoke dikeluarkan. "
        "Log 3.5 mencatat `tabpfn==10.2.0` dan `tabpfn-v3.5-20260909.safetensors`. Versi 2.5 mengikuti nama folder karena log lama belum mencatat versi paket atau bobot.",
        "", "## 1. Skor test dan pembanding tahap 5–6", "",
        "Rata-rata ± simpangan baku antar tiga seed. QWK dan macro-F1 lebih besar lebih baik, MSE lebih kecil lebih baik. "
        "Pembanding TabM yang ditampilkan adalah metode yang sudah ditetapkan, dengan split yang sama dan protokol training masing-masing. "
        "Tahap 6 memilih checkpoint regresi/ordinal dengan MAE, sehingga perbedaan angka lintas tahap termasuk efek protokol. "
        "QWK tahap 5–7 di bagian ini memakai pembulatan seperti log asli.", "",
    ]
    controls = [("TabM tahap 5 · rankup_noaug", s5, "rankup_noaug"), ("TabM tahap 5 · angular", s5, "angular"),
                ("TabM tahap 6 · rankup", s6, "rankup"), ("TabM tahap 6 · rcus", s6, "rcus"),
                ("TabM tahap 6 · rcus_angular", s6, "rcus_angular")]
    comparisons = [(f"{v} · {name}", grids[v], name) for name in variants for v in grids] + controls
    for dataset, (key, _) in METRICS.items():
        out += [f"### {dataset} ({key})", "", header(["Versi dan metode", "1%", "10%", "30%"])]
        for label, frame, name in comparisons:
            out.append(line([label, *(fmt(scores(frame, dataset, p, name, "test", key), key) for p in PERCENTS)]))
        out.append("")

    out += ["## 2. Perbandingan berpasangan 3.5 vs 2.5", "",
            "Setiap entri berisi menang/imbang/kalah pada sembilan pasangan fraksi label dan seed. "
            "Ambang imbang absolut 1e-9. Angka validation adalah dasar pemilihan, angka test dilaporkan untuk konfirmasi.", "",
            header(["Varian", *(f"{d} validation" for d in METRICS), *(f"{d} test" for d in METRICS)])]
    for name in variants:
        cells = []
        for split in ("validation", "test"):
            for dataset, (key, sign) in METRICS.items():
                delta = np.concatenate([sign * (scores(new, dataset, p, name, split, key) -
                                                scores(old, dataset, p, name, split, key)) for p in PERCENTS])
                cells.append(f"{(delta > 1e-9).sum()}/{(abs(delta) <= 1e-9).sum()}/{(delta < -1e-9).sum()}")
        out.append(line([name, *cells]))

    out += ["", "## 3. NHANES setelah kalibrasi QWK", "",
            "Ambil batas kelas dari kuantil prediksi validation pada proporsi kelas empiris validation. "
            "Gunakan batas yang sama pada test. QWK mentah dari cache dicocokkan ke setiap baris JSONL sebelum kalibrasi. "
            "Cache tahap 7 menyimpan prediksi teacher `tabpfn`. Kalibrasi head memerlukan prediksi per baris yang akan disimpan tahap 8.", "",
            header(["Versi", "Label", "QWK val mentah", "QWK val kalibrasi", "QWK test mentah",
                    "QWK test kalibrasi", "Spearman test", "MSE test"])]
    for v, frame in grids.items():
        for p in PERCENTS:
            cal = [calibrated(roots[v] / "cache", p, s, frame.loc["nhanes", p, s, "tabpfn"]) for s in SEEDS]
            lookup = lambda split, metric: fmt(scores(frame, "nhanes", p, "tabpfn", split, metric), metric)
            out.append(line([v, p, lookup("validation", "qwk"), fmt([r["validation"] for r in cal], "qwk"),
                             lookup("test", "qwk"), fmt([r["test"] for r in cal], "qwk"),
                             lookup("test", "spearman"), lookup("test", "mse")]))

    out += ["", "## 4. Dampak augmentasi pada 3.5", "",
            "Selisih rerata test dari `head_distill`, memakai QWK mentah, macro-F1 dan MSE sesuai urutan. "
            "Spearman dan MSE juga ditampilkan untuk NHANES agar perubahan ranking terbaca terpisah dari sebaran prediksi.", "",
            header(["Varian", "Label", "Δ QWK", "Δ Spearman", "Δ MSE NHANES", "Δ macro-F1", "Δ MSE Superconductivity"])]
    for p in PERCENTS:
        for method in [v for v in variants if v not in ("head_distill", "head_sup")]:
            values = []
            for d, k in [("nhanes", "qwk"), ("nhanes", "spearman"), ("nhanes", "mse"),
                         ("pendigits", "macro_f1"), ("superconductivity", "mse")]:
                diff = scores(new, d, p, method, "test", k) - scores(new, d, p, "head_distill", "test", k)
                values.append(f"{diff.mean():+.3f}")
            out.append(line([method, p, *values]))

    out += ["", "## 5. Waktu persiapan rata-rata per grup (detik)", "",
            "Satu kali `tabpfn_seconds` per grup dari baris `tabpfn`. Ini waktu persiapan embedding dan teacher per grup, "
            "jadi angka ini tidak mencakup training head dan bukan durasi dinding sesi Kaggle dua GPU.", "",
            header(["Versi", "Dataset", "1%", "10%", "30%"])]
    for v, frame in grids.items():
        for d in METRICS:
            times = [frame.xs((d, p, "tabpfn"), level=("dataset", "label_percent", "name")).tabpfn_seconds.mean()
                     for p in PERCENTS]
            out.append(line([v, d, *(f"{x:.1f}" for x in times)]))
    out += [
        "", "## 6. Posisi terhadap eksperimen sebelumnya", "",
        "- Tahap 4 melaporkan bahwa RankUp dengan augmentasi input menaikkan QWK NHANES 1% dari 0,150 ke 0,251 "
        "dan macro-F1 Pen Digits 1% dari 0,838 ke 0,877 pada eksperimen satu seed tanpa Optuna. "
        "Variasi antar-seed di eksperimen tersebut bisa sebesar selisih antaroperator. "
        "[Ringkasan tahap 4](../4-augmentasi-input/RINGKASAN_SEMUA_EKSPERIMEN.md)",
        "- Tahap 5 memperlihatkan hasil angular yang dekat dengan RankUp tanpa augmentasi. "
        "Pada Superconductivity, teacher 3.5 menurunkan MSE dari 366,5 / 208,8 / 152,8 pada RankUp tanpa augmentasi "
        "ke 318,1 / 163,4 / 116,4. [Hasil tahap 5](../5-augmentasi-latent/hasil/ringkasan.md)",
        "- Di Pen Digits, RCUS tahap 6 unggul atas teacher 3.5 pada 1% dan 10% label "
        "(0,885 vs 0,873 dan 0,973 vs 0,970). Pada 30%, teacher 3.5 mencapai 0,984 dan RCUS 0,978. "
        "[Hasil tahap 6](../6-ssr-rcus/hasil/ringkasan.md)",
        "- Tahap 8 mengulang kontrol dengan TabPFN 3.5 dan menyimpan prediksi setiap head, "
        "sehingga QWK terkalibrasi teacher dan head bisa dibandingkan pada split yang sama. "
        "[Rancangan tahap 8](../8-tabpfn-ssl/dokumen/RANCANGAN.md)",
    ]
    path = HERE / "RINGKASAN_2_5_DAN_3_5.md"
    path.write_text("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
