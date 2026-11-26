"""Buat ulang notebook Kaggle portabel dan salin helper tahap 5 yang diperlukan."""

import ast
import hashlib
import json
from pathlib import Path
from statistics import mean
from textwrap import dedent

HERE = Path(__file__).resolve().parent
STAGE5 = HERE.parent / "5-augmentasi-latent/kode"
cells = []


def cell(kind, source):
    value = {"cell_type": kind, "id": f"tahap8-{len(cells):02d}", "metadata": {},
             "source": dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    cells.append(value)


def functions(path, names):
    source = path.read_text()
    found = {node.name: ast.get_source_segment(source, node)
             for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    return "\n\n\n".join(found[name] for name in names)


def writefile(name):
    source = (HERE / name).read_text()
    cell("code", f"%%writefile /kaggle/working/{name}\n{source}")


def historical_baselines():
    root = HERE.parent
    stage7 = root / "7-tabpfn-latent"
    series = {
        "Tahap 5 · TabM": ([root / "5-augmentasi-latent/hasil/results.jsonl"],
                           ("supervised", "rankup_noaug"), lambda name: name != "supervised"),
        "Tahap 6 · TabM": ([root / "6-ssr-rcus/hasil/results.jsonl"],
                           ("rankup", "rcus"), lambda name: name.startswith("rcus")),
    }
    for revision in ("2.5", "3.5"):
        paths = sorted((stage7 / f"versi {revision}/kaggle_data/hasil").glob("results_gpu*.jsonl"))
        series[f"Tahap 7 · {revision}"] = (
            paths, ("tabpfn", "head_distill"), lambda name: name not in ("tabpfn", "head_sup", "head_distill"),
        )
    metrics = {"nhanes": ("qwk", 1), "pendigits": ("macro_f1", 1), "superconductivity": ("mse", -1)}
    result, data_hashes = [], []
    for percent in (1, 10, 30):
        for dataset in metrics:
            split_root = STAGE5.parent / "data" / f"label_{percent:02d}" / dataset
            for filename in ("manifest.json", "train_labeled.npz", "train_unlabeled.npz", "validation.npz", "test.npz"):
                with (split_root / filename).open("rb") as file:
                    data_hashes.append(hashlib.file_digest(file, "sha256").hexdigest())
    for label, (paths, controls, candidate) in series.items():
        rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
        for dataset, (metric, sign) in metrics.items():
            for percent in (1, 10, 30):
                groups = {}
                for row in rows:
                    if (row["dataset"], row["label_percent"]) == (dataset, percent):
                        groups.setdefault(row["name"], []).append(row)
                complete = {name: runs for name, runs in groups.items() if sorted(r["seed"] for r in runs) == [42, 123, 456]}
                validation = {name: mean(row["validation"][metric] for row in runs) for name, runs in complete.items()}
                winner = max(sorted(name for name in validation if candidate(name)), key=lambda name: sign * validation[name])
                for name in dict.fromkeys((*controls, winner)):
                    runs = complete[name]
                    result.append({
                        "dataset": dataset, "label_percent": percent, "series": label, "name": name,
                        "picked_by_validation": name == winner,
                        "validation": validation[name], "test": mean(row["test"][metric] for row in runs),
                    })
    return {"data_sha256": hashlib.sha256("".join(data_hashes).encode()).hexdigest(), "rows": result}


def main():
    imports = dedent('''\
        """Helper yang disalin oleh buat_notebook.py dari tahap 5."""

        import math

        import numpy as np
        import torch
        import torch.nn.functional as F
        from scipy.stats import spearmanr
        from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
                                     mean_squared_error, r2_score)
    ''')
    helper = "\n\n\n".join([
        imports.strip(),
        functions(STAGE5 / "experiments/usulan.py", ("load", "scores")),
        functions(STAGE5 / "experiments/rankup/tabcutmix.py", ("same_rank_donors",)),
        functions(STAGE5 / "latent.py", ("rotate", "angular", "head_view", "head_direction", "perturb")),
    ])
    (HERE / "stage8_common.py").write_text(helper + "\n")

    cell("markdown", """
        # Tahap 8 · Augmentasi konteks TabPFN-3.5

        Tambah **dataset Kaggle tahap 7** yang berisi `label_01/`, `label_10/`, `label_30/`.
        Aktifkan **GPU T4 x2** dan **Internet**. Kode ada di notebook. Bobot 3.5 diunduh sekali.

        Tujuan: ukur nilai SSL dari unlabeled, lalu tambahan manfaat masing-masing augmentasi.
        Smoke aktif saat pertama dibuka. Setelah selesai, set `RUN_FULL = True`.
    """)
    cell("code", """
        %%bash
        set -euo pipefail
        nvidia-smi -L
        python -m pip install -q 'tabpfn==10.2.0'
    """)
    cell("code", """
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path
        from zipfile import ZIP_DEFLATED, ZipFile

        WORK = Path("/kaggle/working")
        OUT = WORK / "tahap8"
        os.environ["TABPFN_NO_BROWSER"] = "1"
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
        # Isi jika ada lebih dari satu dataset split di /kaggle/input.
        os.environ["DATA_DIR"] = ""
        # Opsional. Hub model publik dapat diunduh langsung tanpa token Hugging Face.
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("TABPFN_TOKEN")
        except Exception:
            token = None
        if token:
            os.environ["TABPFN_TOKEN"] = token
        del token

        RUN_SMOKE = True
        RUN_FULL = False
        RUN_ABLATIONS = False  # tambah grid kekuatan konteks setelah run utama
        FULL_ARGS = []  # contoh pilot skala penuh: ["--datasets", "superconductivity", "--percents", "30", "--seeds", "42"]
        RESUME_ZIP = ""  # opsional: path arsip tahap8 yang ditambahkan ke Kaggle Input
        EXPORT_CACHE = False  # True untuk membawa embedding cache sesi berikutnya

        if RESUME_ZIP:
            with ZipFile(RESUME_ZIP) as archive:
                for entry in archive.infolist():
                    target = (OUT / entry.filename).resolve()
                    if entry.is_dir() or not target.is_relative_to(OUT.resolve()) or target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_suffix(".restore_tmp")
                    with archive.open(entry) as source, partial.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
                    partial.replace(target)

        def run(file, *args):
            subprocess.run([sys.executable, "-u", str(WORK / file), *map(str, args)], check=True)
    """)
    writefile("stage8_common.py")
    writefile("tabpfn_konteks.py")
    writefile("ringkas.py")
    history = json.dumps(historical_baselines(), ensure_ascii=False, separators=(",", ":"))
    cell("code", f"%%writefile /kaggle/working/stage8_history.json\n{history}")
    cell("markdown", """
        ## 1. Smoke dan timing pada konteks besar

        Tiga grup, satu seed, dua view per setting, maksimum 512 baris unlabeled dan 200 langkah.
        Konteks Superconductivity 30% tetap memakai 4.501 label sebelum subsetting.
        Cek grid, arsip prediksi, kolom QWK terkalibrasi dan cosine angular berpasangan.
    """)
    cell("code", """
        if RUN_SMOKE:
            run("tabpfn_konteks.py", "--out", OUT, "--smoke")
            run("ringkas.py", OUT / "smoke", "--expect", "smoke")
    """)
    cell("markdown", """
        ## 2. Grid penuh

        Grid utama mencakup 405 run dari 27 grup. Grid kekuatan konteks menambah 63 run
        jika `RUN_ABLATIONS = True`. Empat view per setting, maksimal 5.000 baris unlabeled,
        3.000 langkah maksimum, dan satu worker per GPU.

        Ringkasan memilih konfigurasi di validation per dataset, fraksi label, dan keluarga
        eksperimen. Tabel historis memuat RankUp/SSR-RCUS dan hasil TabPFN tahap 7.
    """)
    cell("code", """
        if RUN_FULL:
            selection = "all" if RUN_ABLATIONS else "core"
            run("tabpfn_konteks.py", "--out", OUT, "--names", selection, *FULL_ARGS)
            verify_full = [] if FULL_ARGS else ["--expect", selection]
            run("ringkas.py", OUT / "full", *verify_full)
    """)
    cell("markdown", """
        ## 3. Simpan hasil

        Arsip memuat JSONL, prediksi validation/test, ringkasan dan diagnostik.
        `EXPORT_CACHE` menambahkan embedding cache. Simpan arsip dari tab Output sebelum sesi berakhir.
    """)
    cell("code", """
        export = WORK / "hasil_tabpfn_konteks.zip"
        paths = [p for p in OUT.rglob("*") if p.is_file() and p.relative_to(OUT).parts[0] in ("smoke", "full")
                 and not p.name.startswith(".") and p.suffix not in (".tmp", ".restore_tmp")
                 and (EXPORT_CACHE or "cache" not in p.relative_to(OUT).parts)]
        if paths:
            with ZipFile(export, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
                for path in sorted(paths):
                    archive.write(path, path.relative_to(OUT))
            from IPython.display import FileLink, display
            print(f"{export} ({export.stat().st_size / 2**20:.1f} MiB)")
            display(FileLink(export.name))
    """)
    notebook = {"cells": cells, "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "file_extension": ".py"},
    }, "nbformat": 4, "nbformat_minor": 5}
    path = HERE / "kaggle_tabpfn_konteks.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
