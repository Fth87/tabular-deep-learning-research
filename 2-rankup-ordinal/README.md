# Tahap 2 — RankUp pada data ordinal (Task 1 dan Task 2, Juli–September 2026)

Menguji RankUp pada **lima dataset ordinal** (Lev, ESL, NHANES, Student Performance, Melbourne Airbnb) dengan beberapa model tabular, objective ordinal, dan augmentasi weak/strong. Hampir semua eksperimen berupa notebook Kaggle yang berdiri sendiri; hasilnya tersimpan di dalam output notebook.

## Isi folder

| Folder | Isi |
|---|---|
| `dataset/` | Lima dataset ordinal (CSV) |
| `kode/` | Notebook sumber: EDA, benchmark model tabular ordinal, RankUp pada semua dataset (MLP, TabM, RealMLP), dan versi CORAL/CORN |
| `hasil/` | Notebook yang sudah dijalankan (dengan output), termasuk benchmark 5 seed dan versi CORAL/CORN |
| `hasil-tabm/` | RankUp + TabM dengan fraksi label 1%, 10%, 30%, beserta notebook visualisasinya |
| `task1_weak_strong_optuna/` | **Task 1**: Optuna mencari kekuatan weak/strong untuk TabCutMix, SCARF, Range-Limited, dan Gaussian (objective: validation MAE) |
| `hasil-optuna/` | Hasil run Task 1 (folder per augmentasi) |
| `gambar/` | Ringkasan visual: profil lima dataset ordinal, parameter weak/strong terbaik dari Optuna, dan grafik konvergensi Optuna |
| `catatan-rapat-2026-08-26.md` | Arahan pembimbing: cara menentukan kekuatan weak/strong, survei paper, jumlah seed |
| `docs/superpowers/` | Spesifikasi dan rencana notebook CORAL/CORN (19 Agustus) |
| `scripts/`, `tests/` | Pembuat notebook CORAL/CORN dan tesnya |

## Eksperimen utama

- **Benchmark model tabular ordinal:** TabM, RealMLP-TD-S, XGBoost, LightGBM, dan CatBoost; baseline 1 seed, tuning Optuna, lalu retrain 5 seed.
- **RankUp pada semua dataset:** MLP, TabM, dan RealMLP dibandingkan dalam empat skenario: Supervised, Fully-Supervised (batas atas), RankUp + augmentasi range-limited, dan RankUp + TabCutMix.
- **CORAL dan CORN:** objective ordinal ditambahkan di samping regresi.
- **Sweep fraksi label (TabM):** 1%, 10%, 30% label dengan empat augmentasi (Range-Limited, SCARF, TabCutMix, Gaussian).

## Menjalankan

Notebook dirancang untuk Kaggle (GPU). Secara lokal, `uv sync` memasang dependensi dasar (`pyproject.toml`). Untuk melihat hasil, buka notebook di `hasil/` dan `hasil-tabm/`; tidak perlu dijalankan ulang.

Lanjut ke [`../3-pembanding-ssl/`](../3-pembanding-ssl/).
