# Audit kompatibilitas dataset Task 3

Task 3 memakai lima dataset ordinal Task 2 dan lima dataset klasifikasi OpenML. RankUp ordinal dan empat metode SSL pembanding dijalankan pada dataset ordinal; RankUp-CLS dan empat metode SSL pembanding dijalankan pada dataset klasifikasi OpenML. Split stratified memakai seed 42: 70% train pool, 15% validation, dan 15% test. Sepuluh persen train pool menjadi labeled, sisanya unlabeled. Preprocessor hanya di-fit pada train pool. Target ordinal dipertahankan berurutan untuk evaluasi.

## Dataset

| Dataset | Task | Baris × fitur mentah | Nilai target | Numerik / kategorikal | Labeled / unlabeled / val / test | Fitur encoded |
|---|---|---:|---|---:|---:|---:|
| LEV (Task 2) | Ordinal regression | 1,000 × 4 | 5 tingkat | 4 / 0 | 70 / 630 / 150 / 150 | 4 |
| NHANES (Task 2) | Ordinal regression | 5,223 × 30 | 5 tingkat | 30 / 0 | 365 / 3,290 / 784 / 784 | 30 |
| ESL (Task 2) | Ordinal regression | 488 × 4 | 6 tingkat | 4 / 0 | 34 / 306 / 74 / 74 | 4 |
| Student Performance (Task 2) | Ordinal regression | 662 × 43 | 8 tingkat | 43 / 0 | 46 / 416 / 100 / 100 | 43 |
| Melbourne Airbnb (Task 2) | Ordinal regression | 20,036 × 48 | 10 tingkat | 48 / 0 | 1,402 / 12,622 / 3,006 / 3,006 | 48 |
| [Diabetes / Pima (OpenML 37)](https://www.openml.org/d/37) | Classification | 768 × 8 | 2 kelas | 8 / 0 | 53 / 483 / 116 / 116 | 8 |
| [CMC (OpenML 23)](https://www.openml.org/d/23) | Classification | 1,473 × 9 | 3 kelas | 2 / 7 | 103 / 928 / 221 / 221 | 24 |
| [Satimage (OpenML 182)](https://www.openml.org/d/182) | Classification | 6,430 × 36 | 6 kelas | 36 / 0 | 450 / 4,050 / 965 / 965 | 36 |
| [Bank Marketing (OpenML 1461)](https://www.openml.org/d/1461) | Classification | 45,211 × 16 | 2 kelas | 7 / 9 | 3,164 / 28,483 / 6,782 / 6,782 | 51 |
| [Adult / Income (OpenML 1590)](https://www.openml.org/d/1590) | Classification | 48,842 × 14 | 2 kelas | 6 / 8 | 3,418 / 30,770 / 7,327 / 7,327 | 105 |

Bank Marketing OpenML 1461 memiliki 7 fitur numerik dan 9 kategorikal menurut tipe kolom yang diunduh. Adult memiliki 6,465 sel fitur kosong; preprocessor mengimputasinya dengan median untuk numerik dan modus untuk kategori. CMC, Bank, dan Adult mempertahankan tipe kategori pada alur GFTab.

## Runner dan metrik

| Runner | Dataset yang didukung | Metode / evaluasi |
|---|---|---|
| `experiments/rankup/{tabcutmix,scarf,range_limited,gaussian,zero_out,tabcutmix_plus}.py` | Lima dataset Task 2 | Enam file eksperimen mandiri untuk regresi ordinal TabM dengan pairwise ranking RankUp. Metrik utama MAE; tambahan RMSE, R², Spearman, dan QWK. |
| `experiments/rankup_classification/{tabcutmix,scarf,range_limited,gaussian,zero_out,tabcutmix_plus}.py` | Lima dataset klasifikasi OpenML | Enam runner independen untuk adaptasi weak-to-strong dengan pseudo-label multiclass; ARC ordinal tidak digunakan. Tiap file memuat augmentasi, training, Optuna, dan evaluasi sendiri. |
| `experiments/sslae/run.py`, `experiments/vime/run.py` | Lima ordinal Task 2 dan lima klasifikasi OpenML | Untuk target ordinal, level dijadikan kelas berurutan dan prediksi dikembalikan ke skala target; metrik utama MAE. Klasifikasi memakai Accuracy atau Macro-F1 sesuai imbalance. |
| `experiments/stunt/run.py` | Lima ordinal Task 2 dan lima klasifikasi OpenML | Self-generated task dan prototypical few-shot. Untuk ordinal, checkpoint dipilih berdasarkan validation MAE; protokol few-shot berbeda dari RankUp dan SSL lainnya. |
| `experiments/gftab/run.py` | Lima ordinal Task 2 dan lima klasifikasi OpenML | GFK dan tree embedding mixed-variable. Target ordinal diperlakukan sebagai kelas berurutan dan dinilai dengan MAE; membutuhkan GPU T4. |

Untuk membandingkan pada dataset ordinal, SSLAE, VIME, STUNT, dan GFTab mempelajari level target sebagai kelas. Prediksi kelas dipetakan kembali ke nilai ordinal asli untuk menghitung MAE, RMSE, R², Spearman, dan QWK. STUNT tetap memakai protokol few-shot. CSV OpenML ada di `dataset/classification/`, dengan rincian sumber pada `dataset/openml_manifest.json`. Split dan preprocessing dibuat ulang ke Volume Modal lewat aksi `prepare`.
