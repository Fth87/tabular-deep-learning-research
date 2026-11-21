# Eksperimen TA: RankUp dan augmentasi data tabular

Eksperimen ini membandingkan enam augmentasi untuk pembelajaran semi-supervised pada tiga jenis tugas: **ordinal, klasifikasi, dan regresi kontinu**. Setiap dataset diuji dengan 1%, 10%, dan 30% label dari train pool. Mode awal menjalankan tuning Optuna dan lima seed pada satu split. Mode tambahan menggunakan 5-fold dengan satu seed per fold; kedua mode tersimpan terpisah.

Notebook [EDA dataset](eda/eda_datasets.ipynb) dijalankan secara lokal. Training eksperimen menggunakan GPU di Modal; dependensi lokal tidak memasang PyTorch.

## Peta folder

| Lokasi | Isi |
|---|---|
| `RINGKASAN_SEMUA_EKSPERIMEN.md` | **Ringkasan lengkap semua eksperimen** (setting, hasil, temuan, keterbatasan) |
| `dataset/`, `config/`, `prepare_data.py` | Data mentah, daftar dataset, dan script persiapan split |
| `eda/` | Notebook EDA |
| `eksperimen/visualisasi_eksperimen.ipynb` | **Ringkasan visual semua hasil** (mulai dari sini) |
| `eksperimen/hasil/gambar/` | Gambar hasil, siap ditaruh di slide |
| `eksperimen/hasil/` | Hasil run: `ringkasan-5seed-40trial.csv` (Optuna + 5 seed), `usulan-semua/` (13 dataset tanpa Optuna), `pembanding/` (metode SSL lain), `log/` |
| `eksperimen/catatan/` | Catatan riset, audit sumber, dan laporan eksplorasi |
| `eksperimen/experiments/`, `eksperimen/tests/` | Kode training dan test |
| `eksperimen/modal_*.py`, `eksperimen/ringkas_*.py` | Script menjalankan eksperimen di Modal dan meringkas hasilnya |
| `../5-augmentasi-latent/` | Lanjutan: augmentasi di ruang latent TabM (pindah ke folder sendiri) |
| `usulan/` | Arsip eksplorasi augmentasi lain (pilot, tidak dipakai di hasil utama) |

## Dataset dan metrik

| Tugas | Dataset | Baris | Fitur | Target | Metrik utama | Arah |
|---|---|---:|---:|---|---|---|
| Ordinal | NHANES | 5.223 | 30 | Lima level ordinal | Quadratic Weighted Kappa (QWK) | Maksimalkan |
| Klasifikasi | Pen Digits | 10.992 | 16 | Digit 0–9 | Macro-F1 | Maksimalkan |
| Regresi kontinu | Superconductivity | 21.263 | 81 | Temperatur kritis `critical_temp` | Mean Squared Error (MSE) | Minimalkan |

- **NHANES:** dataset ordinal Task 2 dari [TOC-UCO](https://www.uco.es/grupos/ayrna/materials/tocuco/). Fitur bernama anonim, sehingga makna tiap fitur belum dapat diinterpretasikan secara substantif. Pada split 1%, level yang paling jarang hanya memiliki satu contoh berlabel.
- **Pen Digits:** [UCI Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits), Alpaydin dan Alimoglu (1996), [DOI 10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K). Sepuluh kelas diberi bobot yang sama dalam macro-F1. Dataset berlisensi CC BY 4.0.
- **Superconductivity:** [UCI Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data), Hamidieh (2018), [DOI 10.24432/C53P47](https://doi.org/10.24432/C53P47). Target tetap kontinu dan dataset berlisensi CC BY 4.0.

Metrik utama menentukan trial Optuna dan checkpoint terbaik berdasarkan **validation**. Training ordinal tetap menggunakan loss MSE untuk regresi; prediksi dibulatkan ke level ordinal terdekat saat menghitung QWK. Training klasifikasi menggunakan cross-entropy, sedangkan regresi menggunakan loss MSE. MSE evaluasi dihitung pada skala target asli.

## Pembagian data

Validation dan test tetap untuk seluruh fraksi label dan lima seed training. Fraksi label dihitung dari train pool, bukan dari keseluruhan dataset. Sisa train pool menjadi data tanpa label.

| Dataset | Train pool | 1% berlabel | 10% berlabel | 30% berlabel | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| NHANES | 3.655 | 36 | 365 | 1.096 | 784 | 784 |
| Pen Digits | 5.995 | 59 | 599 | 1.798 | 1.499 | 3.498 |
| Superconductivity | 15.006 | 150 | 1.500 | 4.501 | 3.176 | 3.081 |

- NHANES menggunakan split stratified sekitar 70% / 15% / 15% untuk train pool / validation / test.
- Pen Digits mempertahankan test resmi UCI, yang berasal dari penulis berbeda. Validation diambil secara stratified dari 20% train resmi.
- Superconductivity memisahkan kelompok baris dengan vektor fitur identik agar tidak muncul pada train pool, validation, dan test sekaligus. Baris identik tetap dipertahankan; material yang mirip tetapi fiturnya tidak identik masih dapat tersebar antarpartisi. Di dalam train pool, baris berfitur identik masih dapat terbagi antara labeled dan unlabeled.

Preprocessing fitur di-fit pada train pool, termasuk bagian tanpa label, lalu diterapkan ke validation dan test. File unlabeled yang disiapkan tidak menyimpan target. Notebook EDA menganalisis seluruh dataset untuk deskripsi; pemilihan model tetap menggunakan train dan validation.

## Rancangan eksperimen

Sebanyak **3 dataset × 6 augmentasi × 3 fraksi label = 54 kombinasi**. Dalam mode awal, setiap kombinasi menjalankan satu studi Optuna dan lima training akhir: total **54 studi tuning dan 270 training akhir**, di luar trial Optuna. Mode 5-fold dijelaskan pada bagian terpisah di bawah.

| Augmentasi | ID metode | Cara yang digunakan dalam eksperimen |
|---|---|---|
| TabCutMix | `tabcutmix` | Menukar sebagian fitur dengan donor dari mini-batch yang memiliki pseudo-label sekelompok. |
| SCARF | `scarf` | Mengganti fitur terpilih dengan nilai donor independen untuk setiap fitur. |
| Range-Limited Augmentation | `range_limited` | Mengambil nilai fitur dari rentang kuantil lokal. |
| Gaussian | `gaussian` | Menambahkan noise Gaussian pada fitur numerik. |
| Zero Out | `zero_out` | Mengganti fitur terpilih dengan nol setelah standardisasi. |
| TabCutMix Plus | `tabcutmix_plus` | Menukar kelompok fitur berdasarkan korelasi dengan donor yang sesuai. |

Pada **NHANES dan Superconductivity**, implementasi mengadaptasi RankUp dengan model tabular dan auxiliary ranking classifier (ARC). Pada **Pen Digits**, `RankUp-CLS` memakai pseudo-label kelas dengan augmentasi weak-to-strong tanpa ARC; pendekatannya lebih dekat ke FixMatch dan bukan RankUp regresi asli. Implementasi ini juga tidak memakai RDA opsional dari paper RankUp.

Operator augmentasi diadaptasi untuk data tanpa label, bukan keseluruhan pipeline dari paper asal. TabCutMix menggunakan pseudo-label untuk mencari donor; jika donor tidak tersedia, sampel tidak ditukar. Pada regresi kontinu, TabCutMix dan TabCutMix Plus memakai kelompok kuantil prediksi untuk memilih donor, tanpa mengubah target kontinu dan tanpa estimasi confidence donor. RLA mengambil sampel uniform dalam rentang; pada fitur diskret, nilai hasilnya bisa menjadi non-diskret. Zero setelah standardisasi mendekati rata-rata train, bukan selalu nilai hilang.

Rangkaian saat ini membandingkan keenam variasi tersebut; belum ada baseline supervised-only atau tanpa augmentasi. Karena itu, hasilnya belum cukup untuk menyimpulkan besarnya peningkatan dibanding baseline tersebut.

## Optuna dan lima seed

Untuk setiap kombinasi dataset, augmentasi, dan fraksi label:

1. Optuna mencari kekuatan augmentasi **weak** dan **strong** berdasarkan metrik validation. Default maksimal **40 trial**, masing-masing maksimal **40 epoch**; pruning dapat menghentikan trial lebih awal. Seed tuning default adalah 42.
2. Pilih parameter terbaik dari studi tersebut. Learning rate, bobot loss, dan parameter training lain menggunakan nilai tetap, bukan objek tuning.
3. Latih ulang dengan seed **42, 123, 456, 789, dan 101112**, maksimal **120 epoch** per seed. Parameter terbaik dan split data yang sama dipakai untuk kelimanya; tiap seed memilih checkpoint berdasarkan validation.
4. Hitung rata-rata dan standar deviasi sampel dari skor lima seed untuk train berlabel, validation, dan test secara terpisah. Variasi ini menggambarkan variasi training pada satu split, bukan variasi antar-split data.

Pengaturan default lain: AdamW, learning rate `0.002`, weight decay `0.0003`, batch size `32`, maksimal `100` langkah per epoch, patience `15`, bobot ranking `0.2`, bobot unlabeled `0.2`, confidence cutoff `0.95`, dan donor confidence `0.8`. Beberapa parameter hanya berlaku pada metode atau tugas yang membutuhkannya. Rentang pencarian dan konfigurasi aktual disimpan bersama hasil tuning dan training.

## Mode tambahan: 5-fold, satu seed

Mode `kfold` mempertahankan test yang sama untuk seluruh fold. Data di luar test dibagi menjadi lima fold: setiap fold bergantian menjadi validation, dan empat fold lain menjadi train pool. Fraksi 1%, 10%, dan 30% dihitung dari train pool masing-masing fold. Preprocessor di-fit ulang pada train pool tiap fold, termasuk data tanpa label; test dan validation tidak dipakai untuk fitting.

NHANES dan Pen Digits menggunakan stratified 5-fold; Pen Digits tetap memakai test resmi. Superconductivity menggunakan GroupKFold berdasarkan vektor fitur identik, dengan test terpisah berdasarkan kelompok fitur. Split test dan pengacakan stratified fold menggunakan seed 42; GroupKFold bersifat deterministik tanpa pengacakan. Seed tuning dan training default adalah 42. Di setiap fold, Optuna berjalan hingga 40 trial berdasarkan metrik validation, lalu model dilatih satu kali dengan parameter terbaik dan seed 42. Total **54 kombinasi × 5 fold = 270 studi Optuna dan 270 training akhir**, di luar trial Optuna.

Untuk menyiapkan semua fold lalu mengirim seluruh job ke Modal:

```bash
uv run modal run eksperimen/modal_app.py --action prepare --protocol kfold --dataset all --label-percent all
uv run modal run --detach eksperimen/modal_app.py --action train --protocol kfold --dataset all --method all --label-percent all --run-id ta-kfold-001
```

Persiapan menulis split ke `experiment-ta-data/kfold/label_XX/fold_XX/<dataset>/`; hasil masuk ke `experiment-ta-results/<run-id>/kfold/label_XX/fold_XX/`. Kedua folder ini terpisah dari mode satu split dan lima seed. Tunggu pesan `Queued all 270 jobs in Modal` sebelum menutup terminal. Default maksimal empat GPU berjalan bersamaan. Gunakan `--fold 1` sampai `--fold 5` untuk satu fold atau `--fold all` untuk semuanya. `--seed` mengatur seed tuning dan training di semua fold; pembagian data memakai seed di `config/datasets.json`.

Hasil tiap fold mencakup metrik train berlabel, validation, dan test. Unduh dan ringkas:

```bash
mkdir -p eksperimen/hasil
uv run modal volume get experiment-ta-results ta-kfold-001 eksperimen/hasil
uv run python eksperimen/ringkas_kfold.py eksperimen/hasil/ta-kfold-001 --output eksperimen/hasil/ta-kfold-001-ringkasan.csv
```

Ringkasan menyajikan skor tiap fold dan rata-rata serta standar deviasi sampel untuk train, validation, dan test jika lima fold lengkap. Skor test berasal dari lima model pada **test yang sama**, sehingga bukan lima test independen. Gunakan validation untuk pemilihan parameter dan perbandingan; jangan memilih fold berdasarkan test. Variasi fold tidak mengukur variasi antar-seed.

Resume bekerja per kombinasi dan fold: gunakan kembali run ID dan pengaturan yang sama setelah job sebelumnya berhenti. Trial yang sudah tersimpan dan seed yang lengkap digunakan kembali; trial atau seed yang terputus di tengah diulang. Jangan jalankan job yang sama secara bersamaan. Mode ini belum melanjutkan training dari epoch terakhir.

## EDA lokal

Notebook [eda/eda_datasets.ipynb](eda/eda_datasets.ipynb) memuat statistik, distribusi target, missing values, duplikasi, dan visualisasi untuk ketiga dataset. Output tersimpan di `eda/hasil/`; rincian dan catatan dataset sebelumnya ada di [README EDA](eda/README.md).

Dari direktori `4-augmentasi-input`:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace eda/eda_datasets.ipynb
```

Di IDE, gunakan interpreter `4-augmentasi-input/.venv/bin/python` sebagai kernel. `uv sync` memasang dependensi EDA dan Modal yang ringan, sedangkan PyTorch dipasang pada image Modal untuk training.

## Menjalankan eksperimen di Modal

Dari direktori `4-augmentasi-input`, siapkan data lalu jalankan seluruh kombinasi:

```bash
uv sync
uv run modal run eksperimen/modal_app.py --action prepare --dataset all --label-percent all
uv run modal run --detach eksperimen/modal_app.py --action train --dataset all --method all --label-percent all --run-id ta-001
```

Gunakan `--dataset nhanes`, `pendigits`, atau `superconductivity`; `--label-percent 1`, `10`, atau `30`; dan `--method` dengan ID pada tabel augmentasi untuk membatasi eksperimen. Pilihan `all` menjalankan seluruh nilai. `--trials`, `--optuna-epochs`, dan `--epochs` dapat mengubah batas default. `--seed` mengubah seed tuning, bukan lima seed training akhir.

Contoh smoke test satu kombinasi, tetap menjalankan lima seed akhir:

```bash
uv run modal run eksperimen/modal_app.py --action prepare --dataset nhanes --label-percent 1
uv run modal run --detach eksperimen/modal_app.py --action train --dataset nhanes --method range_limited --label-percent 1 --run-id smoke-001 --trials 2 --optuna-epochs 3 --epochs 3
```

Training menggunakan GPU T4 dengan maksimal empat job bersamaan dan timeout 12 jam per job. `--detach` membuat Modal melanjutkan semua job yang sudah masuk antrean setelah proses lokal selesai; tunggu pesan `Queued all ... jobs in Modal` sebelum menutup terminal. Jika job gagal atau melewati timeout, periksa log dan jalankan ulang dengan run ID serta pengaturan yang sama; trial Optuna dan seed yang selesai akan digunakan kembali. Gunakan `--run-id` baru untuk menghindari penimpaan hasil. Prepare ulang untuk dataset dan fraksi yang sama menimpa split pada Volume data.

## Melanjutkan eksperimen yang terputus

Setelah job sebelumnya berhenti, jalankan kembali perintah training yang sama dengan `--run-id` dan opsi yang sama. Kamu juga bisa membatasi dataset, metode, dan fraksi untuk mengulang kombinasi tertentu. Jangan jalankan kombinasi yang sama secara bersamaan dengan run ID yang sama: Modal Volume tidak menyediakan penguncian antar-container. Hasil dari versi sebelum fitur resume tidak dapat dilanjutkan otomatis; gunakan run ID baru. Hindari menjalankan prepare ulang sebelum resume karena checksum file split dapat berubah.

Setiap trial Optuna yang selesai menyimpan snapshot studi dan sampler, lalu melakukan commit Volume. Resume melanjutkan sisa trial; trial yang sedang berjalan saat terputus dihitung ulang. Setiap seed yang selesai memiliki penanda dan checksum berkas hasil serta parameter terbaik; seed yang lengkap dan valid dilewati. Seed yang terputus di tengah training dimulai ulang dari awal, bukan dari epoch terakhir. Saat kode, versi library, data split, jumlah trial, epoch, atau seed tuning berubah, gunakan run ID baru agar hasil tidak tercampur. Jangan ubah atau jalankan file snapshot pickle dari sumber yang tidak dipercaya.

Tautan dashboard muncul saat job dikirim. Lihat status dengan `uv run modal app list` dan log dengan `uv run modal app logs <app-id>`. Jika job masih berjalan, tunggu atau hentikan dulu sebelum mengirim ulang kombinasi yang sama. Simpan run ID untuk menemukan hasil dalam Volume.

## Hasil dan reproduksi

Data yang disiapkan tersimpan di Modal Volume `experiment-ta-data/label_XX/<dataset>/`, termasuk split, manifest, dan preprocessor fitur. Hasil disimpan di Volume `experiment-ta-results/<run-id>/`:

| Lokasi | Isi |
|---|---|
| `label_XX/tuning/<dataset>/` | Riwayat trial Optuna, snapshot studi untuk resume, konfigurasi resume, parameter terbaik, rentang pencarian, dan seed tuning. |
| `label_XX/seed_<seed>/<dataset>/` | Konfigurasi awal dan versi library, metrik train berlabel / validation / test, riwayat loss dan metrik validation per epoch, prediksi test, dan checkpoint model terbaik. |
| `label_XX/summary/<dataset>/` | Nilai masing-masing seed serta rata-rata dan standar deviasi untuk train, validation, dan test. |

Preprocessor fitur berada di Volume data, bukan di dalam checkpoint; simpan keduanya untuk reproduksi. Riwayat per epoch mencatat loss dan metrik validation; skor train berlabel dihitung setelah checkpoint terbaik dipilih.

Unduh dan gabungkan ringkasan:

```bash
mkdir -p eksperimen/hasil
uv run modal volume get experiment-ta-results ta-001 eksperimen/hasil
uv run python eksperimen/ringkas_hasil.py eksperimen/hasil/ta-001 --output eksperimen/hasil/ta-001-ringkasan.csv
```

Ringkasan menandai kombinasi belum tersedia sebagai `missing` atau `partial`; rata-rata dan standar deviasi hanya ditampilkan ketika lima seed tersedia. Petunjuk pemeriksaan ringan dan rincian file ada di [README eksperimen](eksperimen/README.md).

## Referensi paper dan kode

| Referensi | Kegunaan dalam eksperimen |
|---|---|
| Huang, Fu, dan Tsao, **RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier**, NeurIPS 2024. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c26a8494fe31695db965ae8b7244b7c1-Paper-Conference.pdf) · [Kode resmi](https://github.com/pm25/Semi-Supervised-Regression) | Dasar RankUp dan ARC untuk regresi serta ordinal. |
| Sohn dkk., **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence**, NeurIPS 2020. [Paper](https://arxiv.org/abs/2001.07685) · [Kode resmi](https://github.com/google-research/fixmatch) | Dasar pseudo-label dan konsistensi weak-to-strong pada adaptasi klasifikasi. |
| Gorishniy dkk., **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling**. [Paper](https://arxiv.org/abs/2410.24210) · [Kode resmi](https://github.com/yandex-research/tabm) | Arsitektur tabular yang digunakan dalam model. |
| Fang dkk., **Understanding and Mitigating Memorization in Diffusion Models for Tabular Data**, ICML 2025. [Paper](https://proceedings.mlr.press/v267/fang25f.html) · [Kode resmi](https://github.com/fangzy96/TabCutMix) | Sumber TabCutMix dan TabCutMix Plus; penerapannya di sini diadaptasi untuk pseudo-label. |
| Bahri dkk., **SCARF: Self-Supervised Contrastive Learning Using Random Feature Corruption**, ICLR 2022. [Paper](https://openreview.net/pdf?id=CuV_qYkmKb3) | Sumber korupsi fitur SCARF. Repo resmi penulis belum terverifikasi; proyek memakai operatornya, bukan keseluruhan pretraining SCARF. |
| Lee dkk., **Range-limited Augmentation for Few-shot Learning in Tabular Data with Comprehensive Benchmark**, KDD 2025. [Paper](https://openreview.net/pdf?id=kTH3bEH6hW) · [Kode resmi FeSTA](https://github.com/kyungeun-lee/festa) | Sumber ide RLA; implementasi ini menggunakan sampling dalam rentang kuantil. |
| GsRCL. [Paper](https://academic.oup.com/bfg/article/23/4/441/7577350) · [Kode penulis](https://github.com/ibrahimsaggaf/GsRCL) | Referensi penggunaan noise Gaussian; eksperimen ini hanya memakai operator Gaussian. |
| Ucar dkk., **SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning**, NeurIPS 2021. [Paper](https://papers.nips.cc/paper/2021/file/9c8661befae6dbcd08304dbf4dcaf0db-Paper.pdf) · [Kode resmi](https://github.com/AstraZeneca/SubTab) | Referensi zero-out; eksperimen ini tidak menjalankan keseluruhan SubTab. |
| Akiba dkk., **Optuna: A Next-generation Hyperparameter Optimization Framework**, KDD 2019. [Paper](https://arxiv.org/abs/1907.10902) · [Kode resmi](https://github.com/optuna/optuna) | Optimasi kekuatan augmentasi. |

[Catatan audit sumber](eksperimen/catatan/temuan_sumber.md) merekam perbandingan awal dengan implementasi asli. Beberapa temuan dalam catatan tersebut sudah diperbaiki; perhatikan keterangan status di awal file saat membacanya.
