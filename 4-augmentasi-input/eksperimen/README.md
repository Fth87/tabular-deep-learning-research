# Eksperimen RankUp

Panduan ini menjelaskan mode awal satu split dan lima seed. Mode tambahan **5-fold dengan satu seed** beserta perintah dan ringkasan hasilnya ada di [README utama](../README.md#mode-tambahan-5-fold-satu-seed). Keduanya menggunakan direktori data dan hasil terpisah.

Enam augmentasi dibandingkan pada tiga dataset dan tiga fraksi data berlabel: **1%, 10%, dan 30% dari train pool**. Total ada **54 kombinasi**. Setiap kombinasi menjalankan Optuna sekali, lalu lima training akhir dengan parameter terbaik: total 54 proses tuning dan 270 training akhir.

1. TabCutMix (`tabcutmix`)
2. SCARF (`scarf`)
3. Range Limited Augmentation / RLA (`range_limited`)
4. Gaussian (`gaussian`)
5. Zero Out (`zero_out`)
6. TabCutMix Plus (`tabcutmix_plus`)

Superconductivity memakai RankUp untuk regresi kontinu dan MSE. NHANES memakai RankUp untuk regresi ordinal dengan QWK sebagai metrik utama. Pen Digits memakai adaptasi RankUp-CLS untuk klasifikasi dengan macro-F1; adaptasi ini menggunakan pseudo-label kelas dan tidak memakai ARC ordinal.

## Data dan metrik

NHANES memakai split stratified sekitar 70% train pool, 15% validation, 15% test. Pen Digits mempertahankan test resmi UCI (penulis berbeda); validation diambil secara stratified sebesar 20% dari train resmi. Superconductivity memakai split berdasarkan kelompok vektor fitur identik untuk mencegah fitur yang sama muncul antar train, validation, dan test; proporsi baris mendekati 70/15/15. Validation dan test tetap sama antarfraksi. Fraksi 1%, 10%, dan 30% diambil dari train pool; sisanya menjadi train-unlabeled. Preprocessing di-fit pada train pool dan train-unlabeled disimpan tanpa target. Lima seed mengubah training, bukan split data.

| Dataset | 1% labeled | 10% labeled | 30% labeled | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Superconductivity | 150 | 1.500 | 4.501 | 3.176 | 3.081 |
| NHANES | 36 | 365 | 1.096 | 784 | 784 |
| Pen Digits | 59 | 599 | 1.798 | 1.499 | 3.498 |

Pada NHANES 1%, level paling jarang hanya memiliki satu contoh berlabel. Ini dapat membuat hasil tidak stabil.

Superconductivity menggunakan MSE untuk training, pemilihan trial/checkpoint, dan test. Target `critical_temp` kontinu tetap utuh. Pada TabCutMix dan TabCutMix Plus saja, prediksi dikelompokkan berdasarkan kuantil label train untuk memilih donor; ini adaptasi sementara dan bukan pengubahan target atau pendekatan yang wajib untuk semua regresi. Tidak ada estimasi confidence donor pada adaptasi ini.

Untuk ordinal, loss regresi saat training tetap MSE pada target yang distandardisasi. Pemilihan trial dan checkpoint menggunakan QWK pada validation; metrik ordinal yang dilaporkan pada test adalah QWK. Prediksi kontinu dipetakan ke level ordinal terdekat sebelum QWK dihitung. Pen Digits menggunakan cross-entropy untuk training, macro-F1 untuk pemilihan trial dan checkpoint, lalu melaporkan macro-F1 dan metrik klasifikasi lain pada test.

SCARF mengambil donor independen per fitur. Loss konsistensi ranking dihitung sebagai rata-rata loss bermask atas seluruh pasangan, mengikuti normalisasi RankUp. TabCutMix dan RLA tetap adaptasi untuk unlabeled data. TabCutMix Plus menggunakan hierarchical average-linkage seperti kode sumber; pemilihan grup probabilistik memungkinkan sampel tidak berubah; detailnya ada di [temuan sumber](catatan/temuan_sumber.md). Tidak ada baseline supervised-only dalam rangkaian ini.

## Pengujian

Pemeriksaan split yang ringan bisa dijalankan lokal dengan `uv run python -m unittest discover -s eksperimen/tests -p test_splits.py -v`. Tes yang membutuhkan PyTorch dijalankan di Modal dengan `uv run modal run eksperimen/modal_app.py --action check`; perintah ini menggunakan CPU dan data sintetis kecil, tanpa menjalankan eksperimen penuh.

## Jalankan di Modal

Dari folder `4-augmentasi-input`:

```bash
uv sync
uv run modal run eksperimen/modal_app.py --action prepare --dataset all --label-percent all
uv run modal run --detach eksperimen/modal_app.py --action train --dataset all --method all --label-percent all --run-id ta-001
```

Untuk smoke test, setelah menyiapkan split yang sesuai:

```bash
uv run modal run eksperimen/modal_app.py --action prepare --dataset nhanes --label-percent 1
uv run modal run --detach eksperimen/modal_app.py --action train --dataset nhanes --method range_limited --label-percent 1 --run-id smoke-001 --trials 2 --optuna-epochs 3 --epochs 3
```

`--dataset` menerima `all`, `nhanes`, `pendigits`, atau `superconductivity`; `--label-percent` menerima `all`, `1`, `10`, atau `30`; `--method` menerima `all` atau nama augmentasi di atas. Optuna mencari kekuatan augmentasi weak dan strong menggunakan metrik validation masing-masing dataset; learning rate dan bobot loss memakai nilai tetap, bukan objek tuning. Tuning memakai seed 42 secara default, 40 trial, dan 40 epoch per trial. Setelah parameter terbaik diperoleh, model dilatih ulang dengan seed **42, 123, 456, 789, dan 101112**, maksimal 120 epoch per seed; tiap seed memakai parameter terbaik dan split data yang sama. Ringkasan menyimpan rata-rata dan standar deviasi sampel dari kelima skor train berlabel, validation, dan test. `--seed` mengubah seed tuning, bukan daftar lima seed. Maksimum empat job menggunakan GPU T4 bersamaan dan timeout setiap job adalah 12 jam. Semua job dikirim ke antrean Modal; `--detach` menjaga job tetap berjalan setelah proses lokal selesai. Tunggu pesan `Queued all ... jobs in Modal` sebelum menutup terminal. Job gagal atau melewati timeout dapat dikirim ulang dengan run ID dan pengaturan yang sama; trial dan seed yang sudah tersimpan dilanjutkan. Jangan kirim ulang kombinasi yang sama saat job sebelumnya masih berjalan.

Split disimpan di Volume `experiment-ta-data/label_XX/<dataset>`; menjalankan prepare untuk fraksi dan dataset yang sama akan menimpa split tersebut. Hasil tuning disimpan di `experiment-ta-results/<run-id>/label_XX/tuning/<dataset>`, hasil tiap seed di `label_XX/seed_<seed>/<dataset>`, dan ringkasan lima seed di `label_XX/summary/<dataset>`. Pakai run ID baru untuk setiap pengulangan agar file tidak tertimpa.

## Melanjutkan setelah terputus

Jalankan ulang perintah training dengan run ID dan pengaturan yang sama setelah job sebelumnya berhenti. Snapshot Optuna disimpan dan di-commit tiap trial; trial yang sedang berjalan saat terputus dihitung ulang. Seed yang lengkap dan lolos checksum dilewati, sedangkan seed yang terputus di tengah training dimulai lagi dari awal. Kode, data split, versi library, dan pengaturan harus sama; gunakan run ID baru jika berubah. Lihat [panduan utama](../README.md#melanjutkan-eksperimen-yang-terputus).

## Isi hasil tiap eksperimen

Di folder `tuning/<dataset>/`, file `*_optuna_trials.csv` mencatat tiap trial dan statusnya. `*_best_params.json` mencatat parameter terbaik, rentang pencarian yang dipakai pada trial terbaik, jumlah trial berdasarkan status, seed tuning, dan fraksi label. Default adalah **40 trial per kombinasi**; pruning dapat menghentikan sebagian trial sebelum 40 epoch.

Pada setiap `seed_<seed>/<dataset>/`, file `*_run_config.json` menyimpan argumen awal seperti learning rate, batch size, epoch, bobot loss, cutoff, seed, manifest data, serta versi library. File `*_metrics.json` menyimpan metrik train, validation, dan test; skor train dihitung pada data train **yang berlabel** setelah checkpoint terbaik dipilih. `*_history.csv` menyimpan riwayat loss dan metrik validation tiap epoch; `*_predictions.npz` menyimpan prediksi test. `*_model.pt` menyimpan bobot checkpoint terbaik dan metadata model; untuk regresi dan ordinal juga menyimpan parameter scaler target. Preprocessor fitur dan split tersimpan di Volume data, bukan dalam file model. Split Superconductivity mengelompokkan fitur yang identik; material yang serupa namun tidak identik masih tersebar antar partisi.

## Unduh dan ringkas hasil

```bash
mkdir -p eksperimen/hasil
uv run modal volume get experiment-ta-results ta-001 eksperimen/hasil
uv run python eksperimen/ringkas_hasil.py eksperimen/hasil/ta-001 --output eksperimen/hasil/ta-001-ringkasan.csv
```

Ringkasan menandai kombinasi yang belum tersedia sebagai `missing` atau `partial`; rata-rata dan standar deviasi hanya ditampilkan jika kelima seed tersedia. Hasil ordinal Task 3 lama dipilih berdasarkan MAE, sehingga bukan hasil dari protokol QWK ini.
