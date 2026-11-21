# Task 3 — RankUp ordinal, RankUp-CLS, dan pembanding SSL

Task 3 memakai dua kelompok data. RankUp ordinal diuji pada lima dataset Task 2. Adaptasi RankUp-CLS dan empat metode SSL pembanding (SSLAE, VIME, STUNT, GFTab) diuji pada lima dataset klasifikasi OpenML. Empat metode SSL pembanding juga diuji pada kelima dataset ordinal. CSV klasifikasi disimpan terpisah di `dataset/classification/`.

`modal_app.py` mengatur image/GPU dan memilih runner; `prepare_data.py` menyiapkan split bersama.

## Kode paper yang diklon

| Metode | Kode sumber | Catatan adaptasi |
|---|---|---|
| VIME | [`experiments/vime/paper_repo`](experiments/vime/paper_repo) · [repo penulis](https://github.com/jsyoon0823/VIME) | [`experiments/vime/run.py`](experiments/vime/run.py) memakai split bersama, corruption dari utilitas penulis, dan objective VIME dengan adapter TensorFlow 2. |
| STUNT | [`experiments/stunt/paper_repo`](experiments/stunt/paper_repo) · [repo penulis](https://github.com/jaehyun513/STUNT) | [`experiments/stunt/run.py`](experiments/stunt/run.py) memakai task clustering buatan dari train-unlabeled dan evaluasi prototypical few-shot dari labeled-train. Adapter memakai MiniBatchKMeans agar dapat menangani ukuran fitur dan data Task 3. |
| GFTab | [`experiments/gftab/paper_repo`](experiments/gftab/paper_repo) · [repo penulis](https://github.com/Yoontae6719/Geodesic-Flow-Kernels-for-Semi-Supervised-Learning-on-Mixed-Variable-Tabular-Dataset) | [`experiments/gftab/run.py`](experiments/gftab/run.py) mempertahankan tipe numerik/kategorikal dari CSV split bersama, corruption/GFK upstream, dan tree embedding yang dilatih hanya pada labeled train. |
| SSLAE | [`experiments/sslae/paper_repo`](experiments/sslae/paper_repo) · [repo penulis](https://github.com/sintija-s/sslae) | Runner Task 3 menggunakan API classifier penulis lewat [`experiments/sslae/run.py`](experiments/sslae/run.py) dan `modal_app.py`. |
| ReCom | Belum diklon | URL repository publik penulis belum berhasil diverifikasi. Paper: [DOI](https://doi.org/10.1016/j.ins.2021.12.076). |
| SAWTab | Belum diklon | URL repository publik penulis belum berhasil diverifikasi. Paper: [DOI](https://doi.org/10.1007/978-981-97-2259-4_24). |

> Folder `paper_repo/` tidak ikut git (kode dan lisensi milik penulis aslinya). Unduh dengan `./siapkan_paper_repo.sh`: skrip itu mengambil commit yang sama dengan yang dipakai di eksperimen dan memasang perubahan lokal pada GFTab dan SSLAE dari `paper_repo_patch/`.

Kode runner dan penyesuaian Task 3 disimpan di luar clone. Empat metode SSL membaca split bersama. Untuk target ordinal, runner memperlakukan setiap level sebagai kelas, lalu memetakan prediksi kembali ke nilai berurut untuk menghitung MAE dan metrik ordinal. STUNT memakai evaluasi prototypical few-shot, sehingga protokolnya berbeda dari pembanding lainnya dan RankUp.

## Dataset dan split

Sepuluh dataset terdaftar di [`config/datasets.json`](config/datasets.json). Lima dataset Task 2 ditandai `ordinal_regression`; lima dataset OpenML berada di `dataset/classification/` dan ditandai `classification`. Metadata unduhan ada di `dataset/openml_manifest.json`; unduh ulang dengan `python download_openml_datasets.py`. Rincian jumlah, tipe fitur, dan pemeriksaan ada di [`dataset_compatibility.md`](dataset_compatibility.md).

`prepare_data.py` membuat split stratified dengan seed 42 untuk semua target:

- 70% train pool, 15% validation, dan 15% test.
- Dari train pool, 10% menjadi labeled dan 90% menjadi unlabeled untuk SSL.
- Imputasi, scaling, dan one-hot encoding di-fit pada train saja, termasuk train berlabel dan tanpa label.

Target ordinal tetap numerik dan urutannya dipertahankan; train-unlabeled tidak menyimpan target di file model. Kolom kategori CMC, Bank Marketing, dan Adult ditentukan eksplisit dari metadata OpenML. Nilai fitur Adult yang hilang ditangani di preprocessor train-only.

Setiap dataset menghasilkan `.npz` untuk model, CSV untuk inspeksi, preprocessor, dan `manifest.json` di folder output. Lima dataset ordinal memakai MAE sebagai metrik utama; RMSE tetap dilaporkan sebagai metrik tambahan bersama R², Spearman, dan QWK. Untuk klasifikasi OpenML, metrik utama Accuracy jika seimbang dan Macro-F1 jika tidak seimbang. Sepuluh persen train pool berlabel berarti sekitar 7% seluruh data berlabel.

## Jalankan persiapan dan baseline di Modal

Pasang dependency lokal dengan `pip install -r requirements.txt`, lalu login dengan `modal setup`. CSV dari folder `dataset/` termasuk `dataset/classification/` ikut dimasukkan ke image Modal. Dari folder `task 3`, jalankan:

```bash
modal run modal_app.py --dataset all --action prepare
modal run modal_app.py --dataset all --action sslae
modal run modal_app.py --dataset all --action vime
modal run modal_app.py --dataset all --action stunt
modal run modal_app.py --dataset all --action gftab
modal run modal_app.py --dataset all --action rankup
modal run modal_app.py --dataset classification --action rankup_cls
modal run modal_app.py --dataset all --action all --max-parallel-jobs 4
```

`--action prepare --dataset all` menyiapkan seluruh dataset. Untuk lima dataset ordinal, gunakan `--dataset ordinal`; untuk lima dataset klasifikasi OpenML, gunakan `--dataset classification`. Jalankan `prepare` terlebih dahulu. `--action all --dataset all` menjadwalkan empat baseline SSL pada sepuluh dataset, enam RankUp ordinal pada lima dataset ordinal, dan enam RankUp-CLS pada lima dataset klasifikasi. Job dikirim paralel per batch; `--max-parallel-jobs` mengatur batas GPU aktif dan default-nya 4. Modal dapat mengantrikan job jika kuota GPU akun tidak cukup. Paralel lebih banyak memakai lebih banyak GPU pada waktu yang sama.

Volume `task3-data` menyimpan split hasil persiapan. Hasil metrik dan prediksi ditulis ke Volume `task3-results`. Unduh hasil dengan:

```bash
modal volume get task3-results lev/sslae_metrics.json ./sslae_metrics.json
modal volume get task3-results lev/vime_metrics.json ./vime_metrics.json
modal volume get task3-results adult_income/stunt_metrics.json ./stunt_metrics.json
modal volume get task3-results adult_income/gftab_metrics.json ./gftab_metrics.json
modal volume get task3-results lev/rankup_range_limited_metrics.json ./rankup_metrics.json
```

Runner klasifikasi melaporkan Accuracy, Macro-Precision, Macro-Recall, dan Macro-F1. Metrik utama memakai Accuracy jika rasio kelas terbesar/terkecil pada labeled train <= 1.5, selain itu Macro-F1. RankUp ordinal memilih checkpoint berdasarkan validation MAE dan melaporkan MAE, RMSE, R², Spearman, serta QWK.

## Jalankan persiapan secara lokal

Pasang `requirements.txt`, isi path CSV yang valid, lalu jalankan:

```bash
python prepare_data.py --dataset lev --output ./prepared
```

Semua runner training Modal menggunakan GPU T4 dan berhenti dengan error jika GPU tidak terdeteksi, supaya tidak diam-diam berjalan di CPU. Persiapan split dan operasi preprocessing berbasis pandas/scikit-learn tetap berjalan di CPU. GFTab menjalankan SVD GFK dan pohon XGBoost pada GPU; sebagian pemuatan data dan encoding kategori tetap di CPU sebelum batch dikirim ke GPU. Seluruh runner menulis metrik dan prediksi ke `task3-results`. Training belum dijalankan.

Untuk menjalankan empat pembanding SSL pada lima dataset ordinal yang sama dengan RankUp:

```bash
modal run modal_app.py --dataset ordinal --action sslae
modal run modal_app.py --dataset ordinal --action vime
modal run modal_app.py --dataset ordinal --action stunt
modal run modal_app.py --dataset ordinal --action gftab
```

Keempat model ini tetap classifier: target ordinal diubah menjadi kelas untuk training, lalu prediksi kelas dipetakan kembali ke nilai target aslinya. MAE menjadi metrik utama supaya hasilnya bisa dibandingkan dengan RankUp. Ini adaptasi klasifikasi untuk target ordinal, bukan implementasi native ordinal dari paper. STUNT juga tetap memakai protokol few-shot.

## RankUp-style ordinal regression

RankUp memiliki satu file mandiri untuk setiap eksperimen: `experiments/rankup/tabcutmix.py`, `scarf.py`, `range_limited.py`, `gaussian.py`, `zero_out.py`, dan `tabcutmix_plus.py`. Setiap file memuat augmentasi, model, loss, Optuna, training, evaluasi, dan CLI untuk metodenya. Optuna mencari kekuatan weak/strong dengan objective validation MAE; test baru dipakai setelah parameter terbaik dipilih. Default mengikuti Task 2: 200 trial (SCARF 150; dibatasi menjadi enam pasangan pada dataset empat fitur), 40 epoch per trial, lalu final training maksimal 120 epoch. Atur angkanya lewat `--rankup-trials`, `--rankup-optuna-epochs`, dan `--rankup-epochs`.

Rangkuman metode, cara kerja, keterbatasan, dan sumber paper ada di [`related_work.md`](related_work.md).

Eksperimen dijalankan langsung dari terminal dengan Modal; notebook tidak diperlukan. Untuk menjalankan keenam augmentasi pada lima dataset ordinal:

```bash
modal run modal_app.py --dataset all --action rankup --augmentation all
```

Untuk satu augmentasi, ganti `all` dengan `tabcutmix`, `scarf`, `range_limited`, `gaussian`, `zero_out`, atau `tabcutmix_plus`. Untuk smoke run, misalnya `modal run modal_app.py --dataset lev --action rankup --augmentation scarf --rankup-trials 2 --rankup-optuna-epochs 3 --rankup-epochs 3`. Setiap dataset menyimpan `rankup_<metode>_optuna_trials.csv` dan `rankup_<metode>_best_params.json` selain metrik, prediksi, serta riwayat final training.

## RankUp-style classification

RankUp-CLS menjalankan enam augmentasi yang sama pada dataset klasifikasi. Tiap metode memiliki runner mandiri di `experiments/rankup_classification/`, lengkap dengan augmentasi, training, Optuna, evaluasi, dan CLI. Runner memakai cross-entropy untuk target kelas, lalu pseudo-label ber-confidence tinggi dari weak view untuk melatih strong view. ARC ordinal tidak dipakai karena kelas nominal tidak memiliki urutan. Karena itu hasilnya adalah adaptasi weak-to-strong RankUp untuk klasifikasi, bukan reproduksi RankUp asli. Metrik validation mengikuti Accuracy atau Macro-F1 sesuai keseimbangan kelas.

```bash
modal run modal_app.py --dataset classification --action rankup_cls --augmentation all
```

Untuk satu dataset dan augmentasi, misalnya `modal run modal_app.py --dataset diabetes_pima --action rankup_cls --augmentation scarf --rankup-trials 2 --rankup-optuna-epochs 3 --rankup-epochs 3`. Hasil memakai awalan `rankup_cls_<metode>` di folder dataset pada Volume `task3-results`.
