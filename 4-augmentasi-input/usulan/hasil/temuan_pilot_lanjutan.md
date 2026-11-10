# Pilot augmentasi lanjutan — 10 November 2026

Tiga kandidat diuji pada 3 dataset dan 3 fraksi label (27 job, seluruhnya sukses), dengan seed 42, tanpa Optuna, tanpa membaca split test, maksimal 30 epoch dan 40 langkah per epoch. Hasil digabung dengan 9 metode pada pilot sebelumnya; checksum membuktikan split dan data input identik pada setiap kelompok pembandingan. Metrik validation memilih epoch terbaik, sehingga hasil bersifat eksploratif dan tidak mengukur generalisasi yang belum pernah dilihat.

| Dataset | Label | Terbaik dari 12 | Validation | Kandidat hybrid | Kandidat correlated noise | Kandidat gated cutmix |
|---|---:|---|---:|---:|---:|---:|
| NHANES (QWK ↑) | 1% | SCARF | 0,241357 | 0,155928 | 0,155333 | 0,150300 |
| NHANES (QWK ↑) | 10% | TabCutMix | 0,313613 | 0,279601 | 0,310276 | 0,299719 |
| NHANES (QWK ↑) | 30% | local transport | 0,323800 | 0,321395 | 0,318339 | 0,320091 |
| Pen Digits (macro-F1 ↑) | 1% | correlated noise | 0,927742 | 0,924243 | 0,927742 | 0,922552 |
| Pen Digits (macro-F1 ↑) | 10% | RLA | 0,988700 | 0,986726 | 0,987335 | 0,984670 |
| Pen Digits (macro-F1 ↑) | 30% | local transport | 0,995349 | 0,995345 | 0,994706 | 0,995345 |
| Superconductivity (MSE ↓) | 1% | hybrid | 339,213379 | 339,213379 | 341,766486 | 342,951766 |
| Superconductivity (MSE ↓) | 10% | local transport | 200,974790 | 201,524583 | 202,042813 | 201,783689 |
| Superconductivity (MSE ↓) | 30% | hybrid | 191,143179 | 191,143179 | 192,501715 | 192,045835 |

Hybrid memiliki rerata peringkat 4,33 dari 12 metode dan terbaik pada 2/9 kondisi. Namun NHANES 1% dan 10% berkinerja jauh di bawah pemenangnya; ia tidak dapat diklaim stabil atau unggul di semua kondisi. Pada Superconductivity 30%, selisih dengan Gaussian hanya 0,496 MSE (sekitar 0,26%). Pada Pen Digits 1%, correlated noise unggul atas zero-out sebesar 0,000853 macro-F1. Selisih kecil tersebut belum terbukti signifikan. Kesimpulan ini tidak dapat digeneralisasi ke banyak split atau seed dari percobaan tunggal ini.

## Kandidat

- **Correlated noise:** `x' = x + a (x_j - x_k)/sqrt(2)` dengan dua baris dari train pool yang diambil acak dengan pengembalian. Secara ekspektasi, kovarians arah noise mencerminkan kovarians empiris train pool. Ini tidak memastikan label terjaga dan dapat melanggar struktur atau nilai sah fitur.
- **Gated cutmix:** ambil donor terdekat dalam minibatch, lalu ganti masing-masing fitur secara acak dengan probabilitas `a`. Pada ordinal dan klasifikasi, donor dan penerima harus memiliki pseudo-label sama dan confidence minimal 0,8; jika tidak, identitas. Pada regresi belum ada penyaringan berdasarkan prediksi target.
- **Hybrid:** untuk setiap baris, pilih correlated noise atau gated cutmix dengan probabilitas masing-masing 0,5. Pada ordinal/regresi dipasang ke pipeline RankUp proyek; klasifikasi memakai adaptasi pseudo-label multiclass tanpa ARC, bukan RankUp regresi asli.

Kekuatan lemah `a=0,1` dan kuat `a=0,3`, serta parameter training yang sama dengan pilot sebelumnya. Angka strength yang sama menghasilkan perubahan berbeda untuk operator berbeda; skor tidak menunjukkan perbandingan yang sudah dikalibrasi. Pseudo-label bisa salah; confidence ordinal adalah heuristik, bukan probabilitas terkalibrasi. Kandidat adalah kombinasi eksploratif, bukan klaim kebaruan atau jaminan keunggulan. Berbagai operator juga memakai bilangan acak berbeda meski seed sama.

## Audit dataset

- NHANES memiliki 5.223 baris dan 5 tingkat ordinal; tingkat 4 hanya 194 baris (3,71%) pada keseluruhan dataset. QWK dan kualitas sampel berlabel kecil dapat sensitif terhadap pergeseran distribusi dan sedikit prediksi, tetapi hal ini bukan bukti penyebab hasil tertentu.
- Pen Digits memiliki 16 koordinat dan kelas relatif seimbang. Pada validasi 30%, F1 mendekati 1 sehingga ruang perbaikan absolut kecil. Split test resmi dipertahankan dalam konfigurasi.
- Superconductivity memiliki 81 fitur dan target yang bervariasi. Pada train pool split seed 42, 6.088 dari 15.006 baris berada dalam kelompok dengan fitur identik; 1.668 kelompok fitur identik memiliki target berbeda. Median rentang target pada kelompok yang berbeda target 2,31, maksimum 107. Ini bisa mencerminkan informasi yang tidak tersedia atau variasi data; penyebabnya tidak ditetapkan. Tidak ada fitur identik yang menyebrang train–validation atau train–test menurut hash fitur.

Audit deskriptif Superconductivity mengakses semua target train, **termasuk label yang disembunyikan dari model SSL**, tetapi tidak target validation/test. Hal tersebut perlu dinyatakan dalam penelitian karena peneliti telah melihat informasi dari pool unlabeled. Angka audit dan cara mereproduksinya tersedia pada [JSON](audit_superconductivity_train.json) dan [skrip audit](../audit_superconductivity.py).

## Reproduksi dan batas

[Perbandingan CSV 108 hasil](pilot-lanjutan-perbandingan.csv) dan riwayat individual ada di folder hasil lokal. Jalankan `uv run python usulan/bandingkan_pilot.py`. Kode ada pada [augmentasi_lanjutan.py](../augmentasi_lanjutan.py), [pilot_lanjutan.py](../pilot_lanjutan.py), dan [modal_lanjutan.py](../modal_lanjutan.py). Dua tes invariansi lulus pada Modal CPU. Pilot menggunakan T4, maksimal 3 job bersamaan dan timeout 300 detik per job; tidak ada PyTorch yang diinstal lokal.

Sebelum klaim ilmiah, gunakan protokol yang dikunci, validasi dan seed berulang, uji ablation, pembanding yang kuat dan budget sebanding, serta holdout baru yang belum dipakai memilih metode. Hasil test historis telah dilihat sebelumnya. Banyak kandidat telah disaring pada validation yang sama sehingga ada risiko overfitting terhadap validation. Paper relevan dan pembahasan kebaruan ada di [catatan riset sebelumnya](../../eksperimen/riset_augmentasi_usulan.md).
