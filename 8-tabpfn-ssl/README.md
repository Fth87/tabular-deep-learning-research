# Tahap 8 · SSL dan augmentasi dengan TabPFN 3.5

## Mulai dari sini

| Lokasi | Isi |
|---|---|
| [Penjelasan eksperimen](dokumen/EKSPERIMEN_SSL_TABULAR.md) | Rancangan, arsitektur, hasil, dan eksperimen lanjutan |
| [Grafik SSR-RCUS dan TabPFN 3.5](output/perbandingan_ssr_rcus.md) | Perbandingan fokus empat kelompok, validasi 3 seed |
| [Perbandingan lintas metode](output/perbandingan_metode_ssl.md) | Rekap validasi tahap 4–8 |
| [Hasil Kaggle](output/hasil_tabpfn_konteks/full/ringkasan.md) | Rekap 468 run penuh, termasuk ICS, dan diagnostik |
| [Diagram arsitektur](gambar/arsitektur/) | Versi Indonesia serta versi Inggris ringkas dan detail dalam PNG, PDF, dan TikZ |

Output lama untuk eksplorasi dipindahkan ke `arsip/8-tabpfn-ssl/` di root proyek. Cache embedding serta keluaran Kaggle per baris disimpan lokal untuk analisis ulang. Rekap skor dan statistik agregat ada di repositori.

Tujuan penelitian ini adalah mencari **metode SSL yang paling efektif** dan mengukur **nilai tambah augmentasi** pada ordinal, klasifikasi, dan regresi saat label sedikit. Tahap 8 melanjutkan perbandingan RankUp dan SSR-RCUS dari tahap sebelumnya dengan eksperimen terkontrol pada embedding TabPFN 3.5. Rantai eksperimen sejak tahap 1 ada di [rancangan penelitian](dokumen/RANCANGAN.md).

## Yang berubah setelah membaca hasil tahap 7

- **NHANES:** teacher 3.5 dengan kalibrasi validation mencapai QWK test 0,274 / 0,338 / 0,424. Angular dan SCARF diuji ulang dengan QWK terkalibrasi di semua head. QWK mentah, Spearman, dan MSE tetap disimpan.
- **Pen Digits:** SCARF menempati urutan pertama di validation 3.5 pada ketiga fraksi label. SSR-RCUS, termasuk varian input zero-out, masih menjadi pembanding kuat saat label sedikit.
- **Superconductivity:** 2.5 unggul pada 1% label dan 3.5 pada 10%/30%. TabCutMix adalah head 3.5 terbaik di validation pada 1%, sementara angular merusak MSE. Keduanya diikutkan untuk mengukur apakah ICS mengungguli operator yang sesuai maupun yang terlalu kuat.

Hasil lengkap dan protokol aslinya ada di [perbandingan tahap 7 dan tahap 4–6](../7-tabpfn-latent/RINGKASAN_2_5_DAN_3_5.md).

## Perbandingan utama

| Pertanyaan | Perbandingan dengan split, seed, embedding, dan head yang sama |
|---|---|
| Apakah unlabeled membantu? | `head_distill` dan `head_match` masing-masing vs `head_sup`. `tabpfn` menjadi patokan label-only. |
| Metode SSL mana yang lebih baik? | Distilasi teacher TabPFN vs self-training EMA, pertama pada view identik, lalu dengan Gaussian atau ICS yang sama. |
| Apakah ICS memberi nilai tambah? | `ics_distill_70` vs `head_distill`, dan `ics_match_70` vs `head_match`. |
| Apakah ICS lebih baik dari operator sebelumnya? | ICS vs Gaussian, dropout, F-noise, TabCutMix, angular, dan SCARF pada distilasi. Pada EMA, pembandingnya Gaussian. |
| Pengaruh ICS pada split berlabel? | `ics_sup` vs `head_sup`; `ics_both_70` vs `ics_sup` dan `ics_distill_70`. |

Self-training memakai EMA head dengan decay 0,99. Ketiga varian EMA memakai target yang sama jenisnya. Klasifikasi memakai hard pseudo-label saat confidence ≥ 0,95. Regresi dan ordinal memakai target kontinu EMA. Sampel batch dikendalikan per seed, dan semua varian SSL mendapat pool unlabeled yang sama, maksimum 5.000 baris per grup.

## Jalankan di Kaggle

1. Impor [`kaggle_tabpfn_konteks.ipynb`](kaggle_tabpfn_konteks.ipynb) dan tambahkan dataset Kaggle tahap 7 yang berisi `label_01/`, `label_10/`, dan `label_30/`. Aktifkan **GPU T4 x2** dan **Internet On**. Notebook memakai `tabpfn==10.2.0` dan checkpoint 3.5 seperti log tahap 7. [Bobot TabPFN-3.5](https://huggingface.co/Prior-Labs/tabpfn_3_5)
2. Jalankan dengan `RUN_SMOKE = True`. Smoke menjalankan 45 run kecil di NHANES 10%, Pen Digits 1%, dan Superconductivity 30%, seed 42. Anggarannya dua view, maksimum 512 unlabeled, dan 200 langkah. Superconductivity 30% tetap memuat konteks 4.501 label sebelum subsetting.
3. Setelah smoke lulus, set `RUN_FULL = True` untuk **405 run utama** pada 27 grup. Untuk mengukur satu grup dengan anggaran penuh lebih dulu, isi `FULL_ARGS` dengan `["--datasets", "superconductivity", "--percents", "30", "--seeds", "42"]`, lalu kosongkan lagi untuk melanjutkan sisanya. `RUN_ABLATIONS = True` menambahkan 63 run kekuatan konteks (ρ 0,5 / 0,9 / acak) dengan cache yang sama.
4. Jalankan ekspor dan simpan `hasil_tabpfn_konteks.zip` dari **Output**. Untuk melanjutkan dari sesi baru, tambahkan arsip sebagai input dan isi `RESUME_ZIP`. `EXPORT_CACHE = True` ikut menyimpan cache embedding. Gunakan arsip dari revisi notebook ini.

## Baca hasil

- `pilihan_validation.csv` memilih metode per dataset dan fraksi label dari rerata ketiga seed, lalu melaporkan test metode terpilih. Tabel menunjukkan baseline label-only, SSL tanpa augmentasi eksplisit, SSL dengan augmentasi, dan pilihan SSL keseluruhan.
- `ringkasan.md` memuat efek berpasangan tiap perbandingan di atas. QWK terkalibrasi dipilih dan dihitung memakai validation. QWK mentah, Spearman, dan MSE tetap tersimpan bersama prediksi.
- `pembanding_historis.csv` membawa RankUp tahap 5, SSR-RCUS/RankUp tahap 6, dan TabPFN 2.5/3.5 ke laporan Kaggle. Metode historis terpilih memakai validation dari tahap asal. Tabel NHANES lintas tahap memakai QWK mentah di kedua sisi dan mempertahankan perbedaan protokol masing-masing.
- `prediksi/` menyimpan validation/test tiap run. `diagnostik/` menyimpan perubahan prediksi ICS dan rotasi pada cosine yang sama, beserta drift teacher. Diagnostik ini menjelaskan mekanisme dan melengkapi perbandingan akurasi.

Kode ada di [`tabpfn_konteks.py`](tabpfn_konteks.py) dan [`ringkas.py`](ringkas.py). Jalankan `python 8-tabpfn-ssl/buat_notebook.py` dari root repo setelah mengubah sumber agar notebook membawa kode dan referensi terbaru.
