# Ringkasan lengkap semua eksperimen TA

**RankUp + augmentasi data tabular, label 1% / 10% / 30%**

Dokumen ini merangkum seluruh eksperimen di folder `4-augmentasi-input` dalam satu tempat: tujuan, setting, hasil dengan nilai asli, temuan, keterbatasan, dan lokasi file. Semua angka dihitung ulang dari file hasil pada 17 November 2026. Kalau angka di dokumen lain berbeda, file hasil yang dijadikan acuan.

---

## Daftar isi

1. [Gambaran besar](#1-gambaran-besar)
2. [Dataset dan metrik](#2-dataset-dan-metrik)
3. [Model, training, dan augmentasi](#3-model-training-dan-augmentasi)
4. [Peta semua eksperimen](#4-peta-semua-eksperimen)
5. [E1 — Optuna 40 trial + 5 seed (eksperimen utama)](#5-e1--optuna-40-trial--5-seed-eksperimen-utama)
6. [E2 — Tanpa Optuna, 1 seed, 3 dataset utama](#6-e2--tanpa-optuna-1-seed-3-dataset-utama)
7. [E3 — Optuna vs tanpa Optuna](#7-e3--optuna-vs-tanpa-optuna)
8. [E4 — Temuan: variasi seed vs perbedaan augmentasi](#8-e4--temuan-variasi-seed-vs-perbedaan-augmentasi)
9. [E5 — 13 dataset, tanpa Optuna, 1 seed](#9-e5--13-dataset-tanpa-optuna-1-seed)
10. [E6 — RankUp vs metode supervised dan SSL lain](#10-e6--rankup-vs-metode-supervised-dan-ssl-lain)
11. [E7 — Eksplorasi augmentasi k-NN Block-Gibbs](#11-e7--eksplorasi-augmentasi-k-nn-block-gibbs)
12. [E8 — Eksplorasi augmentasi lain (folder `usulan/`)](#12-e8--eksplorasi-augmentasi-lain-folder-usulan)
13. [E9 — Mode 5-fold (dirancang)](#13-e9--mode-5-fold-dirancang)
14. [E10 — Augmentasi latent (dipindah ke 5-augmentasi-latent)](#14-e10--augmentasi-latent-dipindah-ke-experiment-latent)
15. [Kesimpulan lintas eksperimen](#15-kesimpulan-lintas-eksperimen)
16. [Keterbatasan dan ancaman validitas](#16-keterbatasan-dan-ancaman-validitas)
17. [Langkah berikutnya](#17-langkah-berikutnya)
18. [Peta file dan reproduksi](#18-peta-file-dan-reproduksi)
19. [Glosarium](#19-glosarium)

---

## 1. Gambaran besar

**Pertanyaan riset.** Pada semi-supervised learning (SSL) untuk data tabular dengan metode RankUp, augmentasi mana yang paling baik untuk membuat "view strong" dari data tanpa label? Apakah jawabannya berbeda untuk tugas ordinal, klasifikasi, dan regresi, serta untuk jumlah label yang sangat sedikit (1%) sampai cukup (30%)?

**Inti RankUp (disederhanakan).**

```
Data berlabel   ──► model ──► loss supervised (MSE / cross-entropy) + loss ranking berlabel
Data tanpa label:
   view weak   (x asli)       ──► model ──► "kunci jawaban" (pseudo-label / urutan pasangan)
   view strong (x diaugmentasi) ──► model ──► harus konsisten dengan kunci jawaban
Total loss = supervised + 0,2 × konsistensi (hanya pasangan yang yakin ≥ 0,95)
```

Augmentasi menentukan seberapa dan ke arah mana view strong berbeda dari aslinya, jadi augmentasi adalah variabel yang dibandingkan di TA ini.

**Jawaban singkat dari semua eksperimen:**

1. **Tidak ada satu augmentasi yang selalu terbaik.** Pemenangnya berganti antar dataset, persentase label, dan antara validation dan test.
2. **RankUp dengan augmentasi umumnya lebih baik dari Tanpa RankUp**, terutama di label sedikit (NHANES 1%: QWK 0,150 → 0,251; Pen Digits 1%: F1 0,838 → 0,877, tanpa Optuna).
3. **Variasi seed bisa sebesar atau lebih besar dari perbedaan antar augmentasi.** Di Superconductivity dan NHANES 30%, perbedaan antar augmentasi tidak signifikan (ANOVA). Di Pen Digits dan NHANES 1%, augmentasi berpengaruh signifikan.
4. **Optuna tidak selalu membantu**: lebih baik di 33 dari 54 pasangan, dan di NHANES 30% semuanya justru lebih buruk.
5. **Banyak kondisi "tidak sensitif" terhadap augmentasi**, karena checkpoint terbaik dipilih di epoch 1–4, sebelum SSL sempat bekerja.

---

## 2. Dataset dan metrik

### 2.1 Tiga dataset utama

| Tugas | Dataset | Baris | Fitur | Target | Metrik utama | Arah |
|---|---|---:|---:|---|---|---|
| Ordinal | NHANES ([TOC-UCO](https://www.uco.es/grupos/ayrna/materials/tocuco/)) | 5.223 | 30 | 5 level ordinal | QWK | ↑ makin besar makin baik |
| Klasifikasi | Pen Digits ([UCI](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)) | 10.992 | 16 | Digit 0–9 (10 kelas) | Macro F1 | ↑ |
| Regresi | Superconductivity ([UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data)) | 21.263 | 81 | Suhu kritis `critical_temp` (kontinu) | MSE | ↓ makin kecil makin baik |

Hasil EDA (`eda/eda_datasets.ipynb`):
- Tidak ada missing value maupun fitur konstan di ketiga dataset.
- Superconductivity punya 66 baris duplikat penuh dan 6.093 baris berulang menurut seluruh fitur.
- Fitur NHANES bernama anonim, jadi maknanya tidak bisa diinterpretasikan.

### 2.2 Pembagian data

| Dataset | Train pool | 1% berlabel | 10% berlabel | 30% berlabel | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| NHANES | 3.655 | 36 | 365 | 1.096 | 784 | 784 |
| Pen Digits | 5.995 | 59 | 599 | 1.798 | 1.499 | 3.498 |
| Superconductivity | 15.006 | 150 | 1.500 | 4.501 | 3.176 | 3.081 |

- Persentase label dihitung dari **train pool**. Sisanya menjadi data tanpa label (targetnya tidak disimpan).
- **Validation dan test sama** untuk semua persentase label dan semua seed.
- Cara split per dataset:
  - **NHANES:** stratified ±70/15/15.
  - **Pen Digits:** memakai test resmi UCI (penulis berbeda); validation diambil stratified 20% dari train resmi.
  - **Superconductivity:** split per kelompok vektor fitur identik, supaya fitur yang sama tidak muncul di train dan test sekaligus.
- Preprocessing di-fit pada train pool (berlabel + tanpa label), lalu diterapkan ke validation dan test.
- **NHANES 1% sangat sedikit:** distribusi label `{0: 4, 1: 10, 2: 14, 3: 7, 4: 1}`. Level 4 hanya punya **1 contoh berlabel**.

### 2.3 Metrik

| Metrik | Dipakai untuk | Cara membaca |
|---|---|---|
| **QWK** (Quadratic Weighted Kappa) | NHANES | Kesepakatan prediksi dengan label asli. Salah jauh dihukum lebih berat dari salah dekat. 1 = sempurna, 0 = setara tebakan acak. Prediksi kontinu dibulatkan ke level terdekat dulu. |
| **Macro F1** | Pen Digits | Rata-rata F1 dari 10 kelas, semua kelas berbobot sama. 1 = sempurna. |
| **MSE** | Superconductivity | Rata-rata kuadrat selisih prediksi dan suhu asli (dalam satuan K²). Makin kecil makin baik. |

Metrik utama juga dipakai untuk **memilih trial Optuna dan checkpoint terbaik, yang selalu dihitung di validation**. Metrik tambahan yang tersimpan adalah MAE, R², Spearman, dan akurasi.

### 2.4 Sepuluh dataset tambahan (khusus E5)

| Tugas | Dataset |
|---|---|
| Ordinal | LEV, ESL, Student Performance, Melbourne Airbnb |
| Klasifikasi | Satimage, Diabetes Pima, CMC, Bank Marketing, Adult Income |
| Regresi | Grid Stability (target `stab`; `stabf` dan `p1` dibuang karena bocor / deterministik) |

LEV, ESL, dan Student Performance tidak dipakai di label 1% karena 3–7 label tidak cukup untuk mewakili semua level. Totalnya **36 kondisi**.

---

## 3. Model, training, dan augmentasi

### 3.1 Model

- **Backbone:** TabM (`TabM.make`, default 3 blok × 512, 32 submodel paralel, dropout 0,1).
- **NHANES dan Superconductivity:** RankUp = head regresi + **Auxiliary Ranking Classifier (ARC)**, yang memprediksi urutan pasangan sampel.
- **Pen Digits:** **RankUp-CLS**, adaptasi gaya FixMatch: pseudo-label kelas dan konsistensi weak→strong, **tanpa ARC**. Ini bukan RankUp regresi asli dan perlu disebut sebagai adaptasi.
- **"Tanpa RankUp"** (`none`): model yang sama dengan bobot loss unlabeled = 0. Untuk ordinal dan regresi, loss ranking pada data berlabel tetap aktif. Jadi ini bukan baseline yang sepenuhnya polos.

### 3.2 Setting training default (sama di semua eksperimen kecuali disebut lain)

| Parameter | Nilai |
|---|---|
| Optimizer | AdamW, lr 0,002, weight decay 0,0003 |
| Batch | 32 berlabel + 32 tanpa label |
| Langkah per epoch | maks. 100 |
| Epoch | maks. 120 (trial Optuna: maks. 40) |
| Early stopping | patience 15, berdasarkan metrik utama validation |
| Bobot ranking berlabel | 0,2 |
| Bobot loss unlabeled | 0,2 |
| Confidence cutoff pseudo-label | 0,95 |
| Donor confidence (TabCutMix) | 0,8 |
| Infrastruktur | Modal, GPU T4 (benchmark: 6,5× lebih cepat dari CPU 4-core, sehingga lebih murah per run) |

### 3.3 Enam augmentasi yang dibandingkan

| Augmentasi | ID | Cara kerja di eksperimen ini | Sumber |
|---|---|---|---|
| TabCutMix | `tabcutmix` | Menukar sebagian fitur dengan donor dari mini-batch yang pseudo-label-nya sama. Di regresi, donor dipilih dari kelompok kuantil prediksi. | Fang dkk., ICML 2025 |
| TabCutMix+ | `tabcutmix_plus` | Seperti TabCutMix, tapi yang ditukar adalah **kelompok fitur yang berkorelasi** (hierarchical average-linkage). | Fang dkk., ICML 2025 |
| SCARF | `scarf` | Mengganti fitur terpilih dengan nilai dari baris acak lain, dengan donor independen per fitur. | Bahri dkk., ICLR 2022 |
| Range-limited (RLA) | `range_limited` | Mengambil nilai acak (uniform) dalam rentang kuantil lokal fitur. | Lee dkk., KDD 2025 (FeSTA) |
| Gaussian | `gaussian` | Menambah noise Gaussian pada fitur (yang sudah distandardisasi). | Baseline umum |
| Zero-out | `zero_out` | Mengganti fitur terpilih dengan 0 (setelah standardisasi ≈ nilai rata-rata). | Baseline umum |

`strength` berarti proporsi fitur yang diubah (untuk Gaussian: σ). View weak dan view strong masing-masing punya strength sendiri.

---

## 4. Peta semua eksperimen

| ID | Eksperimen | Optuna? | Seed | Dataset | Status | Hasil |
|---|---|---|---|---|---|---|
| **E1** | 6 augmentasi, tuning strength | Ya, 40 trial | 5 (42, 123, 456, 789, 101112) | 3 utama × 3 % | Selesai, 270 training akhir | `eksperimen/hasil/ringkasan-5seed-40trial.csv`, `ta-5seed-40trial-20261105/` |
| **E2** | 6 augmentasi + Tanpa RankUp, strength tetap | Tidak (weak 0, strong 0,5) | 1 (42) | 3 utama × 3 % | Selesai, 63 run | `eksperimen/hasil/usulan-semua/results.jsonl` |
| **E3** | Perbandingan E1 vs E2 | — | — | 3 utama | Analisis | notebook §4.4 |
| **E4** | Seed vs augmentasi (dari E1) | — | 5 | 3 utama | Analisis | notebook §4.3, §4.6, §4.7 |
| **E5** | Seperti E2 di 13 dataset | Tidak | 1 | 13 dataset, 36 kondisi | Selesai | `eksperimen/hasil/usulan-semua/` |
| **E6** | RankUp vs supervised dan SSL lain | Tidak (pembanding) | 1 | 3 utama | Selesai | `eksperimen/hasil/pembanding/` |
| **E7** | Eksplorasi k-NN Block-Gibbs | Tidak | 1–5 | 3 utama, lalu 13 | Selesai | `eksperimen/catatan/LAPORAN_METODE_USULAN.md` |
| **E8** | Eksplorasi augmentasi lain (pilot) | Tidak | 1 (+ cek seed 43) | 3 utama | Selesai, hanya validation | `usulan/hasil/` |
| **E9** | 5-fold, 1 seed | Ya | 1 | 3 utama | Dirancang; hasil tidak ada di lokal | `README.md` bagian 5-fold |
| **E10** | Augmentasi di ruang latent TabM | Tidak | 1 | 3 utama | Dilanjutkan di `../5-augmentasi-latent/` (3 seed, weak/strong latent) | `../5-augmentasi-latent/` |

Semua visualisasi E1–E5 ada di `eksperimen/visualisasi_eksperimen.ipynb`, dengan gambar di `eksperimen/hasil/gambar/`.

---

## 5. E1 — Optuna 40 trial + 5 seed (eksperimen utama)

### 5.1 Setting

Untuk setiap kombinasi (3 dataset × 6 augmentasi × 3 % label = **54 kombinasi**):

1. **Optuna** mencari strength weak dan strong yang memaksimalkan metrik validation. Maksimal 40 trial, tiap trial maksimal 40 epoch, dengan pruning, dan seed tuning 42. Parameter lain tetap (tabel §3.2).
2. **Training ulang** dengan parameter terbaik memakai 5 seed (42, 123, 456, 789, 101112), maksimal 120 epoch. Tiap seed memilih checkpoint terbaiknya sendiri di validation.
3. Hitung rata-rata ± std dari skor **test** kelima seed.

**Seed hanya mengubah proses training**, yaitu bobot awal, urutan batch, dan keacakan augmentasi. Split data dan hyperparameter tetap sama.

Total: 54 studi Optuna dan 270 training akhir.

### 5.2 Hasil (test, rata-rata ± std 5 seed, peringkat dalam kurung)

Baris diurutkan dari rata-rata peringkat terbaik ke terburuk.

**NHANES (QWK ↑)**

| Augmentasi | 1% | 10% | 30% | Rata-rata peringkat |
|---|---|---|---|---:|
| Zero-out | 0,271 ± 0,009 (#3) | 0,257 ± 0,027 (#2) | 0,305 ± 0,011 (#1) | 2,00 |
| Range-limited | **0,277** ± 0,008 (#1) | 0,236 ± 0,007 (#3) | 0,303 ± 0,008 (#4) | 2,67 |
| Gaussian | 0,230 ± 0,009 (#6) | **0,258** ± 0,034 (#1) | 0,303 ± 0,015 (#3) | 3,33 |
| SCARF | 0,273 ± 0,006 (#2) | 0,228 ± 0,007 (#6) | 0,304 ± 0,008 (#2) | 3,33 |
| TabCutMix+ | 0,252 ± 0,008 (#4) | 0,228 ± 0,005 (#5) | 0,300 ± 0,011 (#5) | 4,67 |
| TabCutMix | 0,238 ± 0,011 (#5) | 0,231 ± 0,002 (#4) | 0,299 ± 0,010 (#6) | 5,00 |

**Pen Digits (Macro F1 ↑)**

| Augmentasi | 1% | 10% | 30% | Rata-rata peringkat |
|---|---|---|---|---:|
| TabCutMix | **0,897** ± 0,012 (#1) | **0,967** ± 0,000 (#1) | 0,972 ± 0,002 (#5) | 2,33 |
| Gaussian | 0,880 ± 0,005 (#3) | 0,953 ± 0,006 (#4) | **0,976** ± 0,001 (#1) | 2,67 |
| Range-limited | 0,878 ± 0,013 (#4) | 0,952 ± 0,001 (#5) | 0,975 ± 0,002 (#2) | 3,67 |
| TabCutMix+ | 0,874 ± 0,004 (#5) | 0,961 ± 0,009 (#2) | 0,972 ± 0,004 (#4) | 3,67 |
| SCARF | 0,874 ± 0,011 (#6) | 0,954 ± 0,008 (#3) | 0,975 ± 0,001 (#3) | 4,00 |
| Zero-out | 0,889 ± 0,004 (#2) | 0,948 ± 0,001 (#6) | 0,969 ± 0,002 (#6) | 4,67 |

**Superconductivity (MSE ↓)**

| Augmentasi | 1% | 10% | 30% | Rata-rata peringkat |
|---|---|---|---|---:|
| TabCutMix+ | **369,7** ± 4,7 (#1) | **205,1** ± 5,3 (#1) | **147,3** ± 2,7 (#1) | 1,00 |
| SCARF | 376,6 ± 6,9 (#4) | 205,8 ± 5,8 (#3) | 148,6 ± 5,1 (#2) | 3,00 |
| Gaussian | 370,7 ± 1,9 (#2) | 205,8 ± 5,1 (#4) | 152,8 ± 7,7 (#6) | 4,00 |
| Range-limited | 375,9 ± 8,3 (#3) | 206,4 ± 5,3 (#6) | 150,0 ± 7,0 (#3) | 4,00 |
| TabCutMix | 379,3 ± 4,6 (#6) | 205,7 ± 5,2 (#2) | 152,0 ± 8,0 (#5) | 4,33 |
| Zero-out | 377,2 ± 6,7 (#5) | 206,2 ± 5,3 (#5) | 152,0 ± 7,6 (#4) | 4,67 |

**Peringkat keseluruhan, 9 kondisi:**

| Augmentasi | Peringkat |
|---|---:|
| TabCutMix+ | 3,11 |
| Gaussian | 3,33 |
| Range-limited | 3,44 |
| SCARF | 3,44 |
| Zero-out | 3,78 |
| TabCutMix | 3,89 |

Semua berdekatan (selisih 0,8 peringkat). Setiap augmentasi pernah peringkat 1 atau 2 dan pernah peringkat 5–6.

Gambar: `4a_optuna_5seed_tabel.png`, `4b_optuna_5seed_peringkat.png`.

### 5.3 Temuan dari E1

- **TabCutMix+ selalu peringkat 1 di Superconductivity**, tetapi selisihnya kecil dibanding std seed (misalnya 205,1 vs 206,4 di 10%, padahal std ±5). Lihat E4.
- **Di Pen Digits, TabCutMix unggul jelas di 1% dan 10%.** Std-nya kecil, jadi perbedaannya nyata.
- **Checkpoint sering dipilih sangat awal**, sebelum SSL aktif. Rata-rata epoch terbaik:

  | Kondisi | Epoch terbaik |
  |---|---|
  | NHANES 30% | 3,0 |
  | NHANES 10% | 7,2 |
  | Superconductivity 1% | 4,9 |
  | Superconductivity 30% | 91,7 |

  Di awal training, pseudo-label hampir tidak ada yang lolos cutoff 0,95; acceptance NHANES 30% hanya 0,05–1% di 5 epoch pertama. Jadi model terpilih praktis masih supervised.
- **Optuna kadang mematikan SSL.** SCARF hasil tuning di Superconductivity 10% dan 30% punya acceptance pseudo-label hanya ±0,8%, karena view weak-nya sudah terlalu rusak sehingga teacher tidak pernah yakin.
- **Strength hasil Optuna melompat-lompat.** Contohnya weak σ Gaussian di NHANES: 0,65 (1%), 0,008 (10%), 1,07 (30%). Ini tanda permukaan objektifnya datar dan pemilihannya didominasi noise. Tabel lengkap ada di notebook §4.5.

---

## 6. E2 — Tanpa Optuna, 1 seed, 3 dataset utama

### 6.1 Setting

Semua augmentasi memakai **setting yang sama tanpa tuning**: weak = identitas (strength 0), strong = 0,5 (Gaussian σ = 0,5), seed 42, maksimal 120 epoch. Ditambah **Tanpa RankUp** sebagai pembanding. Tujuannya membandingkan augmentasi secara adil tanpa pengaruh tuning.

### 6.2 Hasil (test, seed 42, peringkat dari 7 metode)

**NHANES (QWK ↑)**

| Metode | 1% | 10% | 30% | Rata-rata peringkat |
|---|---|---|---|---:|
| Gaussian | 0,229 (#5) | **0,247** (#1) | **0,334** (#1,5) | 2,50 |
| Zero-out | **0,251** (#1) | 0,226 (#7) | **0,334** (#1,5) | 3,17 |
| SCARF | 0,242 (#2) | 0,240 (#3) | 0,324 (#5,5) | 3,50 |
| Range-limited | 0,191 (#6) | 0,245 (#2) | 0,324 (#5,5) | 4,50 |
| TabCutMix+ | 0,239 (#3) | 0,233 (#5) | 0,324 (#5,5) | 4,50 |
| *Tanpa RankUp* | *0,150 (#7)* | *0,240 (#4)* | *0,331 (#3)* | *4,67* |
| TabCutMix | 0,231 (#4) | 0,232 (#6) | 0,324 (#5,5) | 5,17 |

**Pen Digits (Macro F1 ↑)**

| Metode | 1% | 10% | 30% | Rata-rata peringkat |
|---|---|---|---|---:|
| Range-limited | **0,877** (#1) | **0,955** (#1) | **0,981** (#1) | 1,00 |
| Gaussian | 0,871 (#3) | 0,951 (#2) | 0,975 (#2) | 2,33 |
| Zero-out | 0,877 (#2) | 0,945 (#6) | 0,971 (#4) | 4,00 |
| TabCutMix+ | 0,867 (#5) | 0,949 (#4) | 0,971 (#3) | 4,00 |
| TabCutMix | 0,869 (#4) | 0,939 (#7) | 0,969 (#5) | 5,33 |
| SCARF | 0,839 (#6) | 0,950 (#3) | 0,963 (#7) | 5,33 |
| *Tanpa RankUp* | *0,838 (#7)* | *0,947 (#5)* | *0,967 (#6)* | *6,00* |

**Superconductivity (MSE ↓)**

| Metode | 1% | 10% | 30% | Rata-rata peringkat |
|---|---|---|---|---:|
| SCARF | **372,5** (#1) | 197,8 (#5) | **146,8** (#1) | 2,33 |
| Range-limited | 373,6 (#2) | 197,0 (#3) | 150,4 (#3) | 2,67 |
| TabCutMix | 374,6 (#4) | 208,3 (#6) | 149,1 (#2) | 4,00 |
| Gaussian | 375,0 (#5) | **194,4** (#1) | 152,1 (#7) | 4,33 |
| Zero-out | 375,6 (#6) | 196,9 (#2) | 150,7 (#5) | 4,33 |
| *Tanpa RankUp* | *378,5 (#7)* | *197,4 (#4)* | *150,5 (#4)* | *5,00* |
| TabCutMix+ | 374,4 (#3) | 208,7 (#7) | 151,1 (#6) | 5,33 |

Peringkat bernilai x,5 berarti seri: nilainya sama persis dengan metode lain.

**Peringkat keseluruhan, 9 kondisi:**

| Metode | Peringkat |
|---|---:|
| Range-limited | 2,72 |
| Gaussian | 3,06 |
| SCARF | 3,72 |
| Zero-out | 3,83 |
| TabCutMix+ | 4,61 |
| TabCutMix | 4,83 |
| Tanpa RankUp | 5,22 |

Gambar: `1_tiga_dataset.png`, `3_vs_metode_lain.png`.

### 6.3 Temuan dari E2

- **RankUp + augmentasi paling membantu di label 1%.** NHANES 1%: QWK 0,150 → 0,251 (Zero-out). Pen Digits 1%: F1 0,838 → 0,877 (Range-limited). Di 30%, manfaatnya kecil atau tidak ada.
- **Tanpa RankUp peringkat terakhir secara keseluruhan (5,22)**, tetapi di NHANES 30% ia peringkat 3, mengalahkan 4 augmentasi.
- **Banyak nilai yang identik** menunjukkan augmentasi belum sempat berpengaruh. Epoch terbaik per kondisi (7 metode):

  | Kondisi | Epoch terbaik |
  |---|---|
  | NHANES 30% | 1, 1, 1, 1, 1, 1, 1 (semua epoch 1) |
  | Superconductivity 1% | 2, 2, 2, 2, 2, 2, 2 |
  | NHANES 10% | 3–6 |

  Di NHANES 30%, empat augmentasi menghasilkan QWK 0,324 yang sama persis, karena checkpoint-nya diambil sebelum augmentasi bekerja.
- **Range-limited konsisten di Pen Digits** (peringkat 1 di ketiga %), tapi peringkat 6 di NHANES 1%.

---

## 7. E3 — Optuna vs tanpa Optuna

Perbandingan adil memakai **seed yang sama (42)**: skor test Optuna seed 42 (E1) dibanding skor test tanpa Optuna seed 42 (E2), untuk setiap augmentasi dan kondisi.

| Kondisi | Optuna lebih baik (dari 6 augmentasi) |
|---|---:|
| NHANES 1% | 5 |
| NHANES 10% | 2 |
| NHANES 30% | **0** |
| Pen Digits 1% | **6** |
| Pen Digits 10% | 5 |
| Pen Digits 30% | 4 |
| Superconductivity 1% | 5 |
| Superconductivity 10% | 2 |
| Superconductivity 30% | 4 |
| **Total** | **33 dari 54** |

- Kalau memakai **rata-rata 5 seed** Optuna dibanding tanpa Optuna seed 42, angkanya **29 dari 54**.
- **NHANES 30%:** semua augmentasi Optuna (QWK 0,299–0,305) lebih buruk dari tanpa Optuna (0,324–0,334), bahkan lebih buruk dari **Tanpa RankUp (0,331)**.
- **Pen Digits 1%:** Optuna membantu semua augmentasi.
- **Kesimpulan:** tuning 40 trial di validation tidak menjamin hasil test lebih baik. Penyebabnya: validation-nya noisy, permukaan objektifnya datar, dan Optuna kadang memilih strength yang mematikan SSL (§5.3).

Gambar: `4d_optuna_vs_tanpa_optuna.png`.

---

## 8. E4 — Temuan: variasi seed vs perbedaan augmentasi

### 8.1 Pertanyaan

Kalau augmentasi diganti, hasilnya berubah sebesar apa? Kalau **hanya seed** yang diganti (augmentasi, data, dan hyperparameter sama), hasilnya berubah sebesar apa? Kalau efek seed lebih besar, maka memilih "augmentasi terbaik" berdasarkan 1 seed tidak bisa dipercaya.

### 8.2 Cara menghitung (data E1, 6 augmentasi × 5 seed per kondisi)

| Ukuran | Cara hitung |
|---|---|
| **Selisih antar augmentasi** | Rata-ratakan 5 seed per augmentasi, lalu ambil terbaik − terburuk dari 6 rata-rata itu. |
| **Rentang antar seed (setara)** | Rata-ratakan 6 augmentasi per seed, lalu ambil terbaik − terburuk dari 5 rata-rata itu. Cara hitungnya sama persis dengan baris atas, jadi **perbandingannya adil**. |
| Rentang seed per augmentasi | Untuk tiap augmentasi, hitung maks − min dari 5 seed, lalu rata-ratakan untuk 6 augmentasi. Angka ini sedikit menguntungkan sisi seed karena membandingkan nilai satuan, bukan rata-rata. |
| **p ANOVA** | Uji one-way ANOVA: apakah rata-rata 6 augmentasi berbeda, dengan memperhitungkan variasi seed? p < 0,05 berarti augmentasi berpengaruh signifikan. |

### 8.3 Hasil

| Kondisi | Augmentasi terbaik | Augmentasi terburuk | Selisih antar augmentasi | Rentang antar seed (setara) | Rentang seed per augmentasi | p ANOVA |
|---|---|---|---:|---:|---:|---:|
| NHANES 1% | Range-limited 0,277 | Gaussian 0,230 | 0,047 | 0,006 | 0,022 | < 0,001 |
| NHANES 10% | Gaussian 0,258 | SCARF 0,228 | 0,030 | **0,032** | 0,034 | 0,033 |
| NHANES 30% | Zero-out 0,305 | TabCutMix 0,299 | 0,006 | **0,023** | 0,027 | **0,938** |
| Pen Digits 1% | TabCutMix 0,897 | SCARF 0,874 | 0,023 | 0,016 | 0,020 | 0,003 |
| Pen Digits 10% | TabCutMix 0,967 | Zero-out 0,948 | 0,020 | 0,003 | 0,011 | < 0,001 |
| Pen Digits 30% | Gaussian 0,976 | Zero-out 0,969 | 0,007 | 0,002 | 0,005 | < 0,001 |
| Superconductivity 1% | TabCutMix+ 369,7 | TabCutMix 379,3 | 9,6 | 7,8 | 13,1 | **0,099** |
| Superconductivity 10% | TabCutMix+ 205,1 | Range-limited 206,4 | 1,3 | **12,5** | 12,7 | **0,999** |
| Superconductivity 30% | TabCutMix+ 147,3 | Gaussian 152,8 | 5,5 | **11,8** | 15,1 | **0,741** |

Angka tebal di kolom rentang seed (setara) berarti rentang seed **lebih besar** dari selisih augmentasi. Angka tebal di kolom p berarti perbedaan augmentasi **tidak signifikan**.

### 8.4 Cara menyampaikan

- **Augmentasi tidak berbeda signifikan di 4 dari 9 kondisi:** Superconductivity 1/10/30% dan NHANES 30% (ANOVA p > 0,05).
- **Dengan perbandingan yang setara, rentang seed > selisih augmentasi juga di 4 dari 9 kondisi:** NHANES 10%, NHANES 30%, Superconductivity 10%, dan Superconductivity 30%.
- **Contoh terkuat, Superconductivity 10%:**
  - Mengganti augmentasi hanya mengubah MSE 1,3.
  - Mengganti seed mengubah MSE 12,5.
  - Seed 42 memberi MSE terbaik (±197–199) dan seed 123/101112 memberi yang terburuk (±209–212) **untuk keenam augmentasi**. Hasilnya ditentukan oleh seed, bukan augmentasi.
- **Kebalikannya, Pen Digits:** perbedaan augmentasi signifikan (p ≤ 0,003). TabCutMix 10% (0,967 ± 0,000) jelas terpisah dari yang lain.
- **Hindari kalimat** "seed selalu lebih berpengaruh". Klaim yang benar: "di sebagian kondisi (regresi dan NHANES 30%), perbedaan antar augmentasi tertutup oleh variasi seed. Karena itu klaim augmentasi terbaik harus berbasis banyak seed dan uji statistik."

Gambar:
- `4f_slide_seed_vs_augmentasi.png`: untuk slide. Dua panel, titik diwarnai per seed.
- `4e_variasi_seed_vs_augmentasi.png`: ringkasan 9 kondisi.
- `4c_optuna_5seed_sebaran.png`: detail per kondisi.

---

## 9. E5 — 13 dataset, tanpa Optuna, 1 seed

### 9.1 Setting

Sama dengan E2 (weak 0, strong 0,5, seed 42, 120 epoch) pada 13 dataset (§2.4), total **36 kondisi**. Tujuannya mengecek apakah peringkat dari 3 dataset utama berlaku umum.

### 9.2 Hasil (peringkat test dari 7 metode, rata-rata 36 kondisi)

| Metode | Semua | Ordinal | Klasifikasi | Regresi |
|---|---:|---:|---:|---:|
| Zero-out | **3,68** | **2,71** | 4,11 | 4,33 |
| Gaussian | 3,75 | 3,04 | 3,75 | 5,17 |
| Range-limited | 3,83 | 5,12 | **3,31** | **2,83** |
| TabCutMix+ | 3,97 | 4,17 | 3,83 | 4,00 |
| TabCutMix | 4,01 | 4,29 | 3,94 | 3,67 |
| SCARF | 4,26 | 3,96 | 4,50 | 4,17 |
| Tanpa RankUp | 4,49 | 4,71 | 4,56 | 3,83 |

**Uji Friedman** (7 metode × 36 kondisi): χ² = 4,05, **p = 0,67**. Tidak ada metode yang berbeda signifikan secara keseluruhan, termasuk terhadap Tanpa RankUp.

### 9.3 Temuan

- **Augmentasi terbaik bergantung pada tugas:**
  - ordinal: Zero-out dan Gaussian;
  - klasifikasi: Range-limited;
  - regresi: Range-limited, sementara Gaussian terburuk.
- **Tanpa RankUp peringkat terakhir secara keseluruhan, tapi tidak signifikan.** Di regresi, ia peringkat 2 (3,83).
- **Sel yang tidak sensitif:** ESL, LEV, dan NHANES 30% banyak menghasilkan nilai yang sama persis, karena checkpoint dipilih sebelum augmentasi bekerja.
- Tabel per dataset lengkap ada di `eksperimen/catatan/LAPORAN_METODE_USULAN.md` §5.8. Gambar: `2a_ordinal_qwk.png`, `2b_klasifikasi_f1.png`, `2c_regresi_mse.png`, `2d_peringkat_keseluruhan.png`.

---

## 10. E6 — RankUp vs metode supervised dan SSL lain

### 10.1 Setting

3 dataset utama, seed 42. Script `eksperimen/bandingkan_ssl.py` dan `modal_pembanding.py`.

- **Supervised (hanya data berlabel, tanpa tuning):** HistGradientBoosting (HGB), Ridge / Logistic.
- **SSL klasik:**
  - self-training HGB;
  - COREG (dua kNN regressor);
  - Label Spreading (khusus klasifikasi).
- **SSL deep tabular:** VIME, SSLAE, GFTab, STUNT, memakai runner dan setting default Task 3. Runner ini hanya mendukung klasifikasi dan ordinal, jadi tidak dipakai di Superconductivity.

### 10.2 Hasil (test, metrik utama)

| Metode | NHANES 1% | 10% | 30% | Pen Digits 1% | 10% | 30% | Supercond. 1% ↓ | 10% ↓ | 30% ↓ |
|---|---|---|---|---|---|---|---|---|---|
| RankUp terbaik* | **0,286** | 0,254 | **0,334** | 0,907 | **0,967** | **0,981** | **370,4** | 194,4 | 146,2 |
| TabM Tanpa RankUp | 0,150 | 0,240 | 0,331 | 0,838 | 0,947 | 0,967 | 378,6 | 197,4 | 150,5 |
| HGB (supervised) | 0,000 | 0,285 | 0,290 | 0,747 | 0,904 | 0,954 | 374,5 | **178,1** | 143,4 |
| Self-training HGB | 0,011 | **0,291** | 0,293 | 0,766 | 0,915 | 0,956 | 380,6 | 183,4 | **142,7** |
| COREG | 0,173 | 0,199 | 0,247 | – | – | – | 477,2 | 264,0 | 191,4 |
| Label Spreading | – | – | – | **0,918** | 0,965 | 0,967 | – | – | – |
| Ridge / Logistic | 0,083 | 0,205 | 0,245 | 0,811 | 0,885 | 0,901 | 437,8 | 358,0 | 341,5 |
| SSLAE | 0,143 | 0,254 | 0,297 | 0,658 | 0,909 | 0,926 | – | – | – |
| GFTab | 0,027 | 0,178 | 0,287 | 0,264 | 0,858 | 0,920 | – | – | – |
| VIME | 0,029 | 0,172 | 0,227 | 0,812 | 0,878 | 0,895 | – | – | – |
| STUNT | 0,144 | 0,163 | 0,186 | 0,665 | 0,665 | 0,528 | – | – | – |

\*"RankUp terbaik" = nilai **maksimum di test** dari 14 varian RankUp: 6 augmentasi dengan Optuna (seed 42), 6 tanpa Optuna, Block-Gibbs, dan Tanpa RankUp. Ini menguntungkan RankUp karena pembanding hanya punya 1–4 varian tanpa tuning.

### 10.3 Temuan

- **RankUp menang di 5 dari 9 kondisi; pembanding menang di 4.** Gradient boosting dan self-training menang di NHANES 10%, Superconductivity 10% (MSE 178 vs 194), dan Superconductivity 30%. Label Spreading menang di Pen Digits 1%.
- **Kekuatan RankUp/TabM ada di label paling sedikit dan klasifikasi.** Di NHANES 1%, HGB gagal total (QWK 0, prediksinya konstan).
- **SSL deep tabular (VIME, SSLAE, GFTab, STUNT) kalah dari RankUp terbaik di keenam kondisi NHANES dan Pen Digits.**
- **Untuk regresi dengan label ≥ 10%, model pohon supervised sangat kuat.** Backbone dan objective perlu dipertimbangkan, bukan hanya augmentasi. Sebagai pembanding: Extra Trees supervised di validation Superconductivity 10% mencapai MSE 169 (E8).

---

## 11. E7 — Eksplorasi augmentasi k-NN Block-Gibbs

Eksperimen tambahan untuk menguji augmentasi yang **menjaga distribusi data**. Status: **eksperimen eksplorasi**, bukan metode usulan TA. Detail lengkap ada di `eksperimen/catatan/LAPORAN_METODE_USULAN.md`.

### 11.1 Ide

Untuk setiap baris, pilih blok fitur acak S. Cari k tetangga terdekat di train pool, dengan jarak dihitung hanya pada fitur **di luar** S. Lalu salin nilai blok S dari salah satu tetangga itu.

Secara teori ini satu langkah Gibbs: baris hasil augmentasi tetap berdistribusi sama dengan data asli. Augmentasi lain tidak punya sifat ini: SCARF merusak korelasi antar fitur, sedangkan Gaussian dan Zero-out keluar dari rentang data.

### 11.2 Rangkaian percobaan dan hasil

| Tahap | Setting | Hasil |
|---|---|---|
| Uji tanpa training | Realism (C2ST AUC), korelasi, label tetap | Paling realistis (AUC 0,54–0,76 vs SCARF 0,94–1,00), korelasi paling terjaga, label paling terjaga di 6 dari 6 kombinasi |
| Screening 1 seed | 7 konfigurasi k dan strength | `gibbs_k5_s50` terbaik di validation (peringkat 3,2) |
| Protokol RankUp-setia | EMA + uratio 7 | Awalnya gagal karena bug EMA (QWK 0). Setelah diperbaiki, overfit awal tetap terjadi, jadi tidak dilanjutkan |
| 3 seed vs Optuna | `gibbs_k5_s50` | Peringkat 1 (3,90), tapi ternyata noise (lihat 5 seed) |
| Kalibrasi strength tanpa label | ρ = 0,05 | Peringkat validation 2,3. Augmentasi data berlabel merugikan (6,4–6,8) |
| Variasi protokol | lr, bobot unlabeled, cutoff | Default terbaik. lr kecil membuat model kurang terlatih |
| Threshold adaptif (FreeMatch SAT) | 3 seed | Lebih buruk (4,63). Contoh: NHANES 1% validation 0,311 (terbaik) tapi test 0,225 (terburuk) |
| **Konfirmasi 5 seed** | vs 6 augmentasi Optuna | **Peringkat 3 dari 8** (4,17), di bawah TabCutMix+ (3,63) dan SCARF (4,00). Signifikan > Tanpa RankUp di NHANES (p = 0,015) dan Pen Digits (p = 0,010) |
| **13 dataset, 1 seed** | tanpa Optuna | **Peringkat terakhir** (5,28 dari 8). Friedman p = 0,31. Hampir seri dengan Tanpa RankUp (16 menang, 18 kalah) |
| Varian diversity lebih tinggi | 3 langkah Gibbs, k = 50, kalibrasi langkah | Ketiganya lebih buruk |

### 11.3 Pelajaran utama

Augmentasi yang paling "realistis" justru **tidak** memberi manfaat SSL. Augmentasi yang berhasil (Zero-out, Gaussian, Range-limited, TabCutMix) semuanya merusak data ke arah yang tidak realistis.

Hipotesisnya: pada RankUp tabular, manfaat konsistensi lebih banyak datang dari **regularisasi terhadap noise** (memaksa model mulus di arah yang tidak terlihat di data; sejalan dengan Bishop, 1995), bukan dari invariansi antar sampel realistis. Hipotesis ini **belum diuji langsung**. E10 (augmentasi latent) bisa menjadi salah satu cara mengujinya.

---

## 12. E8 — Eksplorasi augmentasi lain (folder `usulan/`)

Serangkaian pilot cepat. **Semuanya 1 seed (42), tanpa Optuna, dan hanya validation** (test tidak dibuka). Budget-nya berbeda-beda: pilot pertama 30 epoch × 40 langkah, sisanya 120 × 100. Jadi angka antar pilot tidak bisa dibandingkan langsung, dan semuanya bersifat screening, bukan bukti.

| Pilot | Ide | Hasil utama (validation) | File temuan |
|---|---|---|---|
| **Bounded local transport** | Geser baris ke tetangga terdekat dengan pseudo-label sama; jarak geser dibatasi | Terbaik di 3 dari 9 kondisi; rata-rata peringkat 4,00 dari 9 (Gaussian 3,89). Selisih sangat kecil | `usulan/hasil/temuan_pilot.md` |
| **Lanjutan:** correlated noise, gated cutmix, hybrid | Noise mengikuti kovarians data; cutmix dengan donor terdekat; campuran keduanya | Hybrid peringkat 4,33 dari 12, terbaik di 2 dari 9, tapi jauh di bawah pemenang di NHANES 1% dan 10% | `temuan_pilot_lanjutan.md` |
| **Conditional donor** (random / group / + soft rank) | Donor dipilih bersyarat pada fitur yang tidak diganti; varian soft rank mengubah bobot pseudo-rank | Group terbaik di Superconductivity 10%, group + soft rank di 30%; lebih buruk di NHANES 1% | `temuan_conditional_screen.md` |
| **Feature-guided** | Fitur yang berasosiasi kuat dengan target lebih jarang diganti | Terbaik di Pen Digits 10% dan 30% (0,989; 0,997); turun di 3 kondisi lain | `temuan_feature_guided.md` |
| **Campuran adaptif** | Pilih operator per baris berdasarkan korelasi fitur dan jumlah label | Lebih baik dari feature-guided di 5 dari 9, tapi tidak pernah menjadi pemenang | `temuan_adaptif.md` |
| **SSL berbasis pohon** | Teacher Extra Trees + pseudo-label (bukan RankUp) | Superconductivity 1%: MSE 313,7 vs supervised 328,0 (membaik); 10% dan 30% memburuk. Extra Trees supervised 10%: 169,1 (lebih baik dari semua varian neural) | `analisis_model_20261110.md` |
| **Autoresearch** (3 putaran) | Iterasi otomatis perbaikan operator feature-guided | Putaran 1 discard (3 menang / 5 kalah), putaran 2 discard (5 / 3), putaran 3 keep (5 menang, 4 seri, 0 kalah). **Cek ulang seed 43: keunggulannya hilang** (NHANES 0,284 vs 0,285; Superconductivity 345,5 vs 339,6) | `usulan/hasil/autoresearch/percobaan.md` |

**Pelajaran dari E8:** setiap "kemenangan" di 1 seed validation cenderung hilang saat diuji seed lain (autoresearch putaran 3 → seed 43). Ini konsisten dengan temuan E4: memilih metode dari validation 1 seed tidak andal.

---

## 13. E9 — Mode 5-fold (dirancang)

- **Rancangan:**
  - test tetap;
  - data non-test dibagi 5 fold (stratified untuk NHANES dan Pen Digits, GroupKFold untuk Superconductivity);
  - Optuna 40 trial per fold, lalu 1 training dengan seed 42;
  - total 270 studi dan 270 training.
- **Tujuan:** mengukur variasi akibat **split data**, yang tidak terukur di E1 (E1 hanya memvariasikan seed training).
- **Status:** perintah dan script ringkasan (`workflow_kfold.py`, `ringkas_kfold.py`) sudah ada, tetapi **hasil `ta-kfold-001` tidak ada di folder lokal**. Belum dicek apakah run-nya pernah dijalankan di Modal.

---

## 14. E10 — Augmentasi latent (dipindah ke 5-augmentasi-latent)

### 14.1 Ide

Input tidak diubah. Yang diganggu adalah **vektor latent** TabM, yaitu output backbone (512 dimensi per submodel), tepat sebelum head:

```
x ──► backbone TabM ──► h (latent) ──► [GANGGU h, hanya view strong] ──► head ──► prediksi / ranking
```

Training tetap end-to-end dari nol. Gradien konsistensi mengalir melewati gangguan sampai ke backbone, sehingga backbone belajar membuat latent yang tahan gangguan.

### 14.2 Tiga operasi × strength {0,2; 0,5} = 6 konfigurasi

| Operasi | Cara kerja |
|---|---|
| `gaussian` | h + noise; skala noise = strength × std tiap dimensi latent di batch (relatif, sehingga model tidak bisa "curang" memperbesar skala latent) |
| `dropout` | Sebagian dimensi latent dibuat 0 (proporsi = strength) |
| `interp` | h digeser ke arah tetangga terdekatnya di ruang latent dalam batch, maksimal sejauh strength |

### 14.3 Protokol dan status

- **Protokol:** sama dengan E2 (seed 42, 120 epoch, weak = identitas), sehingga hasilnya bisa langsung dibandingkan dengan Tanpa RankUp dan 6 augmentasi input tanpa run ulang.
- **Status:**
  - 4 unit test lulus;
  - smoke run 2 epoch berhasil di ketiga dataset;
  - terverifikasi bahwa gangguan terpasang tepat 1 kali per langkah, hanya pada view strong.
- **Versi awal ini tidak pernah dijalankan.** Kodenya (folder `eksperimen-latent/`) sudah dihapus dan diganti versi lengkap di `../5-augmentasi-latent/kode/`: weak dan strong sama-sama bisa di latent, 7 operator (termasuk angular/cosine), 3 posisi (input, latent, input + latent), 3 seed. Interpolasi tidak dipakai lagi karena literatur menunjukkan efek negatif di tabular.
- Augmentasi latent sudah ada di literatur, termasuk untuk tabular: RaTab (AAAI 2025, self-supervised) dan SDAT (CIKM 2022, semi-supervised). Jadi tidak boleh diklaim sebagai yang pertama. Rujukan lengkap dan best practice ada di `../5-augmentasi-latent/papers/README.md`; rencana lanjutannya (menggantikan grid E10 di sini) ada di `../5-augmentasi-latent/PLAN_EKSPERIMEN_LATENT.md`.

---

## 15. Kesimpulan lintas eksperimen

### Yang didukung data

1. **RankUp + augmentasi > Tanpa RankUp, terutama di label 1%.**

   | Kondisi (tanpa Optuna) | Tanpa RankUp | Augmentasi terbaik |
   |---|---:|---:|
   | NHANES 1% (QWK) | 0,150 | 0,251 |
   | Pen Digits 1% (F1) | 0,838 | 0,877 |
   | Superconductivity 1% (MSE) | 378,5 | 372,5 |

   Di 13 dataset, Tanpa RankUp peringkat terakhir, tapi selisihnya tidak signifikan (Friedman p = 0,67).
2. **Tidak ada augmentasi terbaik universal.**
   - Dengan Optuna: TabCutMix+ (3,11) sampai TabCutMix (3,89).
   - Tanpa Optuna: Range-limited (2,72) sampai TabCutMix (4,83).
   - Di 13 dataset: Zero-out (3,68) sampai SCARF (4,26).

   Pemenangnya berganti antar setting.
3. **Augmentasi terbaik bergantung pada tugas dan setting.**
   - Pen Digits (klasifikasi): Range-limited (tanpa Optuna) atau TabCutMix (Optuna), dan perbedaannya signifikan.
   - Superconductivity: TabCutMix+ (Optuna), tapi tidak signifikan.
   - NHANES: Zero-out dan Gaussian.
4. **Variasi seed menutupi perbedaan augmentasi di 4 dari 9 kondisi** (semua Superconductivity dan NHANES 30%).
5. **Optuna tidak selalu membantu** (33 dari 54) dan bisa merugikan (NHANES 30%: 0 dari 6).
6. **Banyak kondisi tidak sensitif terhadap augmentasi**, karena checkpoint dipilih di epoch 1–4 sebelum SSL aktif (NHANES 30%, Superconductivity 1%).
7. **Augmentasi yang menjaga distribusi data (Block-Gibbs) tidak lebih baik dari Tanpa RankUp.** Augmentasi "kasar" justru lebih baik.
8. **Validation 1 seed tidak andal untuk memilih metode** (SAT di E7, autoresearch seed 43 di E8, dan pemenang validation ≠ pemenang test di E7 §5.9).

### Yang tidak bisa diklaim

- "Augmentasi X terbaik untuk RankUp tabular."
- "Seed selalu lebih berpengaruh dari augmentasi." Ini dibantah oleh Pen Digits dan NHANES 1%.
- "RankUp lebih baik dari metode tabular lain." Gradient boosting menang di 4 dari 9 kondisi.
- "Optuna meningkatkan hasil."

---

## 16. Keterbatasan dan ancaman validitas

| Keterbatasan | Dampak |
|---|---|
| Satu split data untuk E1–E8 | Variasi akibat split belum terukur (E9 belum ada hasilnya) |
| E2, E5, E6, E8 hanya 1 seed | Selisih kecil bisa sepenuhnya noise (lihat E4) |
| Test sudah dilihat berkali-kali | Klaim konfirmasi butuh holdout baru atau eksternal |
| Validation dipakai berulang untuk menyaring banyak kandidat (E7, E8) | Risiko overfit ke validation |
| Checkpoint di epoch 1–4 pada beberapa kondisi | Perbandingan augmentasi di kondisi itu tidak informatif |
| Pen Digits memakai RankUp-CLS (tanpa ARC) | Bukan RankUp asli; harus disebut sebagai adaptasi |
| "Tanpa RankUp" masih punya loss ranking berlabel | Bukan baseline supervised polos |
| Strength yang sama (0,5) punya arti berbeda antar augmentasi | Perbandingan tanpa Optuna adalah "budget sama", bukan "kekuatan setara" |
| NHANES 1%: satu level hanya punya 1 label | Hasil NHANES 1% sangat rapuh |
| Superconductivity: 1.668 kelompok fitur identik punya target berbeda | Ada batas bawah error yang tidak bisa diturunkan |
| "RankUp terbaik" di E6 dipilih dari test | Menguntungkan RankUp dibanding pembanding |

---

## 17. Langkah berikutnya

1. **Augmentasi latent**: dikerjakan mengikuti `../5-augmentasi-latent/PLAN_EKSPERIMEN_LATENT.md` (Fase 0–5, 3 seed), bukan grid E10 1-seed di atas.
2. **Multi-seed untuk kandidat terbaik.** Minimal 3–5 seed untuk setiap klaim, dengan uji statistik (ANOVA / Wilcoxon berpasangan).
3. **Perbaiki jadwal training** supaya SSL sempat bekerja di label sedikit, misalnya warm-up bobot supervised atau menghitung epoch dari putaran data berlabel.
4. **Variasi split (E9)** untuk memisahkan efek split dari efek seed.
5. **Uji hipotesis regularisasi noise (E7 §11.3):** bandingkan augmentasi on-manifold dan off-manifold dengan besar pergeseran yang sama.

---

## 18. Peta file dan reproduksi

| Lokasi | Isi |
|---|---|
| `README.md` | Rancangan eksperimen utama, perintah Modal, peta folder |
| `dataset/`, `config/`, `prepare_data.py` | Data mentah, config dataset, persiapan split |
| `eda/eda_datasets.ipynb` | EDA 3 dataset utama |
| `eksperimen/visualisasi_eksperimen.ipynb` | **Visualisasi E1–E5** (mulai dari sini) |
| `eksperimen/hasil/gambar/` | Semua gambar, siap untuk slide |
| `eksperimen/hasil/ringkasan-5seed-40trial.csv` | E1: mean, std, dan skor tiap seed |
| `eksperimen/hasil/ta-5seed-40trial-20261105/` | E1: best params, riwayat trial, metrik per seed, prediksi |
| `eksperimen/hasil/usulan-semua/results.jsonl` | E2, E5, dan sebagian E7 (13 dataset, 1 seed) |
| `eksperimen/hasil/pembanding/` | E6 |
| `eksperimen/hasil/usulan-screening/`, `usulan-protokol/`, `usulan-final-*.csv` | E7 |
| `eksperimen/hasil/log/` | Log run |
| `eksperimen/catatan/` | Laporan Block-Gibbs, riset augmentasi, audit sumber, pilihan dataset |
| `eksperimen/experiments/` | Kode training RankUp per tugas dan per augmentasi |
| `eksperimen/modal_app.py` | Runner E1 (Optuna + 5 seed) dan E9 |
| `eksperimen/modal_usulan.py` | Runner E2, E5, E7 |
| `eksperimen/modal_pembanding.py`, `bandingkan_ssl.py` | Runner E6 |
| `usulan/` | Kode dan hasil E8 |
| `../5-augmentasi-latent/` | E10 (lanjutan) |

Perintah utama ada di `README.md` (E1, E9), `eksperimen/catatan/LAPORAN_METODE_USULAN.md` §9 (E2, E5, E7), dan `../5-augmentasi-latent/README.md` (E10). Jalankan Modal dengan `MODAL_PROFILE=<profil>`.

Biaya (perkiraan dari catatan, belum dicocokkan dengan tagihan):
- E7 ±3,5 jam GPU (±$2–2,5);
- E5 ±1,6 jam T4;
- pilot pertama E8 ±$0,35.

---

## 19. Glosarium

| Istilah | Arti |
|---|---|
| **SSL** (semi-supervised learning) | Belajar dari sedikit data berlabel + banyak data tanpa label |
| **RankUp** | Metode SSL untuk regresi (NeurIPS 2024) yang menambah ARC dan konsistensi urutan pada data tanpa label |
| **ARC** | Auxiliary Ranking Classifier: head yang memprediksi "sampel A lebih besar dari B?" |
| **Pseudo-label** | Label tebakan model untuk data tanpa label; hanya dipakai jika confidence ≥ cutoff (0,95) |
| **View weak / strong** | Dua versi dari data tanpa label yang sama: weak = sedikit atau tidak diubah (jadi kunci jawaban), strong = diaugmentasi (harus konsisten) |
| **Strength** | Seberapa kuat augmentasi: proporsi fitur yang diubah, atau σ untuk Gaussian |
| **Seed** | Angka awal pengacak; mengubah bobot awal, urutan batch, dan keacakan augmentasi |
| **Optuna** | Library pencarian hyperparameter otomatis; di sini mencari strength weak/strong |
| **Checkpoint terbaik / early stopping** | Model disimpan pada epoch dengan skor validation terbaik; training berhenti jika 15 epoch berturut-turut tidak membaik |
| **Peringkat (#)** | Urutan dalam satu kondisi, 1 = terbaik. Nilai x,5 berarti seri. Untuk MSE, nilai terkecil = #1 |
| **ANOVA** | Uji apakah rata-rata beberapa kelompok (augmentasi) berbeda, dengan memperhitungkan variasi di dalam kelompok (seed) |
| **Friedman** | Uji apakah peringkat beberapa metode berbeda di banyak kondisi |
| **Wilcoxon berpasangan** | Uji apakah satu metode konsisten lebih baik dari metode lain pada pasangan kondisi yang sama |
| **p-value** | p < 0,05 → perbedaan dianggap signifikan (kecil kemungkinan hanya kebetulan) |
| **TabM** | Arsitektur MLP tabular dengan 32 submodel paralel yang berbagi bobot (Gorishniy dkk.) |
| **Latent** | Representasi internal model (output backbone) sebelum head prediksi |
