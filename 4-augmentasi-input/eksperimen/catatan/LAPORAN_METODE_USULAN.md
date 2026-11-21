# Laporan: augmentasi usulan k-NN Block-Gibbs untuk RankUp tabular

Laporan ini merangkum seluruh eksperimen metode usulan, dari diagnosis hasil 5-seed lama sampai uji pada 13 dataset. Semua angka berasal dari file hasil di `eksperimen/hasil/`. Pemilihan konfigurasi hanya memakai **validation**; test tidak dipakai untuk memilih apa pun. Ada satu pengecualian yang ditandai di §5.6.

## 1. Ringkasan

- **Penyebab hasil lama tidak stabil.** Di beberapa kondisi, checkpoint terbaik terpilih di epoch 1–4, sebelum cabang SSL aktif. Di kondisi itu, peringkat augmentasi praktis ditentukan oleh seed dan early stopping, bukan oleh kualitas augmentasi (§3).
- **Metode usulan.** k-NN Block-Gibbs Augmentation mengganti satu blok fitur dengan nilai tetangga terdekat, dengan jarak dihitung pada fitur yang tidak diganti. Secara teori ini satu langkah Gibbs yang menjaga distribusi data. Pada data kita, metode ini paling realistis, paling menjaga korelasi antar-fitur, dan paling menjaga urutan atau label (§4).
- **Hasil pada 3 dataset utama, 5 seed.** Usulan tanpa tuning berada di **peringkat 3 dari 8**, di belakang TabCutMix+ dan SCARF yang di-tuning Optuna 40 trial. Usulan terbaik di NHANES 30% (QWK) dan Superconductivity 1% (MSE), serta signifikan lebih baik dari tanpa-SSL di NHANES dan Pen Digits (§5.7).
- **Hasil pada 13 dataset, 1 seed, semua tanpa tuning.** Usulan di **peringkat terakhir** pada metrik utama (5,28 dari 8). Namun uji Friedman tidak menemukan perbedaan signifikan antar 8 metode (p = 0,31). Usulan hampir seri dengan tanpa-SSL (§5.8).
- **Varian dengan diversity lebih tinggi** (rantai Gibbs 3 langkah, k = 50, langkah dikalibrasi ke ρ = 0,2) tidak membantu: ketiganya lebih buruk di validation dan di test (§5.9).
- **Kesimpulan.** Pada benchmark ini tidak ada augmentasi tunggal yang terbaik. Ada kelompok teratas yang setara: Zero-out, Gaussian, TabCutMix(+), dan RLA. Semua varian Block-Gibbs berkinerja setara tanpa-SSL. Augmentasi yang menjaga distribusi data ternyata tidak memberi manfaat SSL di sini (§6).

## 2. Setting

- **Model.** TabM dengan RankUp: head regresi + Auxiliary Ranking Classifier (ARC) untuk ordinal dan regresi. Untuk klasifikasi dipakai RankUp-CLS, yaitu FixMatch-style tanpa ARC.
- **Training.** Runner asli `experiments/*/`: AdamW, lr 2e-3, batch 32, 100 langkah per epoch, maksimal 120 epoch, patience 15, bobot ranking 0,2, bobot unlabeled 0,2, cutoff 0,95.
- **Metrik utama.**
  - Ordinal: QWK
  - Klasifikasi: macro-F1
  - Regresi: MSE
  - Metrik tambahan: MAE, akurasi, Spearman, R².
- **Baseline "lama".** Run `ta-5seed-40trial-20261105`: 6 augmentasi × 3 dataset × 3 fraksi label, kekuatan weak/strong di-tuning Optuna 40 trial, lalu 5 seed.
- **Infrastruktur.** Modal workspace `lagi`, GPU T4.
  - Benchmark: T4 4,2 detik vs CPU 4-core 27,6 detik untuk job yang sama. Jadi T4 lebih murah per run, meski tarif per jamnya $0,59 vs ±$0,25.
  - Total ±3,5 jam GPU, perkiraan ±$2–2,5 (belum dicocokkan dengan tagihan).

## 3. Diagnosis hasil 5-seed lama

### 3.1 Tidak ada pemenang konsisten

Peringkat tiap augmentasi (metrik utama, rata-rata 5 seed) berganti-ganti antar kondisi. Setiap metode pernah di peringkat 1 dan pernah di peringkat 5–6:

| Metode | Rata-rata peringkat | Min | Maks |
|---|---:|---:|---:|
| TabCutMix+ | 3.11 | 1 | 5 |
| Gaussian | 3.33 | 1 | 6 |
| RLA | 3.44 | 1 | 6 |
| SCARF | 3.44 | 2 | 6 |
| Zero-out | 3.78 | 1 | 6 |
| TabCutMix | 3.89 | 1 | 6 |

Banyak selisih yang lebih kecil dari standar deviasi antar-seed. Contohnya NHANES 30%: QWK 0,299–0,305 dengan std ±0,01.

### 3.2 Checkpoint terpilih sebelum SSL aktif

Rata-rata epoch terbaik pada run lama:

| Kondisi | Rata-rata epoch terbaik | Median |
|---|---:|---:|
| NHANES 30% | 3.0 | 3 |
| NHANES 10% | 7.2 | 4 |
| Superconductivity 1% | 4.9 | 2 |
| Superconductivity 30% | 91.7 | 91 |

Dengan 36 label (NHANES 1%), satu epoch berarti ±89 putaran atas seluruh data berlabel. Model hafal data berlabel dengan cepat: QWK train = 1,0. Di Superconductivity 1%, MSE train 55 sementara test 370. Sementara itu pseudo-label hampir tidak ada yang lolos cutoff 0,95 di awal training; acceptance NHANES 30% hanya 0,05–1% di 5 epoch pertama. Akibatnya checkpoint yang dipilih adalah model yang praktis supervised.

Buktinya: di NHANES 30% (run usulan, seed 42), lima augmentasi berbeda menghasilkan QWK test **identik** (0,3291), semuanya dari checkpoint epoch 1.

### 3.3 Optuna kadang mematikan SSL

SCARF hasil tuning di Superconductivity 10% dan 30% memiliki acceptance pseudo-label hanya ±0,8%. View weak-nya sudah terlalu rusak sehingga teacher tidak pernah yakin. Strength hasil tuning juga melompat-lompat; misalnya weak σ Gaussian NHANES bernilai 0,65, 0,008, lalu 1,07 di tiga fraksi label. Ini menunjukkan permukaan objektifnya datar dan pemilihannya didominasi noise.

### 3.4 Penyimpangan dari kode resmi RankUp

| Aspek | Kode resmi (`rankup_utkface_lb250_s0.yaml`) | Runner TA |
|---|---|---|
| Evaluasi / checkpoint | EMA model, `ema_m` 0,999 | bobot mentah |
| Rasio unlabeled `uratio` | 7 | 1 |
| Iterasi | 262.144 | ≤ 12.000 |
| Bobot loss ARC | 0,2 × 1,0 | 0,2 (ekuivalen) |

## 4. Metode usulan: k-NN Block-Gibbs Augmentation

### 4.1 Celah dari literatur

| Augmentasi | Masalah |
|---|---|
| SCARF / VIME swap | Nilai pengganti diambil dari marginal per fitur, sehingga hubungan antar-fitur rusak (*off-manifold*). |
| Gaussian, Zero-out | Menghasilkan nilai di luar support data; satu skala noise untuk semua fitur. |
| TabCutMix(+) | Butuh label atau pseudo-label untuk memilih donor; tidak alami untuk regresi. |
| RLA | Sampling uniform dalam bin, sehingga fitur diskret bisa menjadi non-diskret. |

Sampler k-NN kondisional sudah dikenal di literatur *feature importance* (misalnya `ConditionalKNNSampler` di xplainfi) dan knockoff. Kami tidak menemukan penggunaannya sebagai strong augmentation untuk SSL tabular.

### 4.2 Algoritma

Untuk setiap baris x dalam batch unlabeled:

1. Pilih blok fitur acak S dengan |S| = round(s · d). Pemilihan S independen dari x.
2. Ambil kandidat acak sebanyak 2048 dari train pool (berlabel + tanpa label).
3. Hitung jarak hanya pada fitur yang tidak diganti: D(x, z) = Σ_{j∉S} (x_j − z_j)². Kandidat yang identik dengan x dikeluarkan.
4. Pilih satu donor secara acak dari k tetangga terdekat, lalu salin seluruh x_S dari donor itu.

Weak view = identitas. Implementasi: `experiments/usulan.py: block_gibbs`; tes: `tests/test_usulan.py`.

### 4.3 Teori

**Proposisi 1 (invariansi distribusi).** Jika x ~ P dan x̃_S ~ P(x_S | x_{−S}) dengan x̃_{−S} = x_{−S}, maka x̃ ~ P.

*Bukti.* Densitas gabungannya p(x_{−S}) p(x̃_S | x_{−S}) = p(x̃). Ini satu langkah block-Gibbs. ∎

**Proposisi 2 (jarak dari distribusi data).** Untuk S tetap:

| Augmentasi | Distribusi view | KL(P ‖ Q) |
|---|---|---|
| Random swap / TabCutMix tanpa label | p(x_{−S}) p(x_S) | I(X_S; X_{−S}) |
| SCARF (donor independen per sel) | p(x_{−S}) Π_{j∈S} p(x_j) | I(X_S; X_{−S}) + TC(X_S) |
| Block-Gibbs | p(x) | 0 |

TC adalah *total correlation* di dalam blok S. Jadi teori memprediksi urutan "keanehan" view: SCARF > random swap > Block-Gibbs. Urutan ini terkonfirmasi pada data (§4.4).

**Proposisi 3 (pergeseran label).** Karena x dan x̃ independen bersyarat diberikan x_{−S} dan berdistribusi sama:

E[(f(x̃) − f(x))²] = 2 · E[Var(f(X) | X_{−S})].

Label hanya bergeser sebesar informasi target yang tidak dapat diprediksi dari fitur sisanya. Kekuatan efektifnya otomatis lebih kecil pada fitur yang redundan.

**Proposisi 4 (keamanan pseudo-label ARC).** Pasangan yang diterima pada cutoff τ memiliki margin logit ≥ log(τ/(1−τ)); untuk τ = 0,95 nilainya 2,94. Dengan Chebyshev, peluang urutan pasangan terbalik ≤ 2·E[δ²] / margin², dengan δ = perubahan skor ranking akibat augmentasi. Drift yang kecil menekan batas noise pseudo-label.

**Peran k.** Untuk k = n operator ini sama dengan random swap; untuk k kecil sangat lokal. Konsistensi estimator k-NN kondisional terpenuhi bila k → ∞ dan k/n → 0 (Stone, 1977).

**Batasan.** Proposisi 1 eksak untuk distribusi bersyarat sebenarnya. Implementasi memakai estimator k-NN pada subset 2048 kandidat, jadi invariansinya bersifat aproksimasi.

### 4.4 Uji tanpa training (`analisis_operator.py`)

Uji ini memakai label 30%, train + validation saja, strength 0,3 dan 0,5. Rentang angka di bawah mencakup ketiga dataset dan kedua strength.

| Ukuran | SCARF | Random swap | Gaussian | Zero-out | Block-Gibbs k = 5 |
|---|---|---|---|---|---|
| Realism AUC (C2ST, 0,5 = tak terbedakan) | 0.94–1.00 | 0.94–0.99 | 1.00 | 1.00 | **0.54–0.76** |
| RMSE matriks korelasi | 0.12–0.31 | 0.09–0.22 | 0.03–0.09 | 0.07–0.21 | **0.02–0.04** |
| Urutan atau label tetap (oracle HGB) | 0.45–0.83 | 0.40–0.82 | 0.74–0.90 | 0.49–0.85 | **0.74–0.97** |

Block-Gibbs k = 5 terbaik pada label-kept di 6 dari 6 kombinasi dataset × strength. Hasil lengkap ada di `hasil/analisis_operator.csv`.

## 5. Eksperimen, urut kronologis

### 5.1 Screening 1: 3 dataset, 1 seed, protokol runner asli

Tujuh konfigurasi, dinilai dengan rata-rata peringkat validation di 9 kondisi:

| Konfigurasi | Peringkat validation |
|---|---:|
| `gibbs_k5_s50` (k = 5, strong 0,5, weak identitas) | **3.2** |
| `gibbs_k20_s50` | 3.9 |
| `gibbs_k20_s30` | 4.1 |
| `gibbs_k5_s30` | 4.1 |
| `gibbs_k5_w10_s50` (weak 0,1) | 4.3 |
| Tanpa SSL | 4.6 |

Catatan: baris `scarf_w0_s50` di screening ini tercemar bug patch modul (sudah diperbaiki) dan tidak dipakai.

### 5.2 Protokol RankUp-setia (EMA + `uratio` 7)

- **Percobaan pertama gagal karena bug.** EMA 0,999 tanpa warm-up selalu menghasilkan QWK validation 0: model EMA masih didominasi bobot acak, dan patience habis.
- **Setelah warm-up `decay = min(0.999, (1+t)/(10+t))`**, kurva validation menunjukkan masalah yang sebenarnya:
  - NHANES 1%: QWK 0,18 → 0,12, turun setelah epoch 2.
  - Superconductivity 1%: MSE memburuk sejak epoch 1–2.
  - Superconductivity 30%: masih membaik di epoch 120.

  EMA tidak menyelesaikan overfit awal, jadi protokol ini tidak dilanjutkan. Run yang tercemar bug disimpan di `hasil/arsip/`.

### 5.3 Usulan vs baseline Optuna, 3 seed

`gibbs_k5_s50`, seed 42/123/456, dibandingkan dengan seed yang sama dari run lama. Rata-rata peringkat semua metrik:

| Metode | Peringkat |
|---|---:|
| **Usulan** | **3.90** |
| SCARF (Optuna) | 4.02 |
| TabCutMix+ (Optuna) | 4.13 |
| Gaussian (Optuna) | 4.42 |
| TabCutMix (Optuna) | 4.47 |
| Tanpa SSL | 4.87 |
| Zero-out (Optuna) | 4.93 |
| RLA (Optuna) | 5.27 |

Usulan kalah jelas di Pen Digits: menang hanya 11–22% pasangan, p ≈ 0,02. Keunggulan peringkat 1 ini ternyata noise (lihat §5.7).

### 5.4 Kalibrasi strength tanpa label dan augmentasi data berlabel

Strength dikalibrasi supaya pergeseran relatif E‖x̃−x‖² / E‖x−x′‖² ≈ ρ. Dengan strength yang sama (0,5), pergeserannya sangat berbeda antar dataset: NHANES 0,26, Pen Digits 0,07, Superconductivity 0,045. Dengan ρ = 0,05, strength hasil kalibrasi: NHANES 0,10, Pen Digits 0,44, Superconductivity 0,58.

| Konfigurasi | Peringkat validation (1 seed) |
|---|---:|
| `cal05` (ρ = 0,05) | **2.3** |
| `gibbs_k5_s50` | 2.9 |
| Tanpa SSL | 4.4 |
| `cal05_lab` (data berlabel ikut diaugmentasi) | 6.4 |
| `gibbs_k5_s50_lab` | 6.4 |
| `cal15_lab` | 6.8 |

Augmentasi data berlabel merugikan, jadi ditolak. `cal05` pada 3 seed:
- peringkat per kondisi 4,08, seri peringkat 1 dengan TabCutMix;
- peringkat semua metrik 4,27;
- MSE Superconductivity 1% 354,6, terbaik;
- memperbaiki kekalahan Pen Digits, tapi MAE NHANES memburuk.

### 5.5 Perbaikan protokol: lr, bobot SSL, cutoff (1 seed, validation)

| Varian (dengan `cal05`) | Peringkat validation |
|---|---:|
| Default (lr 2e-3, bobot unlabeled 0,2, cutoff 0,95) | **2.0** |
| Bobot unlabeled 1,0 + cutoff 0,8 | 2.2 |
| lr 5e-4 + bobot 1,0 + cutoff 0,8 | 3.4 |
| lr 5e-4 | 3.7 |
| lr 5e-4 + bobot 1,0 | 3.7 |

Learning rate kecil membuat model kurang terlatih. SSL yang lebih agresif menang di 5 dari 9 sel tapi kalah di sisanya.

**NHANES 30% ternyata jenuh, bukan overfit.** Gradient boosting supervised hanya mencapai QWK validation 0,28 (Ridge 0,25), sedangkan TabM sudah ±0,33 di epoch 1. Data ini mentok di plafon prediktabilitasnya.

### 5.6 Threshold adaptif FreeMatch (SAT)

SAT: τ_t = 0,999·τ_{t−1} + 0,001·(rata-rata confidence teacher), dimulai dari 1/C. Di validation 1 seed, `cal05_sat_uw1` (SAT + bobot unlabeled 1,0) terbaik di label 1%. Varian ini dipilih dari beberapa varian yang seri di validation (peringkat 3,0 vs 3,0–3,2), atas dasar teori.

Hasil 3 seed: peringkat semua metrik **4,63**, lebih buruk dari `cal05` (4,27). Kasus paling mencolok, NHANES 1%: QWK validation 0,311 (terbaik), tapi test 0,225 (terburuk). Ini bukti bahwa **validation 1 seed terlalu noisy untuk memilih di antara varian yang hampir seri.**

### 5.7 Konfirmasi 5 seed: `gibbs_k5_s50` vs baseline Optuna

| Metode | Peringkat, semua metrik |
|---|---:|
| TabCutMix+ (Optuna) | **3.63** |
| SCARF (Optuna) | 4.00 |
| **Usulan (tanpa tuning)** | 4.17 |
| Tanpa SSL | 4.33 |
| Gaussian (Optuna) | 4.45 |
| TabCutMix (Optuna) | 4.98 |
| RLA (Optuna) | 5.05 |
| Zero-out (Optuna) | 5.38 |

Metrik utama, rata-rata 5 seed (usulan / terbaik lain):

| Dataset | 1% | 10% | 30% |
|---|---|---|---|
| NHANES QWK ↑ | 0.251 / 0.277 | 0.255 / 0.258 | **0.310** / 0.305 |
| Pen Digits macro-F1 ↑ | 0.866 / 0.897 | 0.953 / 0.967 | 0.970 / 0.976 |
| Superconductivity MSE ↓ | **366.5** / 369.7 | 205.9 / 205.1 | 150.0 / 146.2 |

Wilcoxon berpasangan (fraksi label × seed, n = 15):
- **Usulan > tanpa SSL**: NHANES QWK p = 0,015; Pen Digits F1 p = 0,010.
- **Usulan > TabCutMix**: NHANES QWK p = 0,030.
- **Usulan < TabCutMix, TabCutMix+, Gaussian** di Pen Digits (p ≤ 0,03).
- Superconductivity: tidak ada selisih signifikan.

File: `hasil/usulan-final-ranks.csv`, `hasil/usulan-final-tests.csv`.

### 5.8 Uji 13 dataset: 1 seed, tanpa Optuna

- **Dataset (13).**
  - Ordinal: NHANES, LEV, ESL, Student Performance, Melbourne Airbnb.
  - Klasifikasi: Pen Digits, Satimage, Diabetes Pima, CMC, Bank Marketing, Adult Income.
  - Regresi: Superconductivity, Grid Stability (target `stab`; `stabf` dan `p1` dibuang).
- **Kondisi.** Label 1/10/30%, total 36 kondisi. LEV, ESL, dan Student Performance di label 1% tidak memungkinkan, karena 3–7 label tidak cukup untuk mewakili setiap level ordinal.
- **Setting identik untuk semua augmentasi:** weak = identitas, strong = 0,5 (σ = 0,5 untuk Gaussian), seed 42. Setiap baseline dijalankan dengan `fit_model` modulnya sendiri.
- **Total:** 288 run, ±1,6 jam T4.

| Metode | Peringkat metrik utama | Peringkat metrik lain | Jumlah juara 1 | Ordinal | Klasifikasi | Regresi |
|---|---:|---:|---:|---:|---:|---:|
| Zero-out | **4.03** | 4.32 | 6 | 2.83 | 4.56 | 4.83 |
| Gaussian | 4.15 | 4.45 | 5 | 3.33 | 4.19 | 5.67 |
| RLA | 4.25 | 4.48 | 5 | 5.71 | 3.64 | 3.17 |
| TabCutMix | 4.32 | 4.53 | 3 | 4.71 | 4.17 | 4.00 |
| TabCutMix+ | 4.32 | 4.52 | 2 | 4.46 | 4.17 | 4.50 |
| SCARF | 4.69 | 4.25 | 7 | 4.29 | 4.92 | 4.83 |
| Tanpa SSL | 4.96 | 4.83 | 3 | 5.21 | 4.94 | 4.50 |
| **Block-Gibbs (usulan)** | 5.28 | 4.62 | 2 | 5.46 | 5.42 | 4.50 |

**Friedman** (metrik utama, 36 kondisi, 8 metode): χ² = 8,29, **p = 0,31**. Tidak ada metode yang berbeda signifikan secara keseluruhan.

Usulan vs tiap pembanding, per kondisi (metrik utama):

| Pembanding | Usulan menang | Seri | Usulan kalah | p Wilcoxon |
|---|---:|---:|---:|---:|
| Gaussian | 14 | 1 | 21 | 0.094 |
| RLA | 14 | 2 | 20 | 0.098 |
| SCARF | 15 | 1 | 20 | 0.195 |
| TabCutMix | 10 | 2 | 24 | 0.094 |
| TabCutMix+ | 12 | 1 | 23 | 0.134 |
| Zero-out | 12 | 1 | 23 | **0.021** |
| Tanpa SSL | 16 | 2 | 18 | 0.723 |

Metrik utama per dataset (nilai test, peringkat dalam kurung):

| Tugas | Dataset | % | Tanpa SSL | **Usulan** | TabCutMix | SCARF | RLA | Gaussian | Zero-out | TabCutMix+ |
|---|---|---|---|---|---|---|---|---|---|---|
| klasifikasi | adult_income | 1 | 0.7824 (5) | 0.7782 (8) | 0.7829 (3) | 0.7831 (1) | 0.783 (2) | 0.7826 (4) | 0.7819 (7) | 0.7823 (6) |
| klasifikasi | adult_income | 10 | 0.788 (7) | 0.7901 (3) | 0.7872 (8) | 0.7974 (1) | 0.7887 (6) | 0.7915 (2) | 0.7893 (4) | 0.7891 (5) |
| klasifikasi | adult_income | 30 | 0.8003 (4) | 0.797 (7) | 0.7986 (5) | 0.7974 (6) | 0.8017 (2) | 0.8022 (1) | 0.7959 (8) | 0.801 (3) |
| klasifikasi | bank_marketing | 1 | 0.6734 (3) | 0.6399 (7) | 0.6551 (5) | 0.6415 (6) | 0.6327 (8) | 0.6914 (1) | 0.6752 (2) | 0.6704 (4) |
| klasifikasi | bank_marketing | 10 | 0.7404 (2) | 0.7362 (5) | 0.7422 (1) | 0.7364 (4) | 0.7198 (8) | 0.7204 (7) | 0.723 (6) | 0.7397 (3) |
| klasifikasi | bank_marketing | 30 | 0.7577 (4) | 0.7634 (1) | 0.761 (2) | 0.7525 (8) | 0.7569 (5) | 0.7554 (7) | 0.7594 (3) | 0.7558 (6) |
| klasifikasi | cmc | 1 | 0.3326 (6) | 0.3152 (7) | 0.3589 (2) | 0.2903 (8) | 0.3729 (1) | 0.3534 (3) | 0.3446 (5) | 0.351 (4) |
| klasifikasi | cmc | 10 | 0.4586 (1) | 0.442 (8) | 0.4468 (6) | 0.4471 (5) | 0.444 (7) | 0.4485 (3) | 0.4484 (4) | 0.4577 (2) |
| klasifikasi | cmc | 30 | 0.491 (4) | 0.4896 (5) | 0.4922 (3) | 0.4875 (6) | 0.4956 (1) | 0.4842 (7) | 0.4842 (7) | 0.4956 (1) |
| klasifikasi | diabetes_pima | 1 | 0.6174 (7) | 0.593 (8) | 0.696 (3) | 0.6395 (6) | 0.7181 (2) | 0.6548 (5) | 0.672 (4) | 0.728 (1) |
| klasifikasi | diabetes_pima | 10 | 0.764 (4) | 0.7517 (6) | 0.7548 (5) | 0.7517 (6) | 0.7783 (1) | 0.7359 (8) | 0.7701 (2) | 0.7701 (2) |
| klasifikasi | diabetes_pima | 30 | 0.681 (8) | 0.7044 (5) | 0.7149 (2) | 0.6888 (7) | 0.7092 (4) | 0.7017 (6) | 0.7308 (1) | 0.7122 (3) |
| klasifikasi | pendigits | 1 | 0.8377 (8) | 0.86 (6) | 0.869 (4) | 0.8392 (7) | 0.8773 (1) | 0.8709 (3) | 0.8766 (2) | 0.8672 (5) |
| klasifikasi | pendigits | 10 | 0.9473 (6) | 0.9525 (2) | 0.9386 (8) | 0.9496 (4) | 0.9552 (1) | 0.9513 (3) | 0.9449 (7) | 0.9492 (5) |
| klasifikasi | pendigits | 30 | 0.967 (7) | 0.9687 (6) | 0.9692 (5) | 0.9634 (8) | 0.9813 (1) | 0.9745 (2) | 0.971 (4) | 0.9711 (3) |
| klasifikasi | satimage | 1 | 0.7758 (5) | 0.7732 (6) | 0.7825 (4) | 0.7869 (3) | 0.7668 (7) | 0.7923 (2) | 0.7943 (1) | 0.7646 (8) |
| klasifikasi | satimage | 10 | 0.8313 (2) | 0.8298 (3) | 0.8231 (7) | 0.8337 (1) | 0.8279 (5) | 0.8289 (4) | 0.8247 (6) | 0.8129 (8) |
| klasifikasi | satimage | 30 | 0.8577 (6) | 0.8627 (4) | 0.8654 (2) | 0.8797 (1) | 0.865 (3) | 0.8575 (7) | 0.8572 (8) | 0.8608 (5) |
| ordinal | esl | 10 | 0.9023 (3) | 0.9037 (1) | 0.8991 (6) | 0.8991 (6) | 0.8991 (6) | 0.8991 (6) | 0.9033 (2) | 0.8991 (6) |
| ordinal | esl | 30 | 0.902 (5) | 0.902 (5) | 0.902 (5) | 0.9069 (1) | 0.902 (5) | 0.902 (5) | 0.902 (5) | 0.902 (5) |
| ordinal | lev | 10 | 0.65 (7) | 0.6322 (8) | 0.6737 (4) | 0.6737 (4) | 0.6737 (4) | 0.6809 (2) | 0.6857 (1) | 0.6737 (4) |
| ordinal | lev | 30 | 0.6602 (6) | 0.6602 (6) | 0.6602 (6) | 0.6616 (2) | 0.6602 (6) | 0.6616 (2) | 0.6616 (2) | 0.6616 (2) |
| ordinal | melbourne_airbnb | 1 | 0.7388 (8) | 0.7484 (5) | 0.7786 (1) | 0.7457 (6) | 0.7395 (7) | 0.758 (3) | 0.7564 (4) | 0.7676 (2) |
| ordinal | melbourne_airbnb | 10 | 0.794 (7) | 0.7936 (8) | 0.8019 (1) | 0.7955 (5) | 0.7958 (4) | 0.7969 (2) | 0.7964 (3) | 0.7941 (6) |
| ordinal | melbourne_airbnb | 30 | 0.833 (3) | 0.833 (2) | 0.832 (7) | 0.8329 (4) | 0.832 (6) | 0.8312 (8) | 0.836 (1) | 0.8327 (5) |
| ordinal | nhanes | 1 | 0.1503 (8) | 0.2383 (4) | 0.2314 (5) | 0.2424 (2) | 0.1908 (7) | 0.2293 (6) | 0.2511 (1) | 0.2391 (3) |
| ordinal | nhanes | 10 | 0.2404 (4) | 0.2198 (8) | 0.2323 (6) | 0.2405 (3) | 0.2446 (2) | 0.2473 (1) | 0.2259 (7) | 0.2327 (5) |
| ordinal | nhanes | 30 | 0.3311 (3) | 0.3291 (4) | 0.3239 (6) | 0.3239 (6) | 0.3239 (6) | 0.3335 (1) | 0.3335 (1) | 0.3239 (6) |
| ordinal | student_performance | 10 | 0.1702 (1) | 0.1291 (8) | 0.138 (6) | 0.138 (6) | 0.138 (6) | 0.1552 (2) | 0.1552 (3) | 0.1415 (4) |
| ordinal | student_performance | 30 | 0.1937 (7) | 0.2109 (6) | 0.2417 (2) | 0.2204 (5) | 0.1927 (8) | 0.2453 (1) | 0.2311 (3) | 0.23 (4) |
| regresi | grid_stability | 1 | 3.953e-04 (6) | 3.943e-04 (5) | 3.786e-04 (2) | 3.999e-04 (7) | 3.897e-04 (3) | 3.910e-04 (4) | 4.042e-04 (8) | 3.730e-04 (1) |
| regresi | grid_stability | 10 | 6.410e-05 (1) | 6.954e-05 (5) | 6.925e-05 (4) | 7.074e-05 (7) | 6.676e-05 (3) | 7.128e-05 (8) | 7.043e-05 (6) | 6.649e-05 (2) |
| regresi | grid_stability | 30 | 4.336e-05 (3) | 4.328e-05 (2) | 4.374e-05 (5) | 4.444e-05 (7) | 4.355e-05 (4) | 4.521e-05 (8) | 4.292e-05 (1) | 4.386e-05 (6) |
| regresi | superconductivity | 1 | 378.5 (8) | 377.4 (7) | 374.6 (4) | 372.5 (1) | 373.6 (2) | 375.0 (5) | 375.6 (6) | 374.4 (3) |
| regresi | superconductivity | 10 | 197.4 (4) | 197.7 (5) | 208.3 (7) | 197.8 (6) | 197.0 (3) | 194.4 (1) | 196.9 (2) | 208.7 (8) |
| regresi | superconductivity | 30 | 150.5 (5) | 149.5 (3) | 149.1 (2) | 146.8 (1) | 150.4 (4) | 152.1 (8) | 150.7 (6) | 151.1 (7) |

Nilai yang identik di banyak kolom (ESL, LEV, NHANES 30%) muncul karena checkpoint terbaik terpilih sebelum augmentasi berpengaruh, persis seperti diagnosis §3.2. File: `hasil/usulan-semua/` (`results.jsonl`, `ranks_<split>_<metode>.csv`, `wilcoxon_<split>_<metode>.csv`, `ringkasan.md`).

### 5.9 Varian dengan diversity lebih tinggi (13 dataset, 1 seed)

Hipotesis §6 butir 3 diuji langsung dengan tiga varian yang tetap menjaga P(x) tapi menggeser data lebih jauh:
- **`gibbs_t3`:** 3 langkah Gibbs berurutan. Komposisi kernel yang masing-masing menjaga P tetap menjaga P.
- **`gibbs_k50`:** tetangga k = 50.
- **`gibbs_rho20`:** jumlah langkah dikalibrasi tanpa label ke pergeseran relatif 0,2, maksimal 10 langkah.

Pergeseran relatif untuk Superconductivity naik dari 0,047 (1 langkah) ke 0,098 (3 langkah) dan 0,114 (k = 50). Aturan pemilihan ditetapkan sebelum melihat hasil: varian dengan rata-rata peringkat validation terbaik di 36 kondisi. Total 108 run.

Peringkat metrik utama, 11 metode:

| Metode | Validation | Test |
|---|---:|---:|
| Zero-out | 5.93 | **5.07** |
| Gaussian | 5.81 | 5.19 |
| TabCutMix+ | 5.47 | 5.33 |
| RLA | 4.81 | 5.33 |
| TabCutMix | **4.10** | 5.54 |
| SCARF | 5.68 | 5.90 |
| Tanpa SSL | 6.03 | 6.32 |
| Block-Gibbs (`gibbs_k5_s50`) | 6.21 | 6.78 |
| `gibbs_rho20` | 6.49 | 6.78 |
| `gibbs_t3` | 8.25 | 6.85 |
| `gibbs_k50` | 7.24 | 6.90 |

- **Tidak ada varian baru yang lebih baik dari Block-Gibbs asli di validation.** Makin banyak langkah, makin buruk. Hipotesis "diversity rendah" ditolak.
- `gibbs_rho20` seri dengan versi asli di 27 dari 36 kondisi, karena kalibrasi memilih 1 langkah di banyak dataset.
- **Friedman di test** (11 metode): p = 0,045. Signifikansi ini terutama datang dari keempat varian Block-Gibbs dan tanpa-SSL yang mengelompok di bawah. Kelompok teratas (Zero-out sampai TabCutMix, 5,07–5,54) saling berdekatan.
- **Kesesuaian validation dan test.** Korelasi peringkat metode (rata-rata atas kondisi) 0,73, tapi korelasi per kondisi hanya 0,39. Pemenang validation (TabCutMix) hanya peringkat 5 di test. Pada tingkat sel, validation 1 seed tidak andal.

### 5.10 RankUp vs metode supervised dan SSL tabular lain (3 dataset utama, seed 42)

Script `bandingkan_ssl.py`; hasil di `hasil/pembanding/` (`semua.jsonl`, `peringkat.csv`).

**Pembanding:**
- **Supervised:** HistGradientBoosting (HGB), Ridge/Logistic. Hanya memakai data berlabel, tanpa tuning.
- **SSL klasik:**
  - self-training HGB (klasifikasi: `SelfTrainingClassifier` threshold 0,9; regresi: 3 putaran, 10% pseudo-label dengan variansi bootstrap terkecil);
  - COREG (Zhou & Li, 2005), dua kNN regressor dengan p = 2 dan p = 5, k = 3;
  - Label Spreading (graf kNN, k = 10), khusus klasifikasi.
- **SSL deep** (runner Task 3): VIME, SSLAE, GFTab, STUNT.
  - Split Task 3 terbukti identik dengan split 4-augmentasi-input (array fitur dan target sama persis), jadi hasil NHANES 10% dipakai ulang.
  - Lima kondisi lainnya (NHANES 1/30%, Pen Digits 1/10/30%) dijalankan dengan `modal_pembanding.py` memakai image dan setting default Task 3, seed 42.
  - Runner ini hanya untuk klasifikasi dan ordinal, sehingga tidak bisa dipakai di Superconductivity.

**RankUp:** 6 augmentasi dengan kekuatan hasil Optuna (seed 42 dari run 5-seed), 6 augmentasi tanpa tuning, Block-Gibbs, dan TabM tanpa SSL.

Metrik utama test:

| Metode | NHANES 1% | 10% | 30% | Pen Digits 1% | 10% | 30% | Supercond. 1% ↓ | 10% ↓ | 30% ↓ |
|---|---|---|---|---|---|---|---|---|---|
| RankUp terbaik (dari 14 varian) | **0.2856** | 0.2540 | **0.3335** | 0.9066 | **0.9672** | **0.9813** | **370.39** | 194.42 | 146.19 |
| TabM tanpa SSL | 0.1503 | 0.2404 | 0.3311 | 0.8377 | 0.9473 | 0.9670 | 378.55 | 197.41 | 150.47 |
| HGB (supervised) | 0.0000 | 0.2849 | 0.2900 | 0.7473 | 0.9040 | 0.9543 | 374.47 | **178.13** | 143.40 |
| Self-training HGB | 0.0106 | **0.2910** | 0.2932 | 0.7656 | 0.9148 | 0.9558 | 380.64 | 183.38 | **142.70** |
| COREG | 0.1730 | 0.1986 | 0.2474 | – | – | – | 477.24 | 263.97 | 191.40 |
| Label Spreading | – | – | – | **0.9179** | 0.9650 | 0.9671 | – | – | – |
| Ridge / Logistic | 0.0828 | 0.2050 | 0.2450 | 0.8113 | 0.8852 | 0.9006 | 437.79 | 357.96 | 341.48 |
| SSLAE | 0.1431 | 0.2535 | 0.2966 | 0.6578 | 0.9088 | 0.9255 | – | – | – |
| GFTab | 0.0268 | 0.1782 | 0.2868 | 0.2639 | 0.8583 | 0.9199 | – | – | – |
| VIME | 0.0294 | 0.1723 | 0.2271 | 0.8118 | 0.8775 | 0.8948 | – | – | – |
| STUNT | 0.1440 | 0.1631 | 0.1860 | 0.6645 | 0.6645 | 0.5280 | – | – | – |

**Pemenang per kondisi:** RankUp 5, pembanding 4.
- **Pembanding menang di:**
  - NHANES 10% (self-training HGB);
  - Pen Digits 1% (Label Spreading);
  - Superconductivity 10% (HGB supervised; MSE 178 vs 194);
  - Superconductivity 30% (self-training HGB).
- **Bias yang menguntungkan RankUp:** baris "RankUp terbaik" adalah nilai maksimum dari 14 varian **di test**. Pembanding hanya punya 1–4 varian, tanpa tuning. Dengan keuntungan itu pun, RankUp hanya menang 5 dari 9 kondisi.
- **Bila dibandingkan satu lawan satu dengan varian RankUp tertentu**, gradient boosting tanpa SSL mengalahkan setiap varian RankUp di Superconductivity 10% dan 30% serta NHANES 10%.
- **Kekuatan RankUp/TabM ada di label paling sedikit dan klasifikasi** (NHANES 1%, Pen Digits 10–30%). Di NHANES 1%, HGB gagal total (QWK 0, prediksi konstan dari 36 label).
- **SSL deep tabular kalah dari RankUp di keenam kondisi NHANES dan Pen Digits.** SSLAE paling kuat di antara keempatnya, tapi tetap di bawah RankUp terbaik di setiap kondisi. Di NHANES 1%, VIME dan GFTab hampir gagal (QWK 0,03); di Pen Digits 1%, GFTab hanya 0,26. Keempat runner ini memakai setting default Task 3 tanpa tuning, sama seperti pembanding klasik.

## 6. Interpretasi

1. **Selisih antar-augmentasi lebih kecil dari noise.** Friedman p = 0,31 di 13 dataset, dan sebagian besar Wilcoxon tidak signifikan. Peringkat pada satu seed atau satu dataset tidak bisa digeneralisasi. Inilah sumber kesan "tidak stabil" di eksperimen awal.
2. **Banyak sel tidak sensitif terhadap augmentasi.** Penyebabnya bisa overfit sebelum SSL aktif (label 1%), plafon data (NHANES 30%), atau target diskret yang jenuh (ESL, LEV). Di sel seperti ini, perbandingan augmentasi tidak informatif.
3. **Affinity tinggi saja tidak cukup** (hipotesis awal; lihat butir 6 untuk revisinya). Block-Gibbs unggul di semua ukuran affinity: realism, korelasi, dan label-kept (§4.4). Tapi di 13 dataset hasilnya setara dengan tanpa-SSL (16 menang, 18 kalah). Ini konsisten dengan temuan Gontijo-Lopes dkk. (2021) bahwa manfaat augmentasi ditentukan oleh **affinity dan diversity** sekaligus. Strong view usulan menggeser data sangat sedikit (pergeseran relatif 0,045–0,07 pada strength 0,5 untuk Superconductivity dan Pen Digits), sehingga konsistensi weak→strong memberi sinyal lemah. Zero-out dan Gaussian yang kasar justru memberi keragaman lebih besar dan unggul tipis di 13 dataset.
4. **Validation 1 seed tidak andal** untuk memilih di antara varian yang hampir seri (§5.6).
5. **Augmentasi data berlabel dan lr kecil merugikan**; SSL agresif (bobot 1,0, cutoff 0,8, SAT) bergantung pada dataset.
6. **Revisi setelah §5.9.** Menambah diversity Block-Gibbs justru memperburuk hasil. Jadi masalahnya bukan jarak pergeseran, melainkan *arah*-nya. Augmentasi yang berhasil (Zero-out, Gaussian, RLA, TabCutMix) semuanya merusak data ke arah yang tidak realistis (*off-manifold*), sedangkan Block-Gibbs selalu tetap di distribusi data. Hipotesis yang konsisten dengan data: pada RankUp tabular, manfaat konsistensi weak→strong terutama datang dari **regularisasi terhadap noise**, yaitu memaksa model mulus di arah yang tidak terlihat di data. Ini sejalan dengan hasil klasik bahwa training dengan noise input setara dengan regularisasi Tikhonov (Bishop, 1995). Manfaatnya bukan dari invariansi semantik antar-sampel realistis. View yang tetap di distribusi data memberi sinyal yang hampir sama dengan data berlabel itu sendiri, sehingga hasilnya setara tanpa-SSL. Hipotesis ini belum diuji secara langsung.

## 7. Klaim yang bisa dan tidak bisa dibuat

**Bisa:**
- Block-Gibbs adalah augmentasi tabular dengan jaminan invariansi distribusi (Proposisi 1–3), dan secara empiris paling realistis serta paling menjaga label dibanding SCARF, Gaussian, Zero-out, dan random swap (§4.4).
- Tanpa tuning, usulan kompetitif dengan baseline yang di-tuning Optuna 40 trial pada 3 dataset utama (peringkat 3 dari 8, 5 seed). Usulan terbaik di NHANES 30% dan Superconductivity 1%, dan signifikan lebih baik dari tanpa-SSL di NHANES dan Pen Digits.
- Diagnosis: pada RankUp tabular dengan jadwal training ini, perbedaan augmentasi sebagian besar tertutup oleh early stopping, overfit, dan plafon data.

**Tidak bisa:**
- "Usulan unggul di semua metrik atau dataset": dibantah oleh §5.7, §5.8, dan §5.9.
- "Usulan lebih baik dari tanpa-SSL secara umum": tidak signifikan di 13 dataset.
- "RankUp lebih baik dari metode tabular lain": gradient boosting supervised atau self-training menang di 4 dari 9 kondisi (§5.10).
- "Ada satu augmentasi terbaik": kelompok teratas (Zero-out, Gaussian, TabCutMix(+), RLA) saling berdekatan, dan pemenangnya berganti antara validation dan test serta antar tugas.

## 8. Saran lanjutan

1. ~~Menambah diversity tanpa mengorbankan affinity~~: sudah diuji di §5.9 dan tidak membantu.
1. **Menguji hipotesis regularisasi noise (§6 butir 6)** secara langsung. Bandingkan augmentasi off-manifold dan on-manifold dengan besar pergeseran yang sama, dan ukur kemulusan model (misalnya norma gradien input).
2. **Memperbaiki jadwal training** supaya SSL sempat bekerja di label 1%: warm-up bobot supervised, atau menghitung epoch per putaran data berlabel.
3. **Evaluasi multi-seed untuk setiap klaim**, minimal 3–5 seed, karena satu seed tidak cukup.

## 9. Reproduksi

Semua perintah dijalankan dari folder `4-augmentasi-input`, dengan `MODAL_PROFILE=lagi`.

```bash
# Uji operator (lokal, tanpa GPU)
uv run python eksperimen/analisis_operator.py
# Tes operator di Modal
uv run modal run eksperimen/modal_usulan.py --action check
# Usulan 5 seed pada 3 dataset utama + ringkasan vs baseline lama
uv run modal run eksperimen/modal_usulan.py --action train --names gibbs_k5_s50,none --seeds 42,123,456,789,101112 --out usulan-screening
uv run python -c "import sys; sys.path.insert(0,'eksperimen'); import ringkas_usulan as r; r.final('eksperimen/hasil/usulan-screening/results.jsonl')"
# 13 dataset: siapkan split, lalu jalankan dan ringkas
for p in 01 10 30; do for d in lev esl student_performance melbourne_airbnb diabetes_pima cmc satimage bank_marketing adult_income grid_stability; do uv run python prepare_data.py --config config/datasets_semua.json --dataset $d --labeled-fraction 0.$p --output eksperimen/hasil/data-semua/label_$p; done; done
uv run modal run eksperimen/modal_usulan.py --action train --datasets semua --names none,gibbs_k5_s50,def_tabcutmix,def_scarf,def_range_limited,def_gaussian,def_zero_out,def_tabcutmix_plus --seeds 42 --out usulan-semua
uv run python eksperimen/ringkas_semua.py eksperimen/hasil/usulan-semua/results.jsonl test
uv run python eksperimen/ringkas_semua.py eksperimen/hasil/usulan-semua/results.jsonl validation
```

| File | Isi |
|---|---|
| `experiments/usulan.py` | Operator Block-Gibbs, rantai Gibbs, kalibrasi ρ dan jumlah langkah, SAT, dan loop RankUp-setia (EMA) |
| `modal_usulan.py` | App Modal: semua konfigurasi (`GRID`), dataset, dan pengumpulan hasil |
| `tests/test_usulan.py` | Tiga tes operator |
| `analisis_operator.py` | Uji realism, korelasi, dan label tanpa training |
| `ringkas_usulan.py`, `ringkas_semua.py` | Ringkasan peringkat dan uji statistik |
| `config/datasets_semua.json` | Config 10 dataset tambahan |
| `hasil/usulan-screening/` | 3 dataset utama: screening, 3 seed, 5 seed |
| `hasil/usulan-protokol/` | Varian protokol (lr, bobot, cutoff, SAT) |
| `hasil/usulan-semua/` | 13 dataset, 1 seed |
| `bandingkan_ssl.py`, `modal_pembanding.py` | Pembanding supervised dan SSL tabular (§5.10) |
| `hasil/pembanding/` | Hasil pembanding |

## 10. Referensi

- Huang, Fu, Tsao. *RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier*. NeurIPS 2024. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c26a8494fe31695db965ae8b7244b7c1-Paper-Conference.pdf), [kode](https://github.com/pm25/Semi-Supervised-Regression)
- Sohn dkk. *FixMatch*. NeurIPS 2020. [arXiv](https://arxiv.org/abs/2001.07685)
- Wang dkk. *FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning*. ICLR 2023 (sumber SAT).
- Bahri dkk. *SCARF*. ICLR 2022. [arXiv](https://arxiv.org/pdf/2106.15147)
- Yoon dkk. *VIME*. NeurIPS 2020.
- Ucar dkk. *SubTab*. NeurIPS 2021.
- Fang dkk. *TabCutMix*. ICML 2025. [paper](https://proceedings.mlr.press/v267/fang25f.html)
- Lee dkk. *Range-limited Augmentation (FeSTA)*. KDD 2025. [kode](https://github.com/kyungeun-lee/festa)
- Yao dkk. *C-Mixup: Improving Generalization in Regression*. NeurIPS 2022. [arXiv](https://ar5iv.labs.arxiv.org/html/2210.05775)
- Schneider dkk. *Anchor Data Augmentation*. NeurIPS 2023. [arXiv](https://arxiv.org/pdf/2311.06965)
- Baena dkk. *Local Mixup*. 2022. [arXiv](https://www.arxiv.org/pdf/2201.04368)
- Yang dkk. *Sample Efficiency of Data Augmentation Consistency Regularization*. AISTATS 2023. [arXiv](https://arxiv.org/pdf/2202.12230.pdf)
- Gontijo-Lopes, Smullin, Cubuk, Dyer. *Tradeoffs in Data Augmentation: An Empirical Study* (affinity & diversity). ICLR 2021.
- Zhou & Li. *Semi-Supervised Regression with Co-Training (COREG)*. IJCAI 2005.
- Zhou dkk. *Learning with Local and Global Consistency* (Label Spreading). NeurIPS 2004.
- Yarowsky 1995; Lee 2013 (*pseudo-label / self-training*).
- Yoon dkk. *VIME*. NeurIPS 2020; Nam dkk. *STUNT*. ICLR 2023; Lee dkk. *GFTab*. AAAI 2025; *SSLAE*. Machine Learning 2025.
- Bishop. *Training with Noise is Equivalent to Tikhonov Regularization*. Neural Computation 1995.
- Geman & Geman. *Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images*. IEEE TPAMI 1984.
- Stone. *Consistent Nonparametric Regression*. Annals of Statistics 1977.
- `ConditionalKNNSampler`, paket xplainfi. [dokumentasi](https://rdrr.io/cran/xplainfi/man/ConditionalKNNSampler.html)
- Romano, Sesia, Candès. *Deep Knockoffs*; knockoff dengan Gibbs sampler. [arXiv](https://arxiv.org/pdf/2208.07959)
- Rabbani dkk. *Attention versus Contrastive Learning of Tabular Data*. 2024. [arXiv](https://arxiv.org/html/2401.04266v1)
- Cui dkk. *Tabular Data Contrastive Learning via Class-Conditioned and Feature-Correlation Based Augmentation*. 2024. [arXiv](https://arxiv.org/html/2404.17489v2)
- Jurek dkk. *SeBA*. 2026. [arXiv](https://arxiv.org/pdf/2605.08519)
- Gorishniy dkk. *TabM*. [arXiv](https://arxiv.org/abs/2410.24210)
