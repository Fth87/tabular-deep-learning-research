# Perbandingan validation metode SSL

**Tahap 4 sampai 8**, validation dengan seed **42, 123, dan 456** pada split yang sama. Varian TabPFN memakai versi 3.5. Peringkat dihitung ulang untuk varian non-Optuna.

[NHANES](#nhanes) · [Pen Digits](#pen-digits) · [Superconductivity](#superconductivity) · [QWK terkalibrasi](#qwk-terkalibrasi) · [Rerata peringkat](#rerata-peringkat) · [Seluruh metode](#seluruh-metode)

**TabPFN 3.5 · distilasi tanpa augmentasi · protokol latent** berada di rank validation gabungan **#1**. Varian terbaik dari eksperimen konteks, **TabPFN 3.5 · distilasi + ICS 70% pada data berlabel dan tanpa label**, berada di **#2**.

## Perbandingan hasil

**NHANES** QWK mentah ↑ · **Pen Digits** macro-F1 ↑ · **Superconductivity** MSE ↓

| Kondisi | Metode rank #1 | Skor validation |
|---|---|---:|
| NHANES 1% | RankUp + SCARF di input + latent | 0.3222 |
| NHANES 10% | TabPFN 3.5 · distilasi + rotasi angular · protokol latent | 0.3624 |
| NHANES 30% | TabPFN 3.5 · distilasi + SCARF · protokol latent | 0.4180 |
| Pen Digits 1% | RankUp-CLS + Gaussian di latent (cos ≈ 0,4) | 0.9476 |
| Pen Digits 10% | TabPFN 3.5 · distilasi + SCARF · protokol latent | 0.9918 |
| Pen Digits 30% | TabPFN 3.5 · distilasi + SCARF · protokol latent | 0.9982 |
| Superconductivity 1% | TabPFN 3.5 · distilasi tanpa augmentasi · protokol konteks | 280.84 |
| Superconductivity 10% | TabPFN 3.5 · distilasi + ICS acak (50–99%) | 149.71 |
| Superconductivity 30% | TabPFN 3.5 · distilasi + noise fitur (F-noise) · protokol konteks | 104.33 |

**Grafik** menampilkan rerata dan ±1 simpangan baku dari tiga seed. Hijau menandai eksperimen konteks.

## NHANES

### NHANES dengan 1% label

![NHANES · 1% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/nhanes_01_ringkas.png)

### NHANES dengan 10% label

![NHANES · 10% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/nhanes_10_ringkas.png)

### NHANES dengan 30% label

![NHANES · 30% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/nhanes_30_ringkas.png)

## Pen Digits

### Pen Digits dengan 1% label

![Pen Digits · 1% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/pendigits_01_ringkas.png)

### Pen Digits dengan 10% label

![Pen Digits · 10% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/pendigits_10_ringkas.png)

### Pen Digits dengan 30% label

![Pen Digits · 30% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/pendigits_30_ringkas.png)

## Superconductivity

### Superconductivity dengan 1% label

![Superconductivity · 1% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_01_ringkas.png)

### Superconductivity dengan 10% label

![Superconductivity · 10% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_10_ringkas.png)

### Superconductivity dengan 30% label

![Superconductivity · 30% label · ringkasan. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_30_ringkas.png)

## QWK terkalibrasi

Batas kelas berasal dari validation. Grafik memuat varian eksperimen konteks dan prediksi langsung TabPFN 3.5.

| Kondisi | Metode rank #1 | Skor validation |
|---|---|---:|
| NHANES 1% | TabPFN 3.5 · distilasi + noise fitur (F-noise) · protokol konteks | 0.2884 |
| NHANES 10% | TabPFN 3.5 · distilasi + dropout · protokol konteks | 0.3717 |
| NHANES 30% | TabPFN 3.5 · distilasi + ICS 70% pada data berlabel dan tanpa label | 0.4087 |

![NHANES · 1% label · QWK terkalibrasi. Skor dan rank validation.](grafik_validation_3_seed/nhanes_01_terkalibrasi.png)

![NHANES · 10% label · QWK terkalibrasi. Skor dan rank validation.](grafik_validation_3_seed/nhanes_10_terkalibrasi.png)

![NHANES · 30% label · QWK terkalibrasi. Skor dan rank validation.](grafik_validation_3_seed/nhanes_30_terkalibrasi.png)

## Rerata peringkat

69 metode lengkap pada sembilan kondisi. Rerata rank lebih kecil lebih baik.

![Rerata peringkat pada sembilan kondisi · metode 1–23 dari 69. Skor dan rank validation.](grafik_validation_3_seed/rerata_rank_01.png)

![Rerata peringkat pada sembilan kondisi · metode 24–46 dari 69. Skor dan rank validation.](grafik_validation_3_seed/rerata_rank_02.png)

![Rerata peringkat pada sembilan kondisi · metode 47–69 dari 69. Skor dan rank validation.](grafik_validation_3_seed/rerata_rank_03.png)

## Sumber

[CSV sumber sebelum penyaringan](peringkat_ssl_dan_augmentasi.csv)

## Seluruh metode

### NHANES · 1% label (71 metode)

![NHANES · 1% label · metode 1–23 dari 71. Skor dan rank validation.](grafik_validation_3_seed/nhanes_01_01.png)

![NHANES · 1% label · metode 24–47 dari 71. Skor dan rank validation.](grafik_validation_3_seed/nhanes_01_02.png)

![NHANES · 1% label · metode 48–71 dari 71. Skor dan rank validation.](grafik_validation_3_seed/nhanes_01_03.png)

### NHANES · 10% label (71 metode)

![NHANES · 10% label · metode 1–23 dari 71. Skor dan rank validation.](grafik_validation_3_seed/nhanes_10_01.png)

![NHANES · 10% label · metode 24–47 dari 71. Skor dan rank validation.](grafik_validation_3_seed/nhanes_10_02.png)

![NHANES · 10% label · metode 48–71 dari 71. Skor dan rank validation.](grafik_validation_3_seed/nhanes_10_03.png)

### NHANES · 30% label (69 metode)

![NHANES · 30% label · metode 1–23 dari 69. Skor dan rank validation.](grafik_validation_3_seed/nhanes_30_01.png)

![NHANES · 30% label · metode 24–46 dari 69. Skor dan rank validation.](grafik_validation_3_seed/nhanes_30_02.png)

![NHANES · 30% label · metode 47–69 dari 69. Skor dan rank validation.](grafik_validation_3_seed/nhanes_30_03.png)

### Pen Digits · 1% label (71 metode)

![Pen Digits · 1% label · metode 1–23 dari 71. Skor dan rank validation.](grafik_validation_3_seed/pendigits_01_01.png)

![Pen Digits · 1% label · metode 24–47 dari 71. Skor dan rank validation.](grafik_validation_3_seed/pendigits_01_02.png)

![Pen Digits · 1% label · metode 48–71 dari 71. Skor dan rank validation.](grafik_validation_3_seed/pendigits_01_03.png)

### Pen Digits · 10% label (71 metode)

![Pen Digits · 10% label · metode 1–23 dari 71. Skor dan rank validation.](grafik_validation_3_seed/pendigits_10_01.png)

![Pen Digits · 10% label · metode 24–47 dari 71. Skor dan rank validation.](grafik_validation_3_seed/pendigits_10_02.png)

![Pen Digits · 10% label · metode 48–71 dari 71. Skor dan rank validation.](grafik_validation_3_seed/pendigits_10_03.png)

### Pen Digits · 30% label (69 metode)

![Pen Digits · 30% label · metode 1–23 dari 69. Skor dan rank validation.](grafik_validation_3_seed/pendigits_30_01.png)

![Pen Digits · 30% label · metode 24–46 dari 69. Skor dan rank validation.](grafik_validation_3_seed/pendigits_30_02.png)

![Pen Digits · 30% label · metode 47–69 dari 69. Skor dan rank validation.](grafik_validation_3_seed/pendigits_30_03.png)

### Superconductivity · 1% label (71 metode)

![Superconductivity · 1% label · metode 1–23 dari 71. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_01_01.png)

![Superconductivity · 1% label · metode 24–47 dari 71. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_01_02.png)

![Superconductivity · 1% label · metode 48–71 dari 71. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_01_03.png)

### Superconductivity · 10% label (71 metode)

![Superconductivity · 10% label · metode 1–23 dari 71. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_10_01.png)

![Superconductivity · 10% label · metode 24–47 dari 71. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_10_02.png)

![Superconductivity · 10% label · metode 48–71 dari 71. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_10_03.png)

### Superconductivity · 30% label (69 metode)

![Superconductivity · 30% label · metode 1–23 dari 69. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_30_01.png)

![Superconductivity · 30% label · metode 24–46 dari 69. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_30_02.png)

![Superconductivity · 30% label · metode 47–69 dari 69. Skor dan rank validation.](grafik_validation_3_seed/superconductivity_30_03.png)
