# Ringkasan augmentasi konteks

45 run selesai.

NHANES memakai QWK terkalibrasi untuk pemilihan checkpoint. Batas kelas hanya dari validation. QWK mentah, Spearman dan MSE tetap dicatat. Angka test hanya dilaporkan.

## nhanes (qwk_calibrated)

| Varian | Label % | Seed | Validation | Test |
|---|---|---|---|---|
| tabpfn | 10 | 1 | 0.3529 | 0.3305 |
| head_sup | 10 | 1 | 0.3660 | 0.3525 |
| ics_sup | 10 | 1 | 0.3634 | 0.3425 |
| head_distill | 10 | 1 | 0.3673 | 0.3349 |
| head_match | 10 | 1 | 0.3621 | 0.3502 |
| head_gaussian | 10 | 1 | 0.3621 | 0.3371 |
| head_dropout | 10 | 1 | 0.3647 | 0.3419 |
| head_fnoise | 10 | 1 | 0.3621 | 0.3446 |
| head_tabcutmix | 10 | 1 | 0.3686 | 0.3415 |
| head_angular | 10 | 1 | 0.3660 | 0.3477 |
| head_scarf | 10 | 1 | 0.3686 | 0.3429 |
| head_match_gaussian | 10 | 1 | 0.3608 | 0.3354 |
| ics_distill_70 | 10 | 1 | 0.3660 | 0.3398 |
| ics_match_70 | 10 | 1 | 0.3647 | 0.3420 |
| ics_both_70 | 10 | 1 | 0.3699 | 0.3349 |

| Pertanyaan | Varian | Pembanding | Label % | Menang validation | Δ rerata searah perbaikan |
|---|---|---|---|---|---|
| Distilasi SSL | head_distill | head_sup | 10 | 1/1 | +0.0013 |
| Self-training EMA | head_match | head_sup | 10 | 0/1 | -0.0039 |
| Distilasi vs EMA, view identik | head_distill | head_match | 10 | 1/1 | +0.0052 |
| ICS pada distilasi | ics_distill_70 | head_distill | 10 | 0/1 | -0.0013 |
| ICS pada EMA | ics_match_70 | head_match | 10 | 1/1 | +0.0026 |
| Gaussian pada EMA | head_match_gaussian | head_match | 10 | 0/1 | -0.0013 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 10 | 1/1 | +0.0013 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 10 | 1/1 | +0.0039 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 10 | 1/1 | +0.0013 |
| ICS pada supervised | ics_sup | head_sup | 10 | 0/1 | -0.0026 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 10 | 1/1 | +0.0065 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 10 | 1/1 | +0.0039 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 10 | 1/1 | +0.0039 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 10 | 0/1 | +0.0000 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 10 | 0/1 | -0.0026 |
| Efek head_gaussian | head_gaussian | head_distill | 10 | 0/1 | -0.0052 |
| Efek head_dropout | head_dropout | head_distill | 10 | 0/1 | -0.0026 |
| Efek head_fnoise | head_fnoise | head_distill | 10 | 0/1 | -0.0052 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 10 | 1/1 | +0.0013 |
| Efek head_angular | head_angular | head_distill | 10 | 0/1 | -0.0013 |
| Efek head_scarf | head_scarf | head_distill | 10 | 1/1 | +0.0013 |

## pendigits (macro_f1)

| Varian | Label % | Seed | Validation | Test |
|---|---|---|---|---|
| tabpfn | 1 | 1 | 0.9117 | 0.8728 |
| head_sup | 1 | 1 | 0.8846 | 0.8459 |
| ics_sup | 1 | 1 | 0.8798 | 0.8512 |
| head_distill | 1 | 1 | 0.9100 | 0.8824 |
| head_match | 1 | 1 | 0.8798 | 0.8522 |
| head_gaussian | 1 | 1 | 0.9106 | 0.8806 |
| head_dropout | 1 | 1 | 0.9082 | 0.8765 |
| head_fnoise | 1 | 1 | 0.9086 | 0.8800 |
| head_tabcutmix | 1 | 1 | 0.8933 | 0.8588 |
| head_angular | 1 | 1 | 0.9096 | 0.8712 |
| head_scarf | 1 | 1 | 0.9119 | 0.8800 |
| head_match_gaussian | 1 | 1 | 0.8841 | 0.8482 |
| ics_distill_70 | 1 | 1 | 0.9120 | 0.8700 |
| ics_match_70 | 1 | 1 | 0.8829 | 0.8483 |
| ics_both_70 | 1 | 1 | 0.9029 | 0.8637 |

| Pertanyaan | Varian | Pembanding | Label % | Menang validation | Δ rerata searah perbaikan |
|---|---|---|---|---|---|
| Distilasi SSL | head_distill | head_sup | 1 | 1/1 | +0.0254 |
| Self-training EMA | head_match | head_sup | 1 | 0/1 | -0.0047 |
| Distilasi vs EMA, view identik | head_distill | head_match | 1 | 1/1 | +0.0302 |
| ICS pada distilasi | ics_distill_70 | head_distill | 1 | 1/1 | +0.0020 |
| ICS pada EMA | ics_match_70 | head_match | 1 | 1/1 | +0.0031 |
| Gaussian pada EMA | head_match_gaussian | head_match | 1 | 1/1 | +0.0043 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 1 | 1/1 | +0.0265 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 1 | 0/1 | -0.0013 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 1 | 1/1 | +0.0291 |
| ICS pada supervised | ics_sup | head_sup | 1 | 0/1 | -0.0048 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 1 | 1/1 | +0.0232 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 1 | 0/1 | -0.0091 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 1 | 1/1 | +0.0014 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 1 | 1/1 | +0.0024 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 1 | 1/1 | +0.0001 |
| Efek head_gaussian | head_gaussian | head_distill | 1 | 1/1 | +0.0006 |
| Efek head_dropout | head_dropout | head_distill | 1 | 0/1 | -0.0018 |
| Efek head_fnoise | head_fnoise | head_distill | 1 | 0/1 | -0.0014 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 1 | 0/1 | -0.0167 |
| Efek head_angular | head_angular | head_distill | 1 | 0/1 | -0.0004 |
| Efek head_scarf | head_scarf | head_distill | 1 | 1/1 | +0.0019 |

## superconductivity (mse)

| Varian | Label % | Seed | Validation | Test |
|---|---|---|---|---|
| tabpfn | 30 | 1 | 105.6835 | 116.0043 |
| head_sup | 30 | 1 | 112.4272 | 121.0714 |
| ics_sup | 30 | 1 | 113.1262 | 123.0896 |
| head_distill | 30 | 1 | 110.1443 | 118.4148 |
| head_match | 30 | 1 | 166.7615 | 175.3748 |
| head_gaussian | 30 | 1 | 109.5625 | 117.2252 |
| head_dropout | 30 | 1 | 109.1549 | 117.4352 |
| head_fnoise | 30 | 1 | 109.0144 | 117.1045 |
| head_tabcutmix | 30 | 1 | 110.1854 | 117.6310 |
| head_angular | 30 | 1 | 143.1351 | 145.6767 |
| head_scarf | 30 | 1 | 150.9091 | 149.9104 |
| head_match_gaussian | 30 | 1 | 159.7190 | 167.9892 |
| ics_distill_70 | 30 | 1 | 109.5790 | 119.2970 |
| ics_match_70 | 30 | 1 | 137.4458 | 147.6337 |
| ics_both_70 | 30 | 1 | 109.9215 | 119.1185 |

| Pertanyaan | Varian | Pembanding | Label % | Menang validation | Δ rerata searah perbaikan |
|---|---|---|---|---|---|
| Distilasi SSL | head_distill | head_sup | 30 | 1/1 | +2.2829 |
| Self-training EMA | head_match | head_sup | 30 | 0/1 | -54.3343 |
| Distilasi vs EMA, view identik | head_distill | head_match | 30 | 1/1 | +56.6172 |
| ICS pada distilasi | ics_distill_70 | head_distill | 30 | 1/1 | +0.5653 |
| ICS pada EMA | ics_match_70 | head_match | 30 | 1/1 | +29.3157 |
| Gaussian pada EMA | head_match_gaussian | head_match | 30 | 1/1 | +7.0425 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 30 | 1/1 | +50.1564 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 30 | 1/1 | +22.2732 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 30 | 1/1 | +27.8668 |
| ICS pada supervised | ics_sup | head_sup | 30 | 0/1 | -0.6991 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 30 | 1/1 | +3.2047 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 30 | 0/1 | -0.3425 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 30 | 0/1 | -0.0165 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 30 | 1/1 | +33.5561 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 30 | 1/1 | +41.3302 |
| Efek head_gaussian | head_gaussian | head_distill | 30 | 1/1 | +0.5817 |
| Efek head_dropout | head_dropout | head_distill | 30 | 1/1 | +0.9893 |
| Efek head_fnoise | head_fnoise | head_distill | 30 | 1/1 | +1.1299 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 30 | 0/1 | -0.0411 |
| Efek head_angular | head_angular | head_distill | 30 | 0/1 | -32.9909 |
| Efek head_scarf | head_scarf | head_distill | 30 | 0/1 | -40.7649 |

## Diagnostik H1 pada ρ = 0,7

Rerata tiap seed diberi bobot sama. Kolom ICS dan angular memakai probe yang sama. Drift teacher berasal dari prediksi TabPFN penuh dan subset, sehingga dilaporkan sebagai pengukuran terpisah.

| Dataset | Label % | Seed | Cosine | Drift ICS | Drift angular | Drift teacher |
|---|---|---|---|---|---|---|
| nhanes | 10 | 1 | 0.98681 | 0.70915 | 4.31270 | 0.11034 |
| pendigits | 1 | 1 | 0.98077 | 0.07422 | 0.06445 | 0.08203 |
| superconductivity | 30 | 1 | 0.99444 | 5.26383 | 21.55211 | 1.41517 |

Data per seed dan bin cosine ada di `diagnostik_per_grup.csv` dan `diagnostik_cosine.csv`.

