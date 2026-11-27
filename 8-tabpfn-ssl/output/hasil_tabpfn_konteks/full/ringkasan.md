# Ringkasan augmentasi konteks

468 run selesai.

NHANES memakai QWK terkalibrasi untuk pemilihan checkpoint. Batas kelas hanya dari validation. QWK mentah, Spearman dan MSE tetap dicatat. Angka test hanya dilaporkan.

## Pilihan per dataset dan fraksi label

Setiap sel memerlukan grid utama lengkap pada ketiga seed. Metode dipilih dari rerata validation untuk masing-masing keluarga. Δ positif berarti membaik terhadap teacher TabPFN di validation.

| Dataset | Label % | Keluarga | Pilihan | Validation | Test | Δ validation vs teacher |
|---|---|---|---|---|---|---|
| nhanes | 1 | Supervised / label-only | tabpfn | 0.2645 | 0.2741 | +0.0000 |
| nhanes | 1 | SSL tanpa augmentasi eksplisit | head_distill | 0.2815 | 0.2656 | +0.0170 |
| nhanes | 1 | SSL dengan augmentasi | head_fnoise | 0.2884 | 0.2724 | +0.0240 |
| nhanes | 1 | SSL pilihan validation | head_fnoise | 0.2884 | 0.2724 | +0.0240 |
| pendigits | 1 | Supervised / label-only | tabpfn | 0.9092 | 0.8734 | +0.0000 |
| pendigits | 1 | SSL tanpa augmentasi eksplisit | head_distill | 0.9123 | 0.8695 | +0.0030 |
| pendigits | 1 | SSL dengan augmentasi | head_scarf | 0.9168 | 0.8691 | +0.0076 |
| pendigits | 1 | SSL pilihan validation | head_scarf | 0.9168 | 0.8691 | +0.0076 |
| superconductivity | 1 | Supervised / label-only | tabpfn | 288.0651 | 318.1292 | -0.0000 |
| superconductivity | 1 | SSL tanpa augmentasi eksplisit | head_distill | 280.8406 | 311.3573 | +7.2246 |
| superconductivity | 1 | SSL dengan augmentasi | head_gaussian | 281.1283 | 312.0869 | +6.9368 |
| superconductivity | 1 | SSL pilihan validation | head_distill | 280.8406 | 311.3573 | +7.2246 |
| nhanes | 10 | Supervised / label-only | ics_sup | 0.3712 | 0.3410 | +0.0192 |
| nhanes | 10 | SSL tanpa augmentasi eksplisit | head_distill | 0.3704 | 0.3381 | +0.0183 |
| nhanes | 10 | SSL dengan augmentasi | head_dropout | 0.3717 | 0.3420 | +0.0196 |
| nhanes | 10 | SSL pilihan validation | head_dropout | 0.3717 | 0.3420 | +0.0196 |
| pendigits | 10 | Supervised / label-only | ics_sup | 0.9887 | 0.9609 | +0.0001 |
| pendigits | 10 | SSL tanpa augmentasi eksplisit | head_distill | 0.9903 | 0.9647 | +0.0016 |
| pendigits | 10 | SSL dengan augmentasi | ics_distill_70 | 0.9912 | 0.9651 | +0.0025 |
| pendigits | 10 | SSL pilihan validation | ics_distill_70 | 0.9912 | 0.9651 | +0.0025 |
| nhanes | 30 | Supervised / label-only | ics_sup | 0.4035 | 0.4139 | +0.0109 |
| nhanes | 30 | SSL tanpa augmentasi eksplisit | head_match | 0.4022 | 0.4133 | +0.0096 |
| nhanes | 30 | SSL dengan augmentasi | ics_both_70 | 0.4087 | 0.4113 | +0.0161 |
| nhanes | 30 | SSL pilihan validation | ics_both_70 | 0.4087 | 0.4113 | +0.0161 |
| pendigits | 30 | Supervised / label-only | ics_sup | 0.9971 | 0.9797 | +0.0000 |
| pendigits | 30 | SSL tanpa augmentasi eksplisit | head_distill | 0.9978 | 0.9828 | +0.0007 |
| pendigits | 30 | SSL dengan augmentasi | head_gaussian | 0.9980 | 0.9842 | +0.0009 |
| pendigits | 30 | SSL pilihan validation | head_gaussian | 0.9980 | 0.9842 | +0.0009 |
| superconductivity | 30 | Supervised / label-only | tabpfn | 105.5832 | 116.4144 | -0.0000 |
| superconductivity | 30 | SSL tanpa augmentasi eksplisit | head_distill | 105.1412 | 116.9086 | +0.4420 |
| superconductivity | 30 | SSL dengan augmentasi | head_fnoise | 104.3256 | 116.5134 | +1.2576 |
| superconductivity | 30 | SSL pilihan validation | head_fnoise | 104.3256 | 116.5134 | +1.2576 |
| superconductivity | 10 | Supervised / label-only | tabpfn | 150.0316 | 163.3623 | -0.0000 |
| superconductivity | 10 | SSL tanpa augmentasi eksplisit | head_distill | 150.1126 | 162.2767 | -0.0810 |
| superconductivity | 10 | SSL dengan augmentasi | head_tabcutmix | 150.1818 | 163.7917 | -0.1502 |
| superconductivity | 10 | SSL pilihan validation | head_distill | 150.1126 | 162.2767 | -0.0810 |

## Pembanding historis

CSV memuat supervised dan RankUp tahap 5, RankUp dan SSR-RCUS tahap 6, serta teacher dan distill tahap 7 versi 2.5/3.5. Varian historis terpilih memakai rerata validation pada setiap sel. Arsitektur dan protokol checkpoint mengikuti tahap asal. Untuk NHANES, CSV memakai QWK mentah pada kedua sisi. Pemilihan tahap 8 tetap memakai QWK terkalibrasi.

| Dataset | Label % | Metrik test | SSL tahap 8 | SSL tahap 5 | RCUS tahap 6 | Augmentasi tahap 7 · 2.5 | Augmentasi tahap 7 · 3.5 |
|---|---|---|---|---|---|---|---|
| nhanes | 1 | qwk | head_fnoise (0.0558) | both_scarf (0.2800) | rcus_input_scarf (0.1624) | head_angular (0.2369) | head_angular (0.2609) |
| nhanes | 10 | qwk | head_dropout (0.2590) | angular (0.2616) | rcus (0.2178) | head_scarf (0.3062) | head_angular (0.3453) |
| nhanes | 30 | qwk | ics_both_70 (0.3063) | both_zero_out (0.3131) | rcus_input_scarf (0.2850) | head_angular (0.3715) | head_scarf (0.4128) |
| pendigits | 1 | macro_f1 | head_scarf (0.8691) | matched_gaussian (0.8927) | rcus_input_zero_out (0.8938) | head_scarf (0.8416) | head_scarf (0.8742) |
| pendigits | 10 | macro_f1 | ics_distill_70 (0.9651) | rankup_noaug (0.9492) | rcus (0.9726) | head_scarf (0.9442) | head_scarf (0.9680) |
| pendigits | 30 | macro_f1 | head_gaussian (0.9842) | latent_fnoise (0.9756) | rcus_angular_weak (0.9776) | head_dropout (0.9682) | head_scarf (0.9801) |
| superconductivity | 1 | mse | head_distill (311.3573) | matched_gaussian (364.5365) | rcus (446.2915) | head_fnoise (304.5947) | head_tabcutmix (314.4853) |
| superconductivity | 10 | mse | head_distill (162.2767) | angular_weak_prerelu (202.2932) | rcus_input_scarf (204.3363) | head_fnoise (168.0807) | head_gaussian (165.3933) |
| superconductivity | 30 | mse | head_fnoise (116.5134) | latent_gaussian (145.7297) | rcus (154.4634) | head_tabcutmix (123.6784) | head_tabcutmix (118.8988) |

## nhanes (qwk_calibrated)

| Varian | Label % | Seed | Validation | Test |
|---|---|---|---|---|
| tabpfn | 1 | 3 | 0.2645 ± 0.0096 | 0.2741 ± 0.0062 |
| tabpfn | 10 | 3 | 0.3521 ± 0.0027 | 0.3384 ± 0.0079 |
| tabpfn | 30 | 3 | 0.3926 ± 0.0015 | 0.4240 ± 0.0018 |
| head_sup | 1 | 3 | 0.2436 ± 0.0865 | 0.2313 ± 0.1131 |
| head_sup | 10 | 3 | 0.3690 ± 0.0027 | 0.3439 ± 0.0075 |
| head_sup | 30 | 3 | 0.4026 ± 0.0086 | 0.4193 ± 0.0133 |
| ics_sup | 1 | 3 | 0.2527 ± 0.0432 | 0.2552 ± 0.0461 |
| ics_sup | 10 | 3 | 0.3712 ± 0.0057 | 0.3410 ± 0.0040 |
| ics_sup | 30 | 3 | 0.4035 ± 0.0038 | 0.4139 ± 0.0111 |
| head_distill | 1 | 3 | 0.2815 ± 0.0020 | 0.2656 ± 0.0033 |
| head_distill | 10 | 3 | 0.3704 ± 0.0064 | 0.3381 ± 0.0032 |
| head_distill | 30 | 3 | 0.3987 ± 0.0026 | 0.4148 ± 0.0027 |
| head_match | 1 | 3 | 0.1817 ± 0.0443 | 0.1838 ± 0.0776 |
| head_match | 10 | 3 | 0.3643 ± 0.0053 | 0.3424 ± 0.0045 |
| head_match | 30 | 3 | 0.4022 ± 0.0096 | 0.4133 ± 0.0148 |
| head_gaussian | 1 | 3 | 0.2797 ± 0.0023 | 0.2683 ± 0.0133 |
| head_gaussian | 10 | 3 | 0.3708 ± 0.0072 | 0.3371 ± 0.0027 |
| head_gaussian | 30 | 3 | 0.4026 ± 0.0000 | 0.4178 ± 0.0111 |
| head_dropout | 1 | 3 | 0.2828 ± 0.0046 | 0.2712 ± 0.0084 |
| head_dropout | 10 | 3 | 0.3717 ± 0.0064 | 0.3420 ± 0.0085 |
| head_dropout | 30 | 3 | 0.4026 ± 0.0026 | 0.4186 ± 0.0106 |
| head_fnoise | 1 | 3 | 0.2884 ± 0.0053 | 0.2724 ± 0.0049 |
| head_fnoise | 10 | 3 | 0.3682 ± 0.0053 | 0.3425 ± 0.0069 |
| head_fnoise | 30 | 3 | 0.4000 ± 0.0026 | 0.4206 ± 0.0087 |
| head_tabcutmix | 1 | 3 | 0.2828 ± 0.0066 | 0.2731 ± 0.0053 |
| head_tabcutmix | 10 | 3 | 0.3695 ± 0.0015 | 0.3381 ± 0.0049 |
| head_tabcutmix | 30 | 3 | 0.4000 ± 0.0023 | 0.4172 ± 0.0127 |
| head_angular | 1 | 3 | 0.2832 ± 0.0075 | 0.2645 ± 0.0123 |
| head_angular | 10 | 3 | 0.3695 ± 0.0008 | 0.3429 ± 0.0048 |
| head_angular | 30 | 3 | 0.4039 ± 0.0035 | 0.4128 ± 0.0027 |
| head_scarf | 1 | 3 | 0.2841 ± 0.0080 | 0.2704 ± 0.0046 |
| head_scarf | 10 | 3 | 0.3704 ± 0.0027 | 0.3452 ± 0.0056 |
| head_scarf | 30 | 3 | 0.4017 ± 0.0015 | 0.4203 ± 0.0117 |
| head_match_gaussian | 1 | 3 | 0.2279 ± 0.0347 | 0.2174 ± 0.0307 |
| head_match_gaussian | 10 | 3 | 0.3669 ± 0.0060 | 0.3391 ± 0.0100 |
| head_match_gaussian | 30 | 3 | 0.4022 ± 0.0033 | 0.4167 ± 0.0132 |
| ics_distill_70 | 1 | 3 | 0.2209 ± 0.1181 | 0.2167 ± 0.1126 |
| ics_distill_70 | 10 | 3 | 0.3673 ± 0.0023 | 0.3417 ± 0.0020 |
| ics_distill_70 | 30 | 3 | 0.4043 ± 0.0079 | 0.4125 ± 0.0048 |
| ics_match_70 | 1 | 3 | 0.2388 ± 0.0821 | 0.2325 ± 0.1066 |
| ics_match_70 | 10 | 3 | 0.3699 ± 0.0035 | 0.3418 ± 0.0021 |
| ics_match_70 | 30 | 3 | 0.4030 ± 0.0053 | 0.4142 ± 0.0123 |
| ics_both_70 | 1 | 3 | 0.2536 ± 0.0403 | 0.2539 ± 0.0409 |
| ics_both_70 | 10 | 3 | 0.3677 ± 0.0040 | 0.3414 ± 0.0031 |
| ics_both_70 | 30 | 3 | 0.4087 ± 0.0040 | 0.4113 ± 0.0053 |
| ics_distill_50 | 1 | 3 | 0.2767 ± 0.0253 | 0.2511 ± 0.0359 |
| ics_distill_50 | 10 | 3 | 0.3660 ± 0.0035 | 0.3428 ± 0.0065 |
| ics_distill_90 | 1 | 3 | 0.2501 ± 0.0909 | 0.2556 ± 0.0826 |
| ics_distill_90 | 10 | 3 | 0.3695 ± 0.0020 | 0.3451 ± 0.0011 |
| ics_distill_ta | 1 | 3 | 0.2353 ± 0.1064 | 0.2379 ± 0.0936 |
| ics_distill_ta | 10 | 3 | 0.3677 ± 0.0008 | 0.3451 ± 0.0031 |
| ics_distill_ta | 30 | 3 | 0.4043 ± 0.0038 | 0.4143 ± 0.0138 |

| Pertanyaan | Varian | Pembanding | Label % | Menang validation | Δ rerata searah perbaikan |
|---|---|---|---|---|---|
| Distilasi SSL | head_distill | head_sup | 1 | 1/3 | +0.0379 |
| Distilasi SSL | head_distill | head_sup | 10 | 2/3 | +0.0013 |
| Distilasi SSL | head_distill | head_sup | 30 | 1/3 | -0.0039 |
| Self-training EMA | head_match | head_sup | 1 | 1/3 | -0.0619 |
| Self-training EMA | head_match | head_sup | 10 | 0/3 | -0.0048 |
| Self-training EMA | head_match | head_sup | 30 | 2/3 | -0.0004 |
| Distilasi vs EMA, view identik | head_distill | head_match | 1 | 3/3 | +0.0998 |
| Distilasi vs EMA, view identik | head_distill | head_match | 10 | 3/3 | +0.0061 |
| Distilasi vs EMA, view identik | head_distill | head_match | 30 | 1/3 | -0.0035 |
| ICS pada distilasi | ics_distill_70 | head_distill | 1 | 1/3 | -0.0606 |
| ICS pada distilasi | ics_distill_70 | head_distill | 10 | 1/3 | -0.0031 |
| ICS pada distilasi | ics_distill_70 | head_distill | 30 | 2/3 | +0.0057 |
| ICS pada EMA | ics_match_70 | head_match | 1 | 2/3 | +0.0571 |
| ICS pada EMA | ics_match_70 | head_match | 10 | 3/3 | +0.0057 |
| ICS pada EMA | ics_match_70 | head_match | 30 | 2/3 | +0.0009 |
| Gaussian pada EMA | head_match_gaussian | head_match | 1 | 2/3 | +0.0462 |
| Gaussian pada EMA | head_match_gaussian | head_match | 10 | 2/3 | +0.0026 |
| Gaussian pada EMA | head_match_gaussian | head_match | 30 | 2/3 | +0.0000 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 1 | 3/3 | +0.0519 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 10 | 3/3 | +0.0039 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 30 | 1/3 | +0.0004 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 1 | 2/3 | +0.0109 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 10 | 2/3 | +0.0031 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 30 | 2/3 | +0.0009 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 1 | 1/3 | -0.0179 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 10 | 1/3 | -0.0026 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 30 | 2/3 | +0.0013 |
| ICS pada supervised | ics_sup | head_sup | 1 | 2/3 | +0.0092 |
| ICS pada supervised | ics_sup | head_sup | 10 | 2/3 | +0.0022 |
| ICS pada supervised | ics_sup | head_sup | 30 | 1/3 | +0.0009 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 1 | 1/3 | +0.0009 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 10 | 0/3 | -0.0035 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 30 | 2/3 | +0.0052 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 1 | 2/3 | +0.0327 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 10 | 2/3 | +0.0004 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 30 | 2/3 | +0.0044 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 1 | 1/3 | -0.0588 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 10 | 1/3 | -0.0035 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 30 | 2/3 | +0.0017 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 1 | 1/3 | -0.0623 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 10 | 0/3 | -0.0022 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 30 | 1/3 | +0.0004 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 1 | 2/3 | -0.0632 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 10 | 0/3 | -0.0031 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 30 | 2/3 | +0.0026 |
| Efek head_gaussian | head_gaussian | head_distill | 1 | 1/3 | -0.0017 |
| Efek head_gaussian | head_gaussian | head_distill | 10 | 2/3 | +0.0004 |
| Efek head_gaussian | head_gaussian | head_distill | 30 | 3/3 | +0.0039 |
| Efek head_dropout | head_dropout | head_distill | 1 | 1/3 | +0.0013 |
| Efek head_dropout | head_dropout | head_distill | 10 | 2/3 | +0.0013 |
| Efek head_dropout | head_dropout | head_distill | 30 | 2/3 | +0.0039 |
| Efek head_fnoise | head_fnoise | head_distill | 1 | 3/3 | +0.0070 |
| Efek head_fnoise | head_fnoise | head_distill | 10 | 1/3 | -0.0022 |
| Efek head_fnoise | head_fnoise | head_distill | 30 | 2/3 | +0.0013 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 1 | 2/3 | +0.0013 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 10 | 2/3 | -0.0009 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 30 | 2/3 | +0.0013 |
| Efek head_angular | head_angular | head_distill | 1 | 2/3 | +0.0017 |
| Efek head_angular | head_angular | head_distill | 10 | 2/3 | -0.0009 |
| Efek head_angular | head_angular | head_distill | 30 | 3/3 | +0.0052 |
| Efek head_scarf | head_scarf | head_distill | 1 | 2/3 | +0.0026 |
| Efek head_scarf | head_scarf | head_distill | 10 | 1/3 | +0.0000 |
| Efek head_scarf | head_scarf | head_distill | 30 | 2/3 | +0.0031 |
| Efek ics_distill_90 | ics_distill_90 | head_distill | 1 | 2/3 | -0.0314 |
| Efek ics_distill_90 | ics_distill_90 | head_distill | 10 | 2/3 | -0.0009 |
| Efek ics_distill_50 | ics_distill_50 | head_distill | 1 | 1/3 | -0.0048 |
| Efek ics_distill_50 | ics_distill_50 | head_distill | 10 | 0/3 | -0.0044 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 1 | 2/3 | -0.0462 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 10 | 1/3 | -0.0026 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 30 | 3/3 | +0.0057 |

## pendigits (macro_f1)

| Varian | Label % | Seed | Validation | Test |
|---|---|---|---|---|
| tabpfn | 1 | 3 | 0.9092 ± 0.0023 | 0.8734 ± 0.0023 |
| tabpfn | 10 | 3 | 0.9887 ± 0.0000 | 0.9703 ± 0.0019 |
| tabpfn | 30 | 3 | 0.9971 ± 0.0004 | 0.9841 ± 0.0006 |
| head_sup | 1 | 3 | 0.8833 ± 0.0029 | 0.8406 ± 0.0142 |
| head_sup | 10 | 3 | 0.9881 ± 0.0011 | 0.9564 ± 0.0023 |
| head_sup | 30 | 3 | 0.9969 ± 0.0008 | 0.9817 ± 0.0031 |
| ics_sup | 1 | 3 | 0.8810 ± 0.0141 | 0.8347 ± 0.0205 |
| ics_sup | 10 | 3 | 0.9887 ± 0.0017 | 0.9609 ± 0.0036 |
| ics_sup | 30 | 3 | 0.9971 ± 0.0004 | 0.9797 ± 0.0041 |
| head_distill | 1 | 3 | 0.9123 ± 0.0010 | 0.8695 ± 0.0007 |
| head_distill | 10 | 3 | 0.9903 ± 0.0010 | 0.9647 ± 0.0020 |
| head_distill | 30 | 3 | 0.9978 ± 0.0004 | 0.9828 ± 0.0016 |
| head_match | 1 | 3 | 0.9017 ± 0.0161 | 0.8532 ± 0.0250 |
| head_match | 10 | 3 | 0.9886 ± 0.0010 | 0.9595 ± 0.0056 |
| head_match | 30 | 3 | 0.9971 ± 0.0015 | 0.9810 ± 0.0039 |
| head_gaussian | 1 | 3 | 0.9123 ± 0.0018 | 0.8697 ± 0.0015 |
| head_gaussian | 10 | 3 | 0.9903 ± 0.0004 | 0.9630 ± 0.0012 |
| head_gaussian | 30 | 3 | 0.9980 ± 0.0000 | 0.9842 ± 0.0019 |
| head_dropout | 1 | 3 | 0.9137 ± 0.0017 | 0.8670 ± 0.0029 |
| head_dropout | 10 | 3 | 0.9901 ± 0.0007 | 0.9661 ± 0.0031 |
| head_dropout | 30 | 3 | 0.9980 ± 0.0000 | 0.9823 ± 0.0014 |
| head_fnoise | 1 | 3 | 0.9131 ± 0.0019 | 0.8680 ± 0.0026 |
| head_fnoise | 10 | 3 | 0.9901 ± 0.0007 | 0.9652 ± 0.0029 |
| head_fnoise | 30 | 3 | 0.9980 ± 0.0000 | 0.9833 ± 0.0024 |
| head_tabcutmix | 1 | 3 | 0.8994 ± 0.0083 | 0.8422 ± 0.0172 |
| head_tabcutmix | 10 | 3 | 0.9886 ± 0.0017 | 0.9588 ± 0.0074 |
| head_tabcutmix | 30 | 3 | 0.9964 ± 0.0010 | 0.9808 ± 0.0019 |
| head_angular | 1 | 3 | 0.9134 ± 0.0019 | 0.8688 ± 0.0060 |
| head_angular | 10 | 3 | 0.9897 ± 0.0010 | 0.9683 ± 0.0011 |
| head_angular | 30 | 3 | 0.9978 ± 0.0004 | 0.9827 ± 0.0037 |
| head_scarf | 1 | 3 | 0.9168 ± 0.0020 | 0.8691 ± 0.0068 |
| head_scarf | 10 | 3 | 0.9905 ± 0.0014 | 0.9665 ± 0.0036 |
| head_scarf | 30 | 3 | 0.9977 ± 0.0004 | 0.9828 ± 0.0014 |
| head_match_gaussian | 1 | 3 | 0.8982 ± 0.0134 | 0.8553 ± 0.0306 |
| head_match_gaussian | 10 | 3 | 0.9885 ± 0.0010 | 0.9600 ± 0.0034 |
| head_match_gaussian | 30 | 3 | 0.9969 ± 0.0008 | 0.9816 ± 0.0023 |
| ics_distill_70 | 1 | 3 | 0.9158 ± 0.0037 | 0.8742 ± 0.0079 |
| ics_distill_70 | 10 | 3 | 0.9912 ± 0.0010 | 0.9651 ± 0.0056 |
| ics_distill_70 | 30 | 3 | 0.9977 ± 0.0004 | 0.9818 ± 0.0020 |
| ics_match_70 | 1 | 3 | 0.8927 ± 0.0073 | 0.8432 ± 0.0155 |
| ics_match_70 | 10 | 3 | 0.9896 ± 0.0010 | 0.9621 ± 0.0040 |
| ics_match_70 | 30 | 3 | 0.9971 ± 0.0010 | 0.9809 ± 0.0031 |
| ics_both_70 | 1 | 3 | 0.9147 ± 0.0027 | 0.8726 ± 0.0085 |
| ics_both_70 | 10 | 3 | 0.9911 ± 0.0008 | 0.9658 ± 0.0043 |
| ics_both_70 | 30 | 3 | 0.9978 ± 0.0004 | 0.9813 ± 0.0005 |
| ics_distill_50 | 1 | 3 | 0.9127 ± 0.0074 | 0.8691 ± 0.0160 |
| ics_distill_50 | 10 | 3 | 0.9905 ± 0.0013 | 0.9676 ± 0.0014 |
| ics_distill_90 | 1 | 3 | 0.9163 ± 0.0040 | 0.8685 ± 0.0073 |
| ics_distill_90 | 10 | 3 | 0.9903 ± 0.0004 | 0.9638 ± 0.0033 |
| ics_distill_ta | 1 | 3 | 0.9101 ± 0.0043 | 0.8688 ± 0.0046 |
| ics_distill_ta | 10 | 3 | 0.9903 ± 0.0007 | 0.9673 ± 0.0017 |
| ics_distill_ta | 30 | 3 | 0.9980 ± 0.0000 | 0.9825 ± 0.0016 |

| Pertanyaan | Varian | Pembanding | Label % | Menang validation | Δ rerata searah perbaikan |
|---|---|---|---|---|---|
| Distilasi SSL | head_distill | head_sup | 1 | 3/3 | +0.0289 |
| Distilasi SSL | head_distill | head_sup | 10 | 3/3 | +0.0022 |
| Distilasi SSL | head_distill | head_sup | 30 | 3/3 | +0.0009 |
| Self-training EMA | head_match | head_sup | 1 | 3/3 | +0.0183 |
| Self-training EMA | head_match | head_sup | 10 | 2/3 | +0.0004 |
| Self-training EMA | head_match | head_sup | 30 | 2/3 | +0.0002 |
| Distilasi vs EMA, view identik | head_distill | head_match | 1 | 2/3 | +0.0106 |
| Distilasi vs EMA, view identik | head_distill | head_match | 10 | 3/3 | +0.0017 |
| Distilasi vs EMA, view identik | head_distill | head_match | 30 | 2/3 | +0.0007 |
| ICS pada distilasi | ics_distill_70 | head_distill | 1 | 3/3 | +0.0035 |
| ICS pada distilasi | ics_distill_70 | head_distill | 10 | 3/3 | +0.0009 |
| ICS pada distilasi | ics_distill_70 | head_distill | 30 | 1/3 | -0.0000 |
| ICS pada EMA | ics_match_70 | head_match | 1 | 0/3 | -0.0089 |
| ICS pada EMA | ics_match_70 | head_match | 10 | 3/3 | +0.0011 |
| ICS pada EMA | ics_match_70 | head_match | 30 | 1/3 | -0.0000 |
| Gaussian pada EMA | head_match_gaussian | head_match | 1 | 1/3 | -0.0035 |
| Gaussian pada EMA | head_match_gaussian | head_match | 10 | 1/3 | -0.0000 |
| Gaussian pada EMA | head_match_gaussian | head_match | 30 | 1/3 | -0.0002 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 1 | 2/3 | +0.0142 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 10 | 3/3 | +0.0018 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 30 | 3/3 | +0.0011 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 1 | 1/3 | -0.0054 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 10 | 3/3 | +0.0011 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 30 | 1/3 | +0.0002 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 1 | 3/3 | +0.0230 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 10 | 3/3 | +0.0016 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 30 | 3/3 | +0.0007 |
| ICS pada supervised | ics_sup | head_sup | 1 | 1/3 | -0.0024 |
| ICS pada supervised | ics_sup | head_sup | 10 | 2/3 | +0.0006 |
| ICS pada supervised | ics_sup | head_sup | 30 | 2/3 | +0.0002 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 1 | 3/3 | +0.0337 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 10 | 3/3 | +0.0024 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 30 | 3/3 | +0.0007 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 1 | 1/3 | -0.0011 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 10 | 1/3 | -0.0000 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 30 | 1/3 | +0.0000 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 1 | 2/3 | +0.0034 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 10 | 3/3 | +0.0009 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 30 | 0/3 | -0.0002 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 1 | 2/3 | +0.0024 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 10 | 3/3 | +0.0015 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 30 | 1/3 | -0.0000 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 1 | 1/3 | -0.0011 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 10 | 2/3 | +0.0007 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 30 | 1/3 | +0.0000 |
| Efek head_gaussian | head_gaussian | head_distill | 1 | 1/3 | +0.0000 |
| Efek head_gaussian | head_gaussian | head_distill | 10 | 1/3 | +0.0000 |
| Efek head_gaussian | head_gaussian | head_distill | 30 | 1/3 | +0.0002 |
| Efek head_dropout | head_dropout | head_distill | 1 | 3/3 | +0.0014 |
| Efek head_dropout | head_dropout | head_distill | 10 | 1/3 | -0.0002 |
| Efek head_dropout | head_dropout | head_distill | 30 | 1/3 | +0.0002 |
| Efek head_fnoise | head_fnoise | head_distill | 1 | 2/3 | +0.0009 |
| Efek head_fnoise | head_fnoise | head_distill | 10 | 0/3 | -0.0002 |
| Efek head_fnoise | head_fnoise | head_distill | 30 | 1/3 | +0.0002 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 1 | 0/3 | -0.0129 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 10 | 0/3 | -0.0017 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 30 | 0/3 | -0.0013 |
| Efek head_angular | head_angular | head_distill | 1 | 2/3 | +0.0011 |
| Efek head_angular | head_angular | head_distill | 10 | 0/3 | -0.0006 |
| Efek head_angular | head_angular | head_distill | 30 | 1/3 | +0.0000 |
| Efek head_scarf | head_scarf | head_distill | 1 | 3/3 | +0.0046 |
| Efek head_scarf | head_scarf | head_distill | 10 | 1/3 | +0.0002 |
| Efek head_scarf | head_scarf | head_distill | 30 | 1/3 | -0.0000 |
| Efek ics_distill_90 | ics_distill_90 | head_distill | 1 | 3/3 | +0.0040 |
| Efek ics_distill_90 | ics_distill_90 | head_distill | 10 | 1/3 | +0.0000 |
| Efek ics_distill_50 | ics_distill_50 | head_distill | 1 | 2/3 | +0.0004 |
| Efek ics_distill_50 | ics_distill_50 | head_distill | 10 | 2/3 | +0.0002 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 1 | 1/3 | -0.0022 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 10 | 1/3 | -0.0000 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 30 | 1/3 | +0.0002 |

## superconductivity (mse)

| Varian | Label % | Seed | Validation | Test |
|---|---|---|---|---|
| tabpfn | 1 | 3 | 288.0651 ± 1.1058 | 318.1292 ± 0.1611 |
| tabpfn | 10 | 3 | 150.0316 ± 0.0987 | 163.3623 ± 0.3290 |
| tabpfn | 30 | 3 | 105.5832 ± 1.1296 | 116.4144 ± 0.5328 |
| head_sup | 1 | 3 | 306.0565 ± 8.3560 | 331.2539 ± 3.9768 |
| head_sup | 10 | 3 | 152.0610 ± 4.1844 | 164.4635 ± 1.8289 |
| head_sup | 30 | 3 | 106.8987 ± 3.2173 | 118.5764 ± 2.1812 |
| ics_sup | 1 | 3 | 304.9726 ± 8.7633 | 329.7636 ± 5.1565 |
| ics_sup | 10 | 3 | 151.9409 ± 4.6648 | 164.6087 ± 1.9970 |
| ics_sup | 30 | 3 | 106.5592 ± 1.9035 | 118.8415 ± 2.4786 |
| head_distill | 1 | 3 | 280.8406 ± 4.5631 | 311.3573 ± 6.0406 |
| head_distill | 10 | 3 | 150.1126 ± 1.4080 | 162.2767 ± 1.4229 |
| head_distill | 30 | 3 | 105.1412 ± 2.5708 | 116.9086 ± 1.8087 |
| head_match | 1 | 3 | 469.3276 ± 39.7012 | 468.7172 ± 33.6656 |
| head_match | 10 | 3 | 159.1521 ± 3.2718 | 171.5496 ± 3.7715 |
| head_match | 30 | 3 | 107.3498 ± 1.8869 | 118.0328 ± 2.9471 |
| head_gaussian | 1 | 3 | 281.1283 ± 4.8622 | 312.0869 ± 7.3052 |
| head_gaussian | 10 | 3 | 150.4875 ± 1.8111 | 163.6793 ± 1.7237 |
| head_gaussian | 30 | 3 | 104.9710 ± 1.0602 | 117.5573 ± 1.0685 |
| head_dropout | 1 | 3 | 282.6103 ± 4.4304 | 312.7677 ± 4.4761 |
| head_dropout | 10 | 3 | 150.5733 ± 1.1181 | 163.0836 ± 1.7624 |
| head_dropout | 30 | 3 | 105.4317 ± 1.1516 | 117.6384 ± 1.1076 |
| head_fnoise | 1 | 3 | 283.0376 ± 2.8303 | 313.3940 ± 3.4328 |
| head_fnoise | 10 | 3 | 150.3384 ± 1.6784 | 163.8111 ± 1.9158 |
| head_fnoise | 30 | 3 | 104.3256 ± 1.6343 | 116.5134 ± 1.3690 |
| head_tabcutmix | 1 | 3 | 285.2307 ± 1.5343 | 316.6153 ± 2.6758 |
| head_tabcutmix | 10 | 3 | 150.1818 ± 2.0687 | 163.7917 ± 1.2656 |
| head_tabcutmix | 30 | 3 | 105.3221 ± 1.7395 | 117.5273 ± 2.4226 |
| head_angular | 1 | 3 | 402.7402 ± 44.0311 | 433.8507 ± 45.8722 |
| head_angular | 10 | 3 | 174.1750 ± 8.0822 | 183.2419 ± 8.4509 |
| head_angular | 30 | 3 | 137.5394 ± 8.8201 | 147.0860 ± 6.2991 |
| head_scarf | 1 | 3 | 385.4686 ± 10.9438 | 413.4432 ± 9.2002 |
| head_scarf | 10 | 3 | 169.5437 ± 6.4000 | 181.8964 ± 6.7021 |
| head_scarf | 30 | 3 | 131.0725 ± 10.6334 | 133.1031 ± 7.2500 |
| head_match_gaussian | 1 | 3 | 475.9159 ± 29.1422 | 472.8484 ± 18.6464 |
| head_match_gaussian | 10 | 3 | 161.2718 ± 3.9404 | 173.2326 ± 3.8337 |
| head_match_gaussian | 30 | 3 | 108.5335 ± 2.4366 | 119.1730 ± 2.9330 |
| ics_distill_70 | 1 | 3 | 297.1171 ± 5.3111 | 320.3814 ± 6.0851 |
| ics_distill_70 | 10 | 3 | 150.4706 ± 0.6782 | 162.3353 ± 0.3142 |
| ics_distill_70 | 30 | 3 | 105.2398 ± 2.3884 | 117.9199 ± 1.7380 |
| ics_match_70 | 1 | 3 | 324.3324 ± 13.8247 | 347.6106 ± 2.1307 |
| ics_match_70 | 10 | 3 | 153.4885 ± 0.7894 | 165.2935 ± 0.2757 |
| ics_match_70 | 30 | 3 | 106.1311 ± 1.5269 | 118.4029 ± 1.4876 |
| ics_both_70 | 1 | 3 | 296.8093 ± 4.5006 | 319.6708 ± 2.1136 |
| ics_both_70 | 10 | 3 | 150.2070 ± 0.2978 | 162.1034 ± 1.4876 |
| ics_both_70 | 30 | 3 | 105.6067 ± 1.4582 | 118.4763 ± 2.0004 |
| ics_distill_50 | 1 | 3 | 296.6445 ± 9.5785 | 321.3448 ± 5.5597 |
| ics_distill_50 | 10 | 3 | 150.2025 ± 0.4847 | 163.4002 ± 0.9208 |
| ics_distill_90 | 1 | 3 | 292.3490 ± 5.9495 | 314.9504 ± 2.2989 |
| ics_distill_90 | 10 | 3 | 149.7812 ± 1.5716 | 161.9180 ± 1.0851 |
| ics_distill_ta | 1 | 3 | 296.7448 ± 0.5573 | 320.8082 ± 4.8084 |
| ics_distill_ta | 10 | 3 | 149.7143 ± 1.3924 | 163.0698 ± 0.9351 |
| ics_distill_ta | 30 | 3 | 105.2907 ± 1.9826 | 119.0075 ± 1.7983 |

| Pertanyaan | Varian | Pembanding | Label % | Menang validation | Δ rerata searah perbaikan |
|---|---|---|---|---|---|
| Distilasi SSL | head_distill | head_sup | 1 | 3/3 | +25.2160 |
| Distilasi SSL | head_distill | head_sup | 10 | 2/3 | +1.9484 |
| Distilasi SSL | head_distill | head_sup | 30 | 2/3 | +1.7575 |
| Self-training EMA | head_match | head_sup | 1 | 0/3 | -163.2711 |
| Self-training EMA | head_match | head_sup | 10 | 0/3 | -7.0911 |
| Self-training EMA | head_match | head_sup | 30 | 1/3 | -0.4511 |
| Distilasi vs EMA, view identik | head_distill | head_match | 1 | 3/3 | +188.4870 |
| Distilasi vs EMA, view identik | head_distill | head_match | 10 | 3/3 | +9.0395 |
| Distilasi vs EMA, view identik | head_distill | head_match | 30 | 3/3 | +2.2085 |
| ICS pada distilasi | ics_distill_70 | head_distill | 1 | 0/3 | -16.2766 |
| ICS pada distilasi | ics_distill_70 | head_distill | 10 | 1/3 | -0.3580 |
| ICS pada distilasi | ics_distill_70 | head_distill | 30 | 1/3 | -0.0986 |
| ICS pada EMA | ics_match_70 | head_match | 1 | 3/3 | +144.9952 |
| ICS pada EMA | ics_match_70 | head_match | 10 | 3/3 | +5.6636 |
| ICS pada EMA | ics_match_70 | head_match | 30 | 3/3 | +1.2186 |
| Gaussian pada EMA | head_match_gaussian | head_match | 1 | 1/3 | -6.5883 |
| Gaussian pada EMA | head_match_gaussian | head_match | 10 | 0/3 | -2.1197 |
| Gaussian pada EMA | head_match_gaussian | head_match | 30 | 0/3 | -1.1837 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 1 | 3/3 | +194.7875 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 10 | 3/3 | +10.7843 |
| Distilasi vs EMA, keduanya Gaussian | head_gaussian | head_match_gaussian | 30 | 3/3 | +3.5625 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 1 | 3/3 | +151.5835 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 10 | 3/3 | +7.7833 |
| ICS vs Gaussian pada EMA | ics_match_70 | head_match_gaussian | 30 | 3/3 | +2.4024 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 1 | 3/3 | +27.2152 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 10 | 3/3 | +3.0179 |
| Distilasi vs EMA, keduanya ICS | ics_distill_70 | ics_match_70 | 30 | 2/3 | +0.8913 |
| ICS pada supervised | ics_sup | head_sup | 1 | 2/3 | +1.0839 |
| ICS pada supervised | ics_sup | head_sup | 10 | 1/3 | +0.1200 |
| ICS pada supervised | ics_sup | head_sup | 30 | 1/3 | +0.3395 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 1 | 2/3 | +8.1633 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 10 | 1/3 | +1.7340 |
| Tambah SSL pada ICS supervised | ics_both_70 | ics_sup | 30 | 2/3 | +0.9525 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 1 | 1/3 | +0.3078 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 10 | 2/3 | +0.2636 |
| Tambah ICS labeled pada ICS-distill | ics_both_70 | ics_distill_70 | 30 | 1/3 | -0.3669 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 1 | 0/3 | -15.9888 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 10 | 1/3 | +0.0169 |
| ICS vs Gaussian pada distilasi | ics_distill_70 | head_gaussian | 30 | 1/3 | -0.2688 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 1 | 3/3 | +105.6230 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 10 | 3/3 | +23.7044 |
| ICS vs angular pada distilasi | ics_distill_70 | head_angular | 30 | 3/3 | +32.2995 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 1 | 3/3 | +88.3515 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 10 | 3/3 | +19.0730 |
| ICS vs SCARF pada distilasi | ics_distill_70 | head_scarf | 30 | 3/3 | +25.8327 |
| Efek head_gaussian | head_gaussian | head_distill | 1 | 1/3 | -0.2878 |
| Efek head_gaussian | head_gaussian | head_distill | 10 | 1/3 | -0.3749 |
| Efek head_gaussian | head_gaussian | head_distill | 30 | 2/3 | +0.1702 |
| Efek head_dropout | head_dropout | head_distill | 1 | 0/3 | -1.7697 |
| Efek head_dropout | head_dropout | head_distill | 10 | 1/3 | -0.4607 |
| Efek head_dropout | head_dropout | head_distill | 30 | 2/3 | -0.2905 |
| Efek head_fnoise | head_fnoise | head_distill | 1 | 1/3 | -2.1970 |
| Efek head_fnoise | head_fnoise | head_distill | 10 | 1/3 | -0.2258 |
| Efek head_fnoise | head_fnoise | head_distill | 30 | 2/3 | +0.8157 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 1 | 0/3 | -4.3902 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 10 | 2/3 | -0.0691 |
| Efek head_tabcutmix | head_tabcutmix | head_distill | 30 | 2/3 | -0.1808 |
| Efek head_angular | head_angular | head_distill | 1 | 0/3 | -121.8996 |
| Efek head_angular | head_angular | head_distill | 10 | 0/3 | -24.0624 |
| Efek head_angular | head_angular | head_distill | 30 | 0/3 | -32.3982 |
| Efek head_scarf | head_scarf | head_distill | 1 | 0/3 | -104.6280 |
| Efek head_scarf | head_scarf | head_distill | 10 | 0/3 | -19.4311 |
| Efek head_scarf | head_scarf | head_distill | 30 | 0/3 | -25.9313 |
| Efek ics_distill_90 | ics_distill_90 | head_distill | 1 | 0/3 | -11.5084 |
| Efek ics_distill_90 | ics_distill_90 | head_distill | 10 | 2/3 | +0.3315 |
| Efek ics_distill_50 | ics_distill_50 | head_distill | 1 | 0/3 | -15.8039 |
| Efek ics_distill_50 | ics_distill_50 | head_distill | 10 | 2/3 | -0.0899 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 1 | 0/3 | -15.9042 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 10 | 3/3 | +0.3983 |
| Efek ics_distill_ta | ics_distill_ta | head_distill | 30 | 2/3 | -0.1494 |

## Diagnostik H1 pada ρ = 0,7

Rerata tiap seed diberi bobot sama. Kolom ICS dan angular memakai probe yang sama. Drift teacher berasal dari prediksi TabPFN penuh dan subset, sehingga dilaporkan sebagai pengukuran terpisah.

| Dataset | Label % | Seed | Cosine | Drift ICS | Drift angular | Drift teacher |
|---|---|---|---|---|---|---|
| nhanes | 1 | 3 | 0.79120 | 1.13382 | 6.01569 | 0.20255 |
| nhanes | 10 | 3 | 0.98602 | 1.14978 | 3.66684 | 0.10290 |
| nhanes | 30 | 3 | 0.99690 | 0.74723 | 2.63859 | 0.08252 |
| pendigits | 1 | 3 | 0.97787 | 0.09747 | 0.04563 | 0.10100 |
| pendigits | 10 | 3 | 0.99321 | 0.00722 | 0.00155 | 0.00793 |
| pendigits | 30 | 3 | 0.99638 | 0.00296 | 0.00089 | 0.00318 |
| superconductivity | 1 | 3 | 0.99006 | 13.75331 | 23.21681 | 3.56331 |
| superconductivity | 10 | 3 | 0.99408 | 11.46727 | 42.56032 | 2.08170 |
| superconductivity | 30 | 3 | 0.99418 | 4.97572 | 20.56678 | 1.60973 |

Data per seed dan bin cosine ada di `diagnostik_per_grup.csv` dan `diagnostik_cosine.csv`.

