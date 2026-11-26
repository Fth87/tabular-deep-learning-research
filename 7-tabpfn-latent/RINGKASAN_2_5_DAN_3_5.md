# Hasil TabPFN 2.5 dan 3.5

Masing-masing versi punya 270 run final yang berpasangan di 3 dataset × 3 fraksi label × 3 seed × 10 varian. Ringkasan memakai `results_gpu0.jsonl` dan `results_gpu1.jsonl`. Run smoke dikeluarkan. Log 3.5 mencatat `tabpfn==10.2.0` dan `tabpfn-v3.5-20260909.safetensors`. Versi 2.5 mengikuti nama folder karena log lama belum mencatat versi paket atau bobot.

## 1. Skor test dan pembanding tahap 5–6

Rata-rata ± simpangan baku antar tiga seed. QWK dan macro-F1 lebih besar lebih baik, MSE lebih kecil lebih baik. Pembanding TabM yang ditampilkan adalah metode yang sudah ditetapkan, dengan split yang sama dan protokol training masing-masing. Tahap 6 memilih checkpoint regresi/ordinal dengan MAE, sehingga perbedaan angka lintas tahap termasuk efek protokol. QWK tahap 5–7 di bagian ini memakai pembulatan seperti log asli.

### nhanes (qwk)

| Versi dan metode | 1% | 10% | 30% |
|---|---|---|---|
| 2.5 · tabpfn | 0.089 ± 0.050 | 0.216 ± 0.009 | 0.320 ± 0.010 |
| 3.5 · tabpfn | 0.084 ± 0.031 | 0.260 ± 0.006 | 0.319 ± 0.003 |
| 2.5 · head_sup | 0.104 ± 0.117 | 0.247 ± 0.075 | 0.320 ± 0.017 |
| 3.5 · head_sup | 0.117 ± 0.110 | 0.302 ± 0.016 | 0.331 ± 0.052 |
| 2.5 · head_distill | 0.143 ± 0.077 | 0.255 ± 0.025 | 0.328 ± 0.013 |
| 3.5 · head_distill | 0.106 ± 0.037 | 0.265 ± 0.017 | 0.321 ± 0.013 |
| 2.5 · head_gaussian | 0.108 ± 0.085 | 0.263 ± 0.003 | 0.342 ± 0.007 |
| 3.5 · head_gaussian | 0.082 ± 0.021 | 0.282 ± 0.025 | 0.326 ± 0.006 |
| 2.5 · head_dropout | 0.094 ± 0.081 | 0.266 ± 0.008 | 0.344 ± 0.012 |
| 3.5 · head_dropout | 0.059 ± 0.024 | 0.271 ± 0.015 | 0.336 ± 0.006 |
| 2.5 · head_fnoise | 0.117 ± 0.085 | 0.258 ± 0.012 | 0.339 ± 0.003 |
| 3.5 · head_fnoise | 0.117 ± 0.017 | 0.291 ± 0.026 | 0.328 ± 0.016 |
| 2.5 · head_angular | 0.237 ± 0.016 | 0.298 ± 0.016 | 0.371 ± 0.002 |
| 3.5 · head_angular | 0.261 ± 0.002 | 0.345 ± 0.010 | 0.401 ± 0.004 |
| 2.5 · head_tabcutmix | 0.151 ± 0.062 | 0.268 ± 0.014 | 0.345 ± 0.014 |
| 3.5 · head_tabcutmix | 0.105 ± 0.038 | 0.260 ± 0.014 | 0.338 ± 0.005 |
| 2.5 · head_zero_out | 0.233 ± 0.015 | 0.318 ± 0.016 | 0.361 ± 0.011 |
| 3.5 · head_zero_out | 0.243 ± 0.009 | 0.343 ± 0.008 | 0.411 ± 0.006 |
| 2.5 · head_scarf | 0.237 ± 0.007 | 0.306 ± 0.021 | 0.373 ± 0.014 |
| 3.5 · head_scarf | 0.248 ± 0.004 | 0.342 ± 0.007 | 0.413 ± 0.002 |
| TabM tahap 5 · rankup_noaug | 0.232 ± 0.008 | 0.246 ± 0.047 | 0.305 ± 0.024 |
| TabM tahap 5 · angular | 0.240 ± 0.014 | 0.262 ± 0.039 | 0.312 ± 0.016 |
| TabM tahap 6 · rankup | 0.158 ± 0.018 | 0.209 ± 0.004 | 0.271 ± 0.008 |
| TabM tahap 6 · rcus | 0.137 ± 0.026 | 0.218 ± 0.004 | 0.263 ± 0.015 |
| TabM tahap 6 · rcus_angular | 0.150 ± 0.016 | 0.222 ± 0.016 | 0.243 ± 0.019 |

### pendigits (macro_f1)

| Versi dan metode | 1% | 10% | 30% |
|---|---|---|---|
| 2.5 · tabpfn | 0.838 ± 0.001 | 0.948 ± 0.009 | 0.973 ± 0.003 |
| 3.5 · tabpfn | 0.873 ± 0.002 | 0.970 ± 0.002 | 0.984 ± 0.001 |
| 2.5 · head_sup | 0.825 ± 0.017 | 0.944 ± 0.010 | 0.970 ± 0.002 |
| 3.5 · head_sup | 0.845 ± 0.021 | 0.956 ± 0.006 | 0.978 ± 0.001 |
| 2.5 · head_distill | 0.839 ± 0.001 | 0.947 ± 0.010 | 0.969 ± 0.001 |
| 3.5 · head_distill | 0.872 ± 0.002 | 0.967 ± 0.004 | 0.981 ± 0.001 |
| 2.5 · head_gaussian | 0.836 ± 0.002 | 0.944 ± 0.004 | 0.968 ± 0.003 |
| 3.5 · head_gaussian | 0.873 ± 0.000 | 0.967 ± 0.002 | 0.983 ± 0.003 |
| 2.5 · head_dropout | 0.839 ± 0.001 | 0.944 ± 0.011 | 0.968 ± 0.003 |
| 3.5 · head_dropout | 0.873 ± 0.001 | 0.968 ± 0.003 | 0.980 ± 0.001 |
| 2.5 · head_fnoise | 0.836 ± 0.001 | 0.941 ± 0.005 | 0.967 ± 0.003 |
| 3.5 · head_fnoise | 0.870 ± 0.002 | 0.965 ± 0.002 | 0.981 ± 0.001 |
| 2.5 · head_angular | 0.842 ± 0.006 | 0.947 ± 0.009 | 0.970 ± 0.001 |
| 3.5 · head_angular | 0.872 ± 0.002 | 0.968 ± 0.001 | 0.982 ± 0.002 |
| 2.5 · head_tabcutmix | 0.819 ± 0.016 | 0.946 ± 0.005 | 0.966 ± 0.002 |
| 3.5 · head_tabcutmix | 0.845 ± 0.006 | 0.946 ± 0.006 | 0.978 ± 0.004 |
| 2.5 · head_zero_out | 0.840 ± 0.000 | 0.945 ± 0.005 | 0.970 ± 0.002 |
| 3.5 · head_zero_out | 0.874 ± 0.002 | 0.968 ± 0.001 | 0.981 ± 0.001 |
| 2.5 · head_scarf | 0.842 ± 0.006 | 0.944 ± 0.009 | 0.970 ± 0.001 |
| 3.5 · head_scarf | 0.874 ± 0.003 | 0.968 ± 0.002 | 0.980 ± 0.005 |
| TabM tahap 5 · rankup_noaug | 0.881 ± 0.004 | 0.949 ± 0.004 | 0.974 ± 0.001 |
| TabM tahap 5 · angular | 0.882 ± 0.004 | 0.948 ± 0.001 | 0.972 ± 0.002 |
| TabM tahap 6 · rankup | 0.842 ± 0.013 | 0.944 ± 0.000 | 0.971 ± 0.001 |
| TabM tahap 6 · rcus | 0.885 ± 0.003 | 0.973 ± 0.004 | 0.978 ± 0.000 |
| TabM tahap 6 · rcus_angular | 0.885 ± 0.006 | 0.959 ± 0.009 | 0.977 ± 0.002 |

### superconductivity (mse)

| Versi dan metode | 1% | 10% | 30% |
|---|---|---|---|
| 2.5 · tabpfn | 306.5 ± 2.0 | 165.8 ± 0.6 | 122.4 ± 0.8 |
| 3.5 · tabpfn | 318.1 ± 0.2 | 163.4 ± 0.3 | 116.4 ± 0.5 |
| 2.5 · head_sup | 365.1 ± 13.3 | 176.3 ± 0.7 | 126.4 ± 1.8 |
| 3.5 · head_sup | 384.1 ± 15.2 | 176.0 ± 3.2 | 120.4 ± 2.2 |
| 2.5 · head_distill | 303.5 ± 2.9 | 166.9 ± 2.2 | 123.5 ± 2.9 |
| 3.5 · head_distill | 314.9 ± 2.0 | 164.9 ± 2.5 | 118.3 ± 1.4 |
| 2.5 · head_gaussian | 304.2 ± 2.7 | 168.4 ± 2.8 | 123.5 ± 1.3 |
| 3.5 · head_gaussian | 312.6 ± 2.2 | 165.4 ± 2.5 | 119.8 ± 1.9 |
| 2.5 · head_dropout | 305.5 ± 0.5 | 168.8 ± 2.5 | 125.7 ± 1.2 |
| 3.5 · head_dropout | 316.1 ± 0.5 | 167.2 ± 1.4 | 118.6 ± 1.9 |
| 2.5 · head_fnoise | 304.6 ± 0.8 | 168.1 ± 2.4 | 123.8 ± 1.8 |
| 3.5 · head_fnoise | 315.6 ± 2.2 | 166.2 ± 4.3 | 118.8 ± 1.6 |
| 2.5 · head_angular | 830.6 ± 240.8 | 228.2 ± 22.4 | 150.4 ± 16.7 |
| 3.5 · head_angular | 780.1 ± 193.3 | 231.9 ± 20.0 | 156.3 ± 12.5 |
| 2.5 · head_tabcutmix | 304.5 ± 2.2 | 169.0 ± 3.0 | 123.7 ± 0.6 |
| 3.5 · head_tabcutmix | 314.5 ± 1.4 | 164.1 ± 2.2 | 118.9 ± 3.9 |
| 2.5 · head_zero_out | 684.9 ± 148.7 | 210.9 ± 34.5 | 161.2 ± 7.1 |
| 3.5 · head_zero_out | 693.9 ± 154.1 | 191.7 ± 7.3 | 149.3 ± 8.3 |
| 2.5 · head_scarf | 633.4 ± 110.5 | 212.7 ± 17.4 | 150.6 ± 7.8 |
| 3.5 · head_scarf | 551.7 ± 108.6 | 200.0 ± 13.1 | 150.3 ± 14.9 |
| TabM tahap 5 · rankup_noaug | 366.5 ± 5.5 | 208.8 ± 6.6 | 152.8 ± 9.1 |
| TabM tahap 5 · angular | 368.4 ± 4.4 | 208.7 ± 5.3 | 151.5 ± 10.6 |
| TabM tahap 6 · rankup | 392.9 ± 12.4 | 208.2 ± 2.6 | 151.7 ± 5.4 |
| TabM tahap 6 · rcus | 446.3 ± 33.8 | 208.1 ± 2.1 | 154.5 ± 1.9 |
| TabM tahap 6 · rcus_angular | 440.9 ± 21.4 | 211.2 ± 1.1 | 155.1 ± 0.5 |

## 2. Perbandingan berpasangan 3.5 vs 2.5

Setiap entri berisi menang/imbang/kalah pada sembilan pasangan fraksi label dan seed. Ambang imbang absolut 1e-9. Angka validation adalah dasar pemilihan, angka test dilaporkan untuk konfirmasi.

| Varian | nhanes validation | pendigits validation | superconductivity validation | nhanes test | pendigits test | superconductivity test |
|---|---|---|---|---|---|---|
| tabpfn | 8/0/1 | 9/0/0 | 6/0/3 | 5/0/4 | 9/0/0 | 6/0/3 |
| head_sup | 5/0/4 | 8/0/1 | 4/0/5 | 7/0/2 | 8/0/1 | 5/0/4 |
| head_distill | 4/0/5 | 9/0/0 | 6/0/3 | 4/0/5 | 9/0/0 | 5/0/4 |
| head_gaussian | 2/0/7 | 9/0/0 | 6/0/3 | 4/0/5 | 9/0/0 | 6/0/3 |
| head_dropout | 3/0/6 | 9/0/0 | 6/0/3 | 4/0/5 | 9/0/0 | 5/0/4 |
| head_fnoise | 4/0/5 | 9/0/0 | 6/0/3 | 5/0/4 | 9/0/0 | 5/0/4 |
| head_angular | 9/0/0 | 9/0/0 | 5/0/4 | 9/0/0 | 9/0/0 | 4/0/5 |
| head_tabcutmix | 3/0/6 | 9/0/0 | 5/0/4 | 3/0/6 | 8/0/1 | 5/0/4 |
| head_zero_out | 7/0/2 | 9/0/0 | 5/0/4 | 9/0/0 | 9/0/0 | 6/0/3 |
| head_scarf | 7/0/2 | 9/0/0 | 7/0/2 | 9/0/0 | 9/0/0 | 6/0/3 |

## 3. NHANES setelah kalibrasi QWK

Ambil batas kelas dari kuantil prediksi validation pada proporsi kelas empiris validation. Gunakan batas yang sama pada test. QWK mentah dari cache dicocokkan ke setiap baris JSONL sebelum kalibrasi. Cache tahap 7 menyimpan prediksi teacher `tabpfn`. Kalibrasi head memerlukan prediksi per baris yang akan disimpan tahap 8.

| Versi | Label | QWK val mentah | QWK val kalibrasi | QWK test mentah | QWK test kalibrasi | Spearman test | MSE test |
|---|---|---|---|---|---|---|---|
| 2.5 | 1 | 0.077 ± 0.057 | 0.228 ± 0.012 | 0.089 ± 0.050 | 0.215 ± 0.011 | 0.253 ± 0.004 | 0.990 ± 0.003 |
| 2.5 | 10 | 0.224 ± 0.007 | 0.329 ± 0.009 | 0.216 ± 0.009 | 0.329 ± 0.008 | 0.354 ± 0.003 | 0.947 ± 0.010 |
| 2.5 | 30 | 0.327 ± 0.007 | 0.375 ± 0.004 | 0.320 ± 0.010 | 0.376 ± 0.001 | 0.397 ± 0.003 | 0.846 ± 0.004 |
| 3.5 | 1 | 0.089 ± 0.016 | 0.264 ± 0.010 | 0.084 ± 0.031 | 0.274 ± 0.006 | 0.293 ± 0.002 | 0.901 ± 0.002 |
| 3.5 | 10 | 0.238 ± 0.006 | 0.352 ± 0.003 | 0.260 ± 0.006 | 0.338 ± 0.008 | 0.381 ± 0.001 | 0.834 ± 0.001 |
| 3.5 | 30 | 0.339 ± 0.003 | 0.393 ± 0.002 | 0.319 ± 0.003 | 0.424 ± 0.002 | 0.415 ± 0.001 | 0.799 ± 0.000 |

## 4. Dampak augmentasi pada 3.5

Selisih rerata test dari `head_distill`, memakai QWK mentah, macro-F1 dan MSE sesuai urutan. Spearman dan MSE juga ditampilkan untuk NHANES agar perubahan ranking terbaca terpisah dari sebaran prediksi.

| Varian | Label | Δ QWK | Δ Spearman | Δ MSE NHANES | Δ macro-F1 | Δ MSE Superconductivity |
|---|---|---|---|---|---|---|
| tabpfn | 1 | -0.023 | +0.006 | -0.001 | +0.002 | +3.254 |
| head_gaussian | 1 | -0.024 | -0.000 | +0.001 | +0.002 | -2.279 |
| head_dropout | 1 | -0.048 | +0.006 | +0.001 | +0.001 | +1.222 |
| head_fnoise | 1 | +0.011 | +0.004 | -0.002 | -0.002 | +0.715 |
| head_angular | 1 | +0.154 | +0.003 | +0.052 | +0.001 | +465.236 |
| head_tabcutmix | 1 | -0.001 | +0.002 | +0.001 | -0.026 | -0.390 |
| head_zero_out | 1 | +0.136 | +0.000 | +0.027 | +0.003 | +378.977 |
| head_scarf | 1 | +0.142 | +0.001 | +0.017 | +0.003 | +236.787 |
| tabpfn | 10 | -0.005 | +0.003 | -0.005 | +0.003 | -1.534 |
| head_gaussian | 10 | +0.017 | +0.001 | +0.001 | -0.000 | +0.496 |
| head_dropout | 10 | +0.007 | +0.002 | -0.004 | +0.001 | +2.280 |
| head_fnoise | 10 | +0.026 | -0.002 | +0.003 | -0.002 | +1.327 |
| head_angular | 10 | +0.081 | +0.002 | +0.257 | +0.000 | +67.026 |
| head_tabcutmix | 10 | -0.005 | +0.002 | +0.001 | -0.021 | -0.817 |
| head_zero_out | 10 | +0.078 | +0.003 | +0.163 | +0.001 | +26.839 |
| head_scarf | 10 | +0.078 | +0.003 | +0.164 | +0.001 | +35.096 |
| tabpfn | 30 | -0.002 | -0.000 | -0.005 | +0.003 | -1.907 |
| head_gaussian | 30 | +0.005 | +0.001 | -0.004 | +0.002 | +1.487 |
| head_dropout | 30 | +0.015 | +0.002 | +0.003 | -0.000 | +0.294 |
| head_fnoise | 30 | +0.008 | -0.000 | -0.000 | +0.001 | +0.477 |
| head_angular | 30 | +0.080 | -0.002 | +0.294 | +0.001 | +37.997 |
| head_tabcutmix | 30 | +0.017 | -0.000 | +0.011 | -0.003 | +0.578 |
| head_zero_out | 30 | +0.090 | -0.002 | +0.261 | +0.001 | +30.937 |
| head_scarf | 30 | +0.092 | -0.001 | +0.214 | -0.001 | +31.965 |

## 5. Waktu persiapan rata-rata per grup (detik)

Satu kali `tabpfn_seconds` per grup dari baris `tabpfn`. Ini waktu persiapan embedding dan teacher per grup, jadi angka ini tidak mencakup training head dan bukan durasi dinding sesi Kaggle dua GPU.

| Versi | Dataset | 1% | 10% | 30% |
|---|---|---|---|---|
| 2.5 | nhanes | 12.6 | 15.5 | 24.0 |
| 2.5 | pendigits | 6.8 | 8.9 | 17.5 |
| 2.5 | superconductivity | 80.8 | 228.5 | 714.6 |
| 3.5 | nhanes | 12.9 | 16.3 | 21.3 |
| 3.5 | pendigits | 14.2 | 19.9 | 34.7 |
| 3.5 | superconductivity | 53.8 | 111.2 | 247.1 |

## 6. Posisi terhadap eksperimen sebelumnya

- Tahap 4 melaporkan bahwa RankUp dengan augmentasi input menaikkan QWK NHANES 1% dari 0,150 ke 0,251 dan macro-F1 Pen Digits 1% dari 0,838 ke 0,877 pada eksperimen satu seed tanpa Optuna. Variasi antar-seed di eksperimen tersebut bisa sebesar selisih antaroperator. [Ringkasan tahap 4](../4-augmentasi-input/RINGKASAN_SEMUA_EKSPERIMEN.md)
- Tahap 5 memperlihatkan hasil angular yang dekat dengan RankUp tanpa augmentasi. Pada Superconductivity, teacher 3.5 menurunkan MSE dari 366,5 / 208,8 / 152,8 pada RankUp tanpa augmentasi ke 318,1 / 163,4 / 116,4. [Hasil tahap 5](../5-augmentasi-latent/hasil/ringkasan.md)
- Di Pen Digits, RCUS tahap 6 unggul atas teacher 3.5 pada 1% dan 10% label (0,885 vs 0,873 dan 0,973 vs 0,970). Pada 30%, teacher 3.5 mencapai 0,984 dan RCUS 0,978. [Hasil tahap 6](../6-ssr-rcus/hasil/ringkasan.md)
- Tahap 8 mengulang kontrol dengan TabPFN 3.5 dan menyimpan prediksi setiap head, sehingga QWK terkalibrasi teacher dan head bisa dibandingkan pada split yang sama. [Rancangan tahap 8](../8-tabpfn-ssl/dokumen/RANCANGAN.md)
