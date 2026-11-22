# Ringkasan SSR-RCUS tabular

Dibuat otomatis oleh `ringkas.py` dari `results.jsonl` (189 run).

## 1. Skor test (rata-rata ± std antar 3 seed)

**nhanes** (qwk)

| name                | 1             | 10            | 30            |
|:--------------------|:--------------|:--------------|:--------------|
| rankup              | 0.158 ± 0.018 | 0.209 ± 0.004 | 0.271 ± 0.008 |
| rcus                | 0.137 ± 0.026 | 0.218 ± 0.004 | 0.263 ± 0.015 |
| rcus_angular        | 0.150 ± 0.016 | 0.222 ± 0.016 | 0.243 ± 0.019 |
| rcus_angular_weak   | 0.145 ± 0.012 | 0.222 ± 0.013 | 0.263 ± 0.011 |
| rcus_input_gaussian | 0.069 ± 0.028 | 0.156 ± 0.035 | 0.229 ± 0.024 |
| rcus_input_zero_out | 0.154 ± 0.025 | 0.226 ± 0.037 | 0.271 ± 0.011 |
| rcus_input_scarf    | 0.162 ± 0.022 | 0.269 ± 0.018 | 0.285 ± 0.017 |

**pendigits** (macro_f1)

| name                | 1             | 10            | 30            |
|:--------------------|:--------------|:--------------|:--------------|
| rankup              | 0.842 ± 0.013 | 0.944 ± 0.000 | 0.971 ± 0.001 |
| rcus                | 0.885 ± 0.003 | 0.973 ± 0.004 | 0.978 ± 0.000 |
| rcus_angular        | 0.885 ± 0.006 | 0.959 ± 0.009 | 0.977 ± 0.002 |
| rcus_angular_weak   | 0.879 ± 0.006 | 0.963 ± 0.012 | 0.978 ± 0.003 |
| rcus_input_gaussian | 0.884 ± 0.012 | 0.957 ± 0.004 | 0.977 ± 0.001 |
| rcus_input_zero_out | 0.894 ± 0.010 | 0.955 ± 0.010 | 0.971 ± 0.002 |
| rcus_input_scarf    | 0.826 ± 0.007 | 0.945 ± 0.005 | 0.969 ± 0.002 |

**superconductivity** (mse)

| name                | 1                | 10               | 30              |
|:--------------------|:-----------------|:-----------------|:----------------|
| rankup              | 392.901 ± 12.384 | 208.170 ± 2.551  | 151.700 ± 5.414 |
| rcus                | 446.292 ± 33.828 | 208.062 ± 2.122  | 154.463 ± 1.867 |
| rcus_angular        | 440.882 ± 21.394 | 211.234 ± 1.068  | 155.103 ± 0.526 |
| rcus_angular_weak   | 437.867 ± 18.005 | 213.147 ± 5.920  | 155.491 ± 1.980 |
| rcus_input_gaussian | 568.661 ± 36.043 | 231.403 ± 6.549  | 180.070 ± 3.502 |
| rcus_input_zero_out | 611.710 ± 93.810 | 241.785 ± 26.077 | 169.277 ± 8.473 |
| rcus_input_scarf    | 446.467 ± 11.807 | 204.336 ± 5.445  | 165.203 ± 5.098 |

## 2. Perbandingan berpasangan: berapa blok yang dimenangkan

Satu blok = satu (label, seed), jadi 9 blok per dataset. "Menang" = skor varian lebih baik dari pembanding. Hanya hitungan menang yang dilaporkan: selisih skor tidak dirata-ratakan lintas dataset karena skalanya berbeda (MSE vs QWK), dan dengan 9 blok per dataset belum ada uji yang berarti.

| varian | vs | split | nhanes | pendigits | superconductivity | total |
|---|---|---|---:|---:|---:|---:|
| rcus | rankup | test | 3/9 | 9/9 | 2/9 | 14/27 |
| rcus | rankup | val | 3/9 | 9/9 | 0/9 | 12/27 |
| rcus_angular | rcus | test | 4/9 | 4/9 | 3/9 | 11/27 |
| rcus_angular | rcus | val | 6/9 | 5/9 | 4/9 | 15/27 |
| rcus_angular_weak | rcus | test | 5/9 | 4/9 | 4/9 | 13/27 |
| rcus_angular_weak | rcus | val | 3/9 | 5/9 | 2/9 | 10/27 |
| rcus_input_gaussian | rcus | test | 0/9 | 2/9 | 0/9 | 2/27 |
| rcus_input_gaussian | rcus | val | 0/9 | 0/9 | 0/9 | 0/27 |
| rcus_input_zero_out | rcus | test | 5/9 | 2/9 | 0/9 | 7/27 |
| rcus_input_zero_out | rcus | val | 1/9 | 3/9 | 0/9 | 4/27 |
| rcus_input_scarf | rcus | test | 8/9 | 0/9 | 4/9 | 12/27 |
| rcus_input_scarf | rcus | val | 5/9 | 0/9 | 3/9 | 8/27 |
