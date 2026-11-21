# Ringkasan eksperimen latent

Dibuat otomatis oleh `kode/ringkas.py` dari `results.jsonl` (945 run). Jangan diedit manual.

Metrik: NHANES = QWK (↑), Pen Digits = Macro F1 (↑), Superconductivity = MSE (↓).
Konfigurasi dipilih dari **validation**; angka test hanya dilaporkan.

## 1. Skor test (rata-rata ± std antar seed)

| name                  | nhanes 1%     | nhanes 10%    | nhanes 30%    | pendigits 1%   | pendigits 10%   | pendigits 30%   | superconductivity 1%   | superconductivity 10%   | superconductivity 30%   |
|:----------------------|:--------------|:--------------|:--------------|:---------------|:----------------|:----------------|:-----------------------|:------------------------|:------------------------|
| angular               | 0.240 ± 0.014 | 0.262 ± 0.039 | 0.312 ± 0.016 | 0.882 ± 0.004  | 0.948 ± 0.001   | 0.972 ± 0.002   | 368.364 ± 4.438        | 208.679 ± 5.342         | 151.476 ± 10.624        |
| angular_prerelu       | 0.221 ± 0.004 | 0.230 ± 0.007 | 0.319 ± 0.010 | 0.884 ± 0.001  | 0.949 ± 0.001   | 0.971 ± 0.003   | 384.368 ± 9.356        | 206.546 ± 4.342         | 150.074 ± 4.430         |
| angular_s10           | 0.243 ± 0.013 | 0.255 ± 0.042 | 0.315 ± 0.014 | 0.877 ± 0.002  | 0.947 ± 0.003   | 0.965 ± 0.002   | 369.341 ± 4.605        | 205.052 ± 7.291         | 153.293 ± 9.124         |
| angular_s50           | 0.234 ± 0.010 | 0.253 ± 0.043 | 0.312 ± 0.016 | 0.884 ± 0.005  | 0.951 ± 0.001   | 0.973 ± 0.001   | 369.815 ± 5.180        | 204.170 ± 7.490         | 152.837 ± 9.251         |
| angular_weak          | 0.243 ± 0.016 | 0.253 ± 0.050 | 0.307 ± 0.020 | 0.878 ± 0.003  | 0.947 ± 0.003   | 0.972 ± 0.002   | 376.574 ± 13.400       | 204.163 ± 5.642         | 153.836 ± 8.621         |
| angular_weak90        | 0.239 ± 0.013 | 0.237 ± 0.002 | 0.312 ± 0.015 | 0.879 ± 0.003  | 0.947 ± 0.002   | 0.973 ± 0.002   | 376.466 ± 11.292       | 204.415 ± 6.183         | 153.992 ± 8.799         |
| angular_weak90_s10    | 0.247 ± 0.008 | 0.235 ± 0.013 | 0.312 ± 0.015 | 0.875 ± 0.004  | 0.946 ± 0.002   | 0.965 ± 0.007   | 378.896 ± 9.202        | 206.346 ± 5.749         | 147.925 ± 2.855         |
| angular_weak90_s50    | 0.233 ± 0.004 | 0.254 ± 0.048 | 0.311 ± 0.016 | 0.880 ± 0.004  | 0.950 ± 0.004   | 0.974 ± 0.002   | 375.908 ± 14.588       | 204.278 ± 6.933         | 153.522 ± 8.556         |
| angular_weak_prerelu  | 0.224 ± 0.010 | 0.261 ± 0.043 | 0.309 ± 0.017 | 0.888 ± 0.004  | 0.950 ± 0.001   | 0.971 ± 0.002   | 382.491 ± 9.273        | 202.293 ± 5.667         | 153.561 ± 9.055         |
| angular_weak_s10      | 0.242 ± 0.013 | 0.250 ± 0.051 | 0.313 ± 0.015 | 0.878 ± 0.004  | 0.945 ± 0.004   | 0.966 ± 0.002   | 377.869 ± 10.612       | 204.834 ± 6.364         | 151.172 ± 5.022         |
| angular_weak_s50      | 0.234 ± 0.010 | 0.249 ± 0.054 | 0.310 ± 0.017 | 0.880 ± 0.002  | 0.951 ± 0.002   | 0.972 ± 0.003   | 376.016 ± 14.376       | 204.142 ± 6.109         | 151.562 ± 10.687        |
| both_gaussian         | 0.207 ± 0.038 | 0.254 ± 0.058 | 0.311 ± 0.013 | 0.877 ± 0.003  | 0.953 ± 0.007   | 0.973 ± 0.008   | 378.749 ± 12.132       | 203.035 ± 8.272         | 152.144 ± 3.733         |
| both_scarf            | 0.280 ± 0.004 | 0.275 ± 0.028 | 0.311 ± 0.011 | 0.843 ± 0.004  | 0.937 ± 0.002   | 0.962 ± 0.005   | 381.875 ± 11.760       | 206.008 ± 5.420         | 153.020 ± 6.056         |
| both_tabcutmix        | 0.205 ± 0.021 | 0.246 ± 0.012 | 0.308 ± 0.010 | 0.863 ± 0.004  | 0.950 ± 0.006   | 0.967 ± 0.004   | 376.249 ± 8.355        | 207.925 ± 5.011         | 153.474 ± 6.444         |
| both_zero_out         | 0.239 ± 0.003 | 0.263 ± 0.048 | 0.313 ± 0.012 | 0.877 ± 0.003  | 0.943 ± 0.003   | 0.963 ± 0.003   | 382.276 ± 10.182       | 206.079 ± 8.073         | 153.575 ± 5.266         |
| input_gaussian        | 0.209 ± 0.020 | 0.260 ± 0.041 | 0.314 ± 0.017 | 0.868 ± 0.004  | 0.953 ± 0.005   | 0.976 ± 0.003   | 381.368 ± 11.924       | 202.909 ± 8.268         | 155.406 ± 8.244         |
| input_scarf           | 0.241 ± 0.003 | 0.242 ± 0.003 | 0.309 ± 0.016 | 0.839 ± 0.006  | 0.952 ± 0.002   | 0.968 ± 0.004   | 375.028 ± 5.450        | 203.738 ± 6.357         | 153.771 ± 11.479        |
| input_tabcutmix       | 0.223 ± 0.016 | 0.256 ± 0.046 | 0.305 ± 0.016 | 0.855 ± 0.013  | 0.941 ± 0.004   | 0.971 ± 0.002   | 377.161 ± 6.429        | 207.898 ± 5.121         | 153.335 ± 3.723         |
| input_zero_out        | 0.253 ± 0.013 | 0.231 ± 0.004 | 0.318 ± 0.015 | 0.880 ± 0.005  | 0.948 ± 0.003   | 0.969 ± 0.003   | 381.003 ± 11.194       | 203.918 ± 7.107         | 154.672 ± 8.710         |
| latent_dropout        | 0.239 ± 0.010 | 0.262 ± 0.043 | 0.312 ± 0.018 | 0.887 ± 0.004  | 0.951 ± 0.004   | 0.974 ± 0.002   | 366.915 ± 5.620        | 203.954 ± 5.926         | 149.102 ± 5.191         |
| latent_fnoise         | 0.233 ± 0.003 | 0.256 ± 0.048 | 0.312 ± 0.018 | 0.883 ± 0.005  | 0.950 ± 0.003   | 0.976 ± 0.000   | 380.039 ± 8.106        | 204.045 ± 6.199         | 152.995 ± 9.169         |
| latent_gaussian       | 0.234 ± 0.006 | 0.261 ± 0.042 | 0.306 ± 0.026 | 0.880 ± 0.000  | 0.948 ± 0.001   | 0.975 ± 0.001   | 379.421 ± 7.307        | 202.998 ± 7.065         | 145.730 ± 1.612         |
| latent_gaussian_weak  | 0.232 ± 0.011 | 0.247 ± 0.051 | 0.310 ± 0.012 | 0.883 ± 0.007  | 0.952 ± 0.004   | 0.975 ± 0.002   | 369.265 ± 6.102        | 207.298 ± 3.970         | 151.116 ± 9.801         |
| latent_scarf          | 0.241 ± 0.002 | 0.260 ± 0.038 | 0.309 ± 0.013 | 0.886 ± 0.004  | 0.950 ± 0.001   | 0.971 ± 0.002   | 367.334 ± 6.894        | 206.677 ± 3.872         | 149.875 ± 5.518         |
| latent_scarf_weak     | 0.240 ± 0.002 | 0.253 ± 0.040 | 0.313 ± 0.016 | 0.887 ± 0.004  | 0.951 ± 0.002   | 0.973 ± 0.002   | 366.613 ± 5.148        | 208.403 ± 3.033         | 150.542 ± 4.249         |
| latent_tabcutmix      | 0.240 ± 0.003 | 0.256 ± 0.039 | 0.314 ± 0.014 | 0.874 ± 0.001  | 0.950 ± 0.003   | 0.972 ± 0.002   | 369.174 ± 5.311        | 203.588 ± 6.328         | 151.793 ± 9.763         |
| latent_tabcutmix_weak | 0.235 ± 0.005 | 0.251 ± 0.051 | 0.312 ± 0.015 | 0.871 ± 0.003  | 0.948 ± 0.002   | 0.973 ± 0.004   | 378.902 ± 10.889       | 203.889 ± 5.831         | 151.609 ± 10.065        |
| latent_zero_out       | 0.234 ± 0.012 | 0.261 ± 0.048 | 0.310 ± 0.020 | 0.881 ± 0.002  | 0.951 ± 0.001   | 0.973 ± 0.002   | 366.146 ± 6.085        | 207.417 ± 4.441         | 153.884 ± 7.418         |
| latent_zero_out_weak  | 0.230 ± 0.004 | 0.248 ± 0.049 | 0.308 ± 0.014 | 0.880 ± 0.007  | 0.951 ± 0.001   | 0.974 ± 0.001   | 367.250 ± 6.294        | 207.927 ± 3.396         | 150.617 ± 6.179         |
| matched_dropout       | 0.231 ± 0.006 | 0.266 ± 0.038 | 0.315 ± 0.015 | 0.885 ± 0.006  | 0.945 ± 0.003   | 0.973 ± 0.002   | 368.812 ± 3.827        | 204.706 ± 6.753         | 151.501 ± 9.892         |
| matched_gaussian      | 0.243 ± 0.010 | 0.252 ± 0.008 | 0.313 ± 0.017 | 0.893 ± 0.004  | 0.950 ± 0.004   | 0.973 ± 0.001   | 364.536 ± 7.037        | 207.756 ± 4.729         | 151.586 ± 10.957        |
| matched_scarf         | 0.243 ± 0.009 | 0.260 ± 0.032 | 0.308 ± 0.014 | 0.879 ± 0.004  | 0.943 ± 0.003   | 0.970 ± 0.002   | 366.227 ± 7.144        | 208.851 ± 4.264         | 154.019 ± 8.889         |
| matched_zero_out      | 0.246 ± 0.017 | 0.270 ± 0.034 | 0.311 ± 0.019 | 0.874 ± 0.001  | 0.942 ± 0.004   | 0.968 ± 0.001   | 369.046 ± 3.919        | 208.713 ± 5.206         | 154.617 ± 8.859         |
| rankup_noaug          | 0.232 ± 0.008 | 0.246 ± 0.047 | 0.305 ± 0.024 | 0.881 ± 0.004  | 0.949 ± 0.004   | 0.974 ± 0.001   | 366.511 ± 5.456        | 208.847 ± 6.612         | 152.755 ± 9.089         |
| supervised            | 0.169 ± 0.016 | 0.254 ± 0.041 | 0.311 ± 0.018 | 0.837 ± 0.000  | 0.950 ± 0.004   | 0.969 ± 0.002   | 380.546 ± 16.366       | 207.701 ± 5.100         | 151.922 ± 9.264         |

## 2. Peringkat rata-rata (1 = terbaik)

Blok lengkap: 27 (dataset × label × seed). Urut menurut validation.

| name                  |   rank_val |   rank_test |
|:----------------------|-----------:|------------:|
| matched_gaussian      |      12.31 |       11.3  |
| latent_scarf_weak     |      13.63 |       13.7  |
| latent_scarf          |      13.78 |       13.8  |
| latent_dropout        |      14.07 |       10.39 |
| latent_gaussian       |      15.41 |       14.41 |
| latent_fnoise         |      15.56 |       14.7  |
| angular_weak90_s50    |      15.57 |       17.02 |
| angular_weak_prerelu  |      15.63 |       18.74 |
| angular               |      15.72 |       16.63 |
| angular_weak_s50      |      15.8  |       17.41 |
| rankup_noaug          |      15.94 |       18.24 |
| latent_zero_out_weak  |      16.19 |       17.57 |
| angular_s50           |      16.44 |       15.96 |
| latent_zero_out       |      16.46 |       14.67 |
| latent_gaussian_weak  |      16.48 |       15.28 |
| angular_weak          |      16.52 |       19.8  |
| angular_weak90        |      16.59 |       19.17 |
| matched_scarf         |      16.85 |       20.76 |
| latent_tabcutmix      |      17.24 |       15.11 |
| latent_tabcutmix_weak |      17.37 |       19.09 |
| angular_s10           |      17.7  |       19.07 |
| angular_prerelu       |      17.89 |       17.89 |
| matched_dropout       |      18.11 |       13.91 |
| both_tabcutmix        |      18.26 |       23.3  |
| angular_weak_s10      |      18.41 |       21.02 |
| angular_weak90_s10    |      20.89 |       22.39 |
| matched_zero_out      |      21.07 |       21.89 |
| input_scarf           |      21.43 |       19.06 |
| both_zero_out         |      21.46 |       21.76 |
| supervised            |      21.46 |       23.06 |
| input_zero_out        |      22.72 |       18.76 |
| input_tabcutmix       |      22.87 |       24.2  |
| both_gaussian         |      23.94 |       17.98 |
| both_scarf            |      24.26 |       22.56 |
| input_gaussian        |      25.94 |       19.43 |

## 3. Perbandingan berpasangan

Pasangan per (dataset, label, seed). "B menang" = proporsi pasangan saat konfigurasi pertama (B) lebih baik dari yang kedua.

Ada 44 pembanding per split; anggap p < 0.0011 (Bonferroni) sebagai bukti kuat.

| operator   | pembanding                                | split   |   n | B menang   |   p Wilcoxon |
|:-----------|:------------------------------------------|:--------|----:|:-----------|-------------:|
| gaussian   | latent_gaussian vs input_gaussian         | val     |  27 | 85%        |       0.0002 |
| gaussian   | latent_gaussian vs input_gaussian         | test    |  27 | 59%        |       0.0655 |
| gaussian   | both_gaussian vs input_gaussian           | val     |  27 | 56%        |       0.2386 |
| gaussian   | both_gaussian vs input_gaussian           | test    |  27 | 67%        |       0.1624 |
| gaussian   | latent_gaussian_weak vs latent_gaussian   | val     |  27 | 44%        |       0.427  |
| gaussian   | latent_gaussian_weak vs latent_gaussian   | test    |  27 | 52%        |       0.8408 |
| gaussian   | input_gaussian vs rankup_noaug            | val     |  27 | 11%        |       0.003  |
| gaussian   | input_gaussian vs rankup_noaug            | test    |  27 | 52%        |       0.2901 |
| gaussian   | latent_gaussian vs rankup_noaug           | val     |  27 | 52%        |       0.5149 |
| gaussian   | latent_gaussian vs rankup_noaug           | test    |  27 | 67%        |       0.3012 |
| zero_out   | latent_zero_out vs input_zero_out         | val     |  27 | 67%        |       0.1111 |
| zero_out   | latent_zero_out vs input_zero_out         | test    |  27 | 59%        |       0.2198 |
| zero_out   | both_zero_out vs input_zero_out           | val     |  27 | 44%        |       0.9341 |
| zero_out   | both_zero_out vs input_zero_out           | test    |  27 | 33%        |       0.2386 |
| zero_out   | latent_zero_out_weak vs latent_zero_out   | val     |  27 | 44%        |       0.5098 |
| zero_out   | latent_zero_out_weak vs latent_zero_out   | test    |  27 | 37%        |       0.2386 |
| zero_out   | input_zero_out vs rankup_noaug            | val     |  27 | 33%        |       0.2584 |
| zero_out   | input_zero_out vs rankup_noaug            | test    |  27 | 48%        |       0.4553 |
| zero_out   | latent_zero_out vs rankup_noaug           | val     |  27 | 48%        |       0.546  |
| zero_out   | latent_zero_out vs rankup_noaug           | test    |  27 | 56%        |       0.427  |
| scarf      | latent_scarf vs input_scarf               | val     |  27 | 67%        |       0.3865 |
| scarf      | latent_scarf vs input_scarf               | test    |  27 | 63%        |       0.1111 |
| scarf      | both_scarf vs input_scarf                 | val     |  27 | 41%        |       0.3483 |
| scarf      | both_scarf vs input_scarf                 | test    |  27 | 41%        |       0.2483 |
| scarf      | latent_scarf_weak vs latent_scarf         | val     |  27 | 52%        |       0.7702 |
| scarf      | latent_scarf_weak vs latent_scarf         | test    |  27 | 56%        |       0.9153 |
| scarf      | input_scarf vs rankup_noaug               | val     |  27 | 41%        |       0.8593 |
| scarf      | input_scarf vs rankup_noaug               | test    |  27 | 48%        |       0.3608 |
| scarf      | latent_scarf vs rankup_noaug              | val     |  27 | 48%        |       0.3997 |
| scarf      | latent_scarf vs rankup_noaug              | test    |  27 | 56%        |       0.3483 |
| tabcutmix  | latent_tabcutmix vs input_tabcutmix       | val     |  27 | 70%        |       0.0385 |
| tabcutmix  | latent_tabcutmix vs input_tabcutmix       | test    |  27 | 89%        |       0.0006 |
| tabcutmix  | both_tabcutmix vs input_tabcutmix         | val     |  27 | 74%        |       0.0585 |
| tabcutmix  | both_tabcutmix vs input_tabcutmix         | test    |  27 | 52%        |       0.9717 |
| tabcutmix  | latent_tabcutmix_weak vs latent_tabcutmix | val     |  27 | 52%        |       0.7496 |
| tabcutmix  | latent_tabcutmix_weak vs latent_tabcutmix | test    |  27 | 22%        |       0.0055 |
| tabcutmix  | input_tabcutmix vs rankup_noaug           | val     |  27 | 33%        |       0.0655 |
| tabcutmix  | input_tabcutmix vs rankup_noaug           | test    |  27 | 37%        |       0.3126 |
| tabcutmix  | latent_tabcutmix vs rankup_noaug          | val     |  27 | 37%        |       0.2901 |
| tabcutmix  | latent_tabcutmix vs rankup_noaug          | test    |  27 | 56%        |       0.6964 |
| -          | angular vs rankup_noaug                   | val     |  27 | 48%        |       0.6109 |
| -          | angular vs rankup_noaug                   | test    |  27 | 52%        |       0.6277 |
| -          | angular_prerelu vs rankup_noaug           | val     |  27 | 48%        |       1      |
| -          | angular_prerelu vs rankup_noaug           | test    |  27 | 52%        |       0.8593 |
| -          | angular_s10 vs rankup_noaug               | val     |  27 | 52%        |       0.546  |
| -          | angular_s10 vs rankup_noaug               | test    |  27 | 48%        |       0.9341 |
| -          | angular_s50 vs rankup_noaug               | val     |  27 | 56%        |       0.3809 |
| -          | angular_s50 vs rankup_noaug               | test    |  27 | 59%        |       0.2483 |
| -          | angular_weak vs rankup_noaug              | val     |  27 | 48%        |       0.427  |
| -          | angular_weak vs rankup_noaug              | test    |  27 | 44%        |       0.441  |
| -          | angular_weak90 vs rankup_noaug            | val     |  27 | 48%        |       0.4133 |
| -          | angular_weak90 vs rankup_noaug            | test    |  27 | 44%        |       0.427  |
| -          | angular_weak90_s10 vs rankup_noaug        | val     |  27 | 41%        |       0.9529 |
| -          | angular_weak90_s10 vs rankup_noaug        | test    |  27 | 37%        |       0.4846 |
| -          | angular_weak90_s50 vs rankup_noaug        | val     |  27 | 52%        |       0.3242 |
| -          | angular_weak90_s50 vs rankup_noaug        | test    |  27 | 56%        |       0.8966 |
| -          | angular_weak_prerelu vs rankup_noaug      | val     |  27 | 48%        |       0.3997 |
| -          | angular_weak_prerelu vs rankup_noaug      | test    |  27 | 48%        |       0.8966 |
| -          | angular_weak_s10 vs rankup_noaug          | val     |  27 | 52%        |       0.4553 |
| -          | angular_weak_s10 vs rankup_noaug          | test    |  27 | 44%        |       0.6446 |
| -          | angular_weak_s50 vs rankup_noaug          | val     |  27 | 52%        |       0.2109 |
| -          | angular_weak_s50 vs rankup_noaug          | test    |  27 | 48%        |       0.7317 |
| -          | latent_dropout vs rankup_noaug            | val     |  27 | 52%        |       0.5303 |
| -          | latent_dropout vs rankup_noaug            | test    |  27 | 67%        |       0.0619 |
| -          | latent_fnoise vs rankup_noaug             | val     |  27 | 56%        |       0.2478 |
| -          | latent_fnoise vs rankup_noaug             | test    |  27 | 63%        |       0.4996 |
| -          | matched_dropout vs rankup_noaug           | val     |  27 | 44%        |       0.8779 |
| -          | matched_dropout vs rankup_noaug           | test    |  27 | 67%        |       0.2109 |
| -          | matched_gaussian vs rankup_noaug          | val     |  27 | 59%        |       0.0585 |
| -          | matched_gaussian vs rankup_noaug          | test    |  27 | 67%        |       0.0491 |
| -          | matched_scarf vs rankup_noaug             | val     |  27 | 48%        |       0.8593 |
| -          | matched_scarf vs rankup_noaug             | test    |  27 | 37%        |       0.4846 |
| -          | matched_zero_out vs rankup_noaug          | val     |  27 | 37%        |       0.3997 |
| -          | matched_zero_out vs rankup_noaug          | test    |  27 | 41%        |       0.2584 |
| angular    | angular_prerelu vs angular                | val     |  27 | 44%        |       0.6752 |
| angular    | angular_prerelu vs angular                | test    |  27 | 44%        |       0.6385 |
| angular    | angular_weak_prerelu vs angular_weak      | val     |  27 | 56%        |       0.6766 |
| angular    | angular_weak_prerelu vs angular_weak      | test    |  27 | 44%        |       0.8093 |
| matched    | matched_dropout vs angular                | val     |  27 | 37%        |       0.1909 |
| matched    | matched_dropout vs angular                | test    |  27 | 63%        |       0.5303 |
| matched    | matched_gaussian vs angular               | val     |  27 | 74%        |       0.0319 |
| matched    | matched_gaussian vs angular               | test    |  27 | 74%        |       0.0121 |
| matched    | matched_scarf vs angular                  | val     |  27 | 59%        |       0.6277 |
| matched    | matched_scarf vs angular                  | test    |  27 | 33%        |       0.1349 |
| matched    | matched_zero_out vs angular               | val     |  27 | 30%        |       0.0731 |
| matched    | matched_zero_out vs angular               | test    |  27 | 37%        |       0.2291 |
| -          | rankup_noaug vs supervised                | val     |  27 | 67%        |       0.0912 |
| -          | rankup_noaug vs supervised                | test    |  27 | 59%        |       0.2277 |

## 4. Cosine aktual cos(h, h′) di latent

Rata-rata selama training; kolom kosong = view itu tidak diganggu di latent.

| name                  |   cos_weak |   cos_strong |
|:----------------------|-----------:|-------------:|
| angular_s10           |    nan     |        0.2   |
| angular_weak90_s10    |      0.945 |        0.2   |
| angular_weak_s10      |      0.825 |        0.2   |
| matched_dropout       |    nan     |        0.393 |
| matched_zero_out      |    nan     |        0.394 |
| angular_weak          |      0.825 |        0.4   |
| angular_weak90        |      0.945 |        0.4   |
| angular_prerelu       |    nan     |        0.4   |
| angular_weak_prerelu  |      0.825 |        0.4   |
| angular               |    nan     |        0.4   |
| matched_gaussian      |    nan     |        0.431 |
| matched_scarf         |    nan     |        0.444 |
| angular_weak90_s50    |      0.945 |        0.6   |
| angular_s50           |    nan     |        0.6   |
| angular_weak_s50      |      0.825 |        0.6   |
| latent_scarf          |    nan     |        0.625 |
| latent_scarf_weak     |      0.919 |        0.625 |
| both_scarf            |    nan     |        0.63  |
| both_zero_out         |    nan     |        0.705 |
| latent_dropout        |    nan     |        0.705 |
| latent_zero_out       |    nan     |        0.705 |
| latent_zero_out_weak  |      0.948 |        0.705 |
| latent_tabcutmix      |    nan     |        0.73  |
| latent_tabcutmix_weak |      0.942 |        0.731 |
| both_tabcutmix        |    nan     |        0.742 |
| both_gaussian         |    nan     |        0.897 |
| latent_gaussian       |    nan     |        0.899 |
| latent_gaussian_weak  |      0.995 |        0.899 |
| latent_fnoise         |    nan     |        0.986 |
