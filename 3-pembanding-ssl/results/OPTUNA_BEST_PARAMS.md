# Tabel best parameter Optuna

Total: **60 hasil tuning** (6 augmentasi × 10 dataset). Parameter dipilih berdasarkan metrik validasi; skor test tidak dipakai untuk tuning.

## Klasifikasi

| Dataset | Augmentasi | Jumlah trial | Trial terbaik | Macro-F1 validasi | Parameter terbaik |
|---|---|---:|---:|---:|---|
| adult_income | gaussian | 40 | 4 | 0.794885 | `relative_gap=0.708365`, `weak_sigma=0.147909` |
| adult_income | range_limited | 40 | 21 | 0.793770 | `relative_gap=0.626018`, `weak_strength=0.652936` |
| adult_income | scarf | 40 | 20 | 0.792586 | `relative_gap=0.306349`, `weak_q=13` |
| adult_income | tabcutmix | 40 | 28 | 0.794167 | `relative_gap=0.589216`, `weak_strength=0.132035` |
| adult_income | tabcutmix_plus | 40 | 39 | 0.793220 | `relative_gap=0.845478`, `weak_strength=0.10525` |
| adult_income | zero_out | 40 | 0 | 0.794860 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| bank_marketing | gaussian | 40 | 23 | 0.748559 | `relative_gap=0.897896`, `weak_sigma=0.00863398` |
| bank_marketing | range_limited | 40 | 6 | 0.739942 | `relative_gap=0.213127`, `weak_strength=0.792496` |
| bank_marketing | scarf | 40 | 25 | 0.736822 | `relative_gap=0.0767121`, `weak_q=1` |
| bank_marketing | tabcutmix | 40 | 26 | 0.739072 | `relative_gap=0.995577`, `weak_strength=0.78696` |
| bank_marketing | tabcutmix_plus | 40 | 3 | 0.739352 | `relative_gap=0.86631`, `weak_strength=0.0645986` |
| bank_marketing | zero_out | 40 | 15 | 0.745615 | `relative_gap=0.775943`, `weak_strength=0.018132` |
| cmc | gaussian | 40 | 30 | 0.516926 | `relative_gap=0.751862`, `weak_sigma=0.267993` |
| cmc | range_limited | 40 | 0 | 0.525293 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| cmc | scarf | 40 | 0 | 0.525293 | `relative_gap=0.950764`, `weak_q=9` |
| cmc | tabcutmix | 40 | 0 | 0.525293 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| cmc | tabcutmix_plus | 40 | 0 | 0.525293 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| cmc | zero_out | 40 | 0 | 0.509666 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| diabetes_pima | gaussian | 40 | 36 | 0.781731 | `relative_gap=0.485076`, `weak_sigma=0.00508414` |
| diabetes_pima | range_limited | 40 | 10 | 0.781866 | `relative_gap=0.473854`, `weak_strength=0.922074` |
| diabetes_pima | scarf | 40 | 1 | 0.773418 | `relative_gap=0.59906`, `weak_q=6` |
| diabetes_pima | tabcutmix | 40 | 12 | 0.776456 | `relative_gap=0.398645`, `weak_strength=0.176914` |
| diabetes_pima | tabcutmix_plus | 40 | 36 | 0.773659 | `relative_gap=0.467923`, `weak_strength=0.101268` |
| diabetes_pima | zero_out | 40 | 2 | 0.779267 | `relative_gap=0.156839`, `weak_strength=0.156658` |
| satimage | gaussian | 40 | 21 | 0.867304 | `relative_gap=0.720881`, `weak_sigma=0.227612` |
| satimage | range_limited | 40 | 38 | 0.850423 | `relative_gap=0.889727`, `weak_strength=0.384868` |
| satimage | scarf | 40 | 30 | 0.863772 | `relative_gap=0.26653`, `weak_q=19` |
| satimage | tabcutmix | 40 | 16 | 0.857516 | `relative_gap=0.206272`, `weak_strength=0.27259` |
| satimage | tabcutmix_plus | 40 | 2 | 0.851463 | `relative_gap=0.156839`, `weak_strength=0.156658` |
| satimage | zero_out | 40 | 32 | 0.850548 | `relative_gap=0.256079`, `weak_strength=0.042436` |

## Ordinal regression

| Dataset | Augmentasi | Jumlah trial | Trial terbaik | MAE validasi | Parameter terbaik |
|---|---|---:|---:|---:|---|
| esl | gaussian | 40 | 27 | 0.366085 | `relative_gap=0.883356`, `weak_sigma=0.047882` |
| esl | range_limited | 40 | 1 | 0.372229 | `relative_gap=0.59906`, `weak_strength=0.698074` |
| esl | scarf | 6 | 0 | 0.360979 | `strong_q=2`, `weak_q=1` |
| esl | tabcutmix | 40 | 3 | 0.375543 | `relative_gap=0.86631`, `weak_strength=0.0645986` |
| esl | tabcutmix_plus | 40 | 0 | 0.376202 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| esl | zero_out | 40 | 11 | 0.364909 | `relative_gap=0.256263`, `weak_strength=0.212776` |
| lev | gaussian | 40 | 18 | 0.481803 | `relative_gap=0.156953`, `weak_sigma=0.243916` |
| lev | range_limited | 40 | 2 | 0.484218 | `relative_gap=0.156839`, `weak_strength=0.156658` |
| lev | scarf | 6 | 1 | 0.487156 | `strong_q=3`, `weak_q=1` |
| lev | tabcutmix | 40 | 0 | 0.483639 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| lev | tabcutmix_plus | 40 | 0 | 0.483639 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| lev | zero_out | 40 | 6 | 0.480847 | `relative_gap=0.213127`, `weak_strength=0.792496` |
| melbourne_airbnb | gaussian | 40 | 11 | 1.323150 | `relative_gap=0.0119936`, `weak_sigma=0.00521722` |
| melbourne_airbnb | range_limited | 40 | 36 | 1.324102 | `relative_gap=0.159241`, `weak_strength=0.0641653` |
| melbourne_airbnb | scarf | 40 | 35 | 1.318929 | `relative_gap=0.0923669`, `weak_q=9` |
| melbourne_airbnb | tabcutmix | 40 | 19 | 1.319495 | `relative_gap=0.84487`, `weak_strength=0.358111` |
| melbourne_airbnb | tabcutmix_plus | 40 | 23 | 1.318536 | `relative_gap=0.124388`, `weak_strength=0.367184` |
| melbourne_airbnb | zero_out | 40 | 39 | 1.320606 | `relative_gap=0.756297`, `weak_strength=0.519518` |
| nhanes | gaussian | 40 | 33 | 0.743068 | `relative_gap=0.270433`, `weak_sigma=0.0501076` |
| nhanes | range_limited | 40 | 7 | 0.741268 | `relative_gap=0.184221`, `weak_strength=0.180915` |
| nhanes | scarf | 40 | 18 | 0.741498 | `relative_gap=0.664099`, `weak_q=5` |
| nhanes | tabcutmix | 40 | 8 | 0.744617 | `relative_gap=0.525232`, `weak_strength=0.295988` |
| nhanes | tabcutmix_plus | 40 | 39 | 0.744486 | `relative_gap=0.146798`, `weak_strength=0.10525` |
| nhanes | zero_out | 40 | 5 | 0.743500 | `relative_gap=0.96994`, `weak_strength=0.0293494` |
| student_performance | gaussian | 40 | 0 | 1.359076 | `relative_gap=0.950764`, `weak_sigma=0.04126` |
| student_performance | range_limited | 40 | 0 | 1.355232 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| student_performance | scarf | 40 | 0 | 1.355232 | `relative_gap=0.950764`, `weak_q=16` |
| student_performance | tabcutmix | 40 | 0 | 1.355232 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| student_performance | tabcutmix_plus | 40 | 0 | 1.355232 | `relative_gap=0.950764`, `weak_strength=0.362068` |
| student_performance | zero_out | 40 | 0 | 1.359076 | `relative_gap=0.950764`, `weak_strength=0.362068` |

Macro-F1 dimaksimalkan dan MAE diminimalkan. Indeks trial dimulai dari 0. Ada 58 run dengan 40 trial dan 2 run SCARF (ESL dan LEV) dengan 6 trial; jumlah trial mengikuti hasil yang tersimpan.
