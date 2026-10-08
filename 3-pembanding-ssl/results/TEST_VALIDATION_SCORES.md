# Tabel skor validation dan test

Metrik validasi yang tersedia dan seluruh metrik test untuk 100 run. Skor test adalah held-out test; validasi ditampilkan sesuai metrik yang benar-benar direkam tiap metode. `—` berarti nilai validasi tidak tersimpan. Nilai ditampilkan 4 desimal; CSV menyimpan presisi penuh.

## Klasifikasi

| Dataset | Model / augmentasi | Validation (metrik skor) | Test accuracy | Test macro-precision | Test macro-recall | Test macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| adult_income | GFTab | Loss 0.4482 | 0.8570 | 0.8093 | 0.7844 | 0.7955 |
| adult_income | RankUp-CLS · gaussian | Macro-F1 0.7949 | 0.8435 | 0.7838 | 0.7964 | 0.7897 |
| adult_income | RankUp-CLS · range_limited | Macro-F1 0.7938 | 0.8469 | 0.7880 | 0.8061 | 0.7962 |
| adult_income | RankUp-CLS · scarf | Macro-F1 0.7926 | 0.8470 | 0.7884 | 0.8025 | 0.7949 |
| adult_income | RankUp-CLS · tabcutmix | Macro-F1 0.7942 | 0.8439 | 0.7842 | 0.7992 | 0.7912 |
| adult_income | RankUp-CLS · tabcutmix_plus | Macro-F1 0.7932 | 0.8433 | 0.7836 | 0.7971 | 0.7899 |
| adult_income | RankUp-CLS · zero_out | Macro-F1 0.7949 | 0.8417 | 0.7813 | 0.8011 | 0.7901 |
| adult_income | SSLAE | — | 0.8411 | 0.7937 | 0.7421 | 0.7619 |
| adult_income | STUNT | Accuracy 0.7904 | 0.7969 | 0.7270 | 0.7553 | 0.7382 |
| adult_income | VIME | Loss 0.3782 | 0.8304 | 0.7820 | 0.7145 | 0.7375 |
| bank_marketing | GFTab | Loss 0.4163 | 0.8941 | 0.7520 | 0.6408 | 0.6742 |
| bank_marketing | RankUp-CLS · gaussian | Macro-F1 0.7486 | 0.8943 | 0.7440 | 0.7438 | 0.7439 |
| bank_marketing | RankUp-CLS · range_limited | Macro-F1 0.7399 | 0.8947 | 0.7453 | 0.7106 | 0.7260 |
| bank_marketing | RankUp-CLS · scarf | Macro-F1 0.7368 | 0.8966 | 0.7502 | 0.7238 | 0.7359 |
| bank_marketing | RankUp-CLS · tabcutmix | Macro-F1 0.7391 | 0.8965 | 0.7496 | 0.7319 | 0.7402 |
| bank_marketing | RankUp-CLS · tabcutmix_plus | Macro-F1 0.7394 | 0.8975 | 0.7525 | 0.7286 | 0.7397 |
| bank_marketing | RankUp-CLS · zero_out | Macro-F1 0.7456 | 0.8972 | 0.7519 | 0.7241 | 0.7368 |
| bank_marketing | SSLAE | — | 0.8810 | 0.7207 | 0.7696 | 0.7412 |
| bank_marketing | STUNT | Accuracy 0.8604 | 0.8596 | 0.5522 | 0.5201 | 0.5187 |
| bank_marketing | VIME | Loss 0.3112 | 0.8888 | 0.7701 | 0.5503 | 0.5630 |
| cmc | GFTab | Loss 1.0087 | 0.4796 | 0.4577 | 0.4516 | 0.4510 |
| cmc | RankUp-CLS · gaussian | Macro-F1 0.5169 | 0.4887 | 0.4851 | 0.4634 | 0.4680 |
| cmc | RankUp-CLS · range_limited | Macro-F1 0.5253 | 0.5520 | 0.5306 | 0.5341 | 0.5312 |
| cmc | RankUp-CLS · scarf | Macro-F1 0.5253 | 0.5520 | 0.5306 | 0.5341 | 0.5312 |
| cmc | RankUp-CLS · tabcutmix | Macro-F1 0.5253 | 0.5520 | 0.5306 | 0.5341 | 0.5312 |
| cmc | RankUp-CLS · tabcutmix_plus | Macro-F1 0.5253 | 0.5520 | 0.5306 | 0.5341 | 0.5312 |
| cmc | RankUp-CLS · zero_out | Macro-F1 0.5097 | 0.5611 | 0.5401 | 0.5420 | 0.5396 |
| cmc | SSLAE | — | 0.4389 | 0.2914 | 0.3769 | 0.3284 |
| cmc | STUNT | Accuracy 0.4480 | 0.4072 | 0.4049 | 0.4144 | 0.3839 |
| cmc | VIME | Loss 1.0258 | 0.4615 | 0.4511 | 0.4304 | 0.4330 |
| diabetes_pima | GFTab | Loss 0.6134 | 0.7069 | 0.6929 | 0.7112 | 0.6938 |
| diabetes_pima | RankUp-CLS · gaussian | Macro-F1 0.7817 | 0.7500 | 0.7231 | 0.7145 | 0.7181 |
| diabetes_pima | RankUp-CLS · range_limited | Macro-F1 0.7819 | 0.8017 | 0.7803 | 0.7895 | 0.7843 |
| diabetes_pima | RankUp-CLS · scarf | Macro-F1 0.7734 | 0.7500 | 0.7236 | 0.7086 | 0.7143 |
| diabetes_pima | RankUp-CLS · tabcutmix | Macro-F1 0.7765 | 0.7931 | 0.7709 | 0.7770 | 0.7737 |
| diabetes_pima | RankUp-CLS · tabcutmix_plus | Macro-F1 0.7737 | 0.7672 | 0.7467 | 0.7632 | 0.7517 |
| diabetes_pima | RankUp-CLS · zero_out | Macro-F1 0.7793 | 0.7845 | 0.7617 | 0.7586 | 0.7601 |
| diabetes_pima | SSLAE | — | 0.7586 | 0.7348 | 0.7447 | 0.7387 |
| diabetes_pima | STUNT | Accuracy 0.6552 | 0.5431 | 0.5247 | 0.5270 | 0.5208 |
| diabetes_pima | VIME | Loss 0.5368 | 0.6466 | 0.5891 | 0.5704 | 0.5698 |
| satimage | GFTab | Loss 0.9495 | 0.8477 | 0.8349 | 0.8111 | 0.8209 |
| satimage | RankUp-CLS · gaussian | Macro-F1 0.8673 | 0.8725 | 0.8522 | 0.8415 | 0.8448 |
| satimage | RankUp-CLS · range_limited | Macro-F1 0.8504 | 0.8549 | 0.8268 | 0.8251 | 0.8252 |
| satimage | RankUp-CLS · scarf | Macro-F1 0.8638 | 0.8549 | 0.8378 | 0.8240 | 0.8296 |
| satimage | RankUp-CLS · tabcutmix | Macro-F1 0.8575 | 0.8570 | 0.8317 | 0.8249 | 0.8267 |
| satimage | RankUp-CLS · tabcutmix_plus | Macro-F1 0.8515 | 0.8560 | 0.8374 | 0.8214 | 0.8278 |
| satimage | RankUp-CLS · zero_out | Macro-F1 0.8505 | 0.8560 | 0.8291 | 0.8321 | 0.8297 |
| satimage | SSLAE | — | 0.8290 | 0.7896 | 0.7947 | 0.7914 |
| satimage | STUNT | Accuracy 0.5938 | 0.5959 | 0.5954 | 0.5374 | 0.5455 |
| satimage | VIME | Loss 0.4079 | 0.8591 | 0.8349 | 0.8184 | 0.8229 |

## Ordinal regression

| Dataset | Model / augmentasi | Validation (metrik skor) | Test MAE | Test RMSE | Test R² | Test Spearman | Test QWK |
|---|---|---:|---:|---:|---:|---:|---:|
| esl | GFTab | MAE 0.5405 | 0.5676 | 0.8220 | 0.6272 | 0.8449 | 0.7387 |
| esl | RankUp · gaussian | MAE 0.3661 | 0.4145 | 0.5129 | 0.8548 | 0.9363 | 0.8918 |
| esl | RankUp · range_limited | MAE 0.3722 | 0.4154 | 0.5151 | 0.8536 | 0.9362 | 0.9019 |
| esl | RankUp · scarf | MAE 0.3610 | 0.4283 | 0.5207 | 0.8504 | 0.9294 | 0.9007 |
| esl | RankUp · tabcutmix | MAE 0.3755 | 0.4154 | 0.5145 | 0.8540 | 0.9362 | 0.9019 |
| esl | RankUp · tabcutmix_plus | MAE 0.3762 | 0.4163 | 0.5148 | 0.8538 | 0.9362 | 0.9019 |
| esl | RankUp · zero_out | MAE 0.3649 | 0.4152 | 0.5120 | 0.8554 | 0.9369 | 0.8931 |
| esl | SSLAE | — | 0.6892 | 0.9515 | 0.5005 | 0.7351 | 0.7121 |
| esl | STUNT | MAE 0.5541 | 0.5270 | 0.8302 | 0.6197 | 0.8901 | 0.8368 |
| esl | VIME | MAE 0.4324 | 0.5000 | 0.7798 | 0.6645 | 0.8526 | 0.7905 |
| lev | GFTab | MAE 0.5800 | 0.5733 | 0.8327 | 0.2418 | 0.5254 | 0.4437 |
| lev | RankUp · gaussian | MAE 0.4818 | 0.4821 | 0.6820 | 0.4914 | 0.7073 | 0.6871 |
| lev | RankUp · range_limited | MAE 0.4842 | 0.4876 | 0.6744 | 0.5027 | 0.7134 | 0.6871 |
| lev | RankUp · scarf | MAE 0.4872 | 0.4997 | 0.6949 | 0.4719 | 0.6883 | 0.6591 |
| lev | RankUp · tabcutmix | MAE 0.4836 | 0.4838 | 0.6725 | 0.5054 | 0.7157 | 0.6848 |
| lev | RankUp · tabcutmix_plus | MAE 0.4836 | 0.4838 | 0.6725 | 0.5054 | 0.7157 | 0.6848 |
| lev | RankUp · zero_out | MAE 0.4808 | 0.4834 | 0.6855 | 0.4861 | 0.7082 | 0.6804 |
| lev | SSLAE | — | 0.5267 | 0.8124 | 0.2783 | 0.5840 | 0.5319 |
| lev | STUNT | MAE 0.7533 | 0.6933 | 0.9661 | -0.0206 | 0.5930 | 0.5855 |
| lev | VIME | MAE 0.5000 | 0.5733 | 0.8083 | 0.2856 | 0.5400 | 0.4416 |
| melbourne_airbnb | GFTab | MAE 1.3422 | 1.2931 | 1.9420 | 0.5821 | 0.7923 | 0.7950 |
| melbourne_airbnb | RankUp · gaussian | MAE 1.3232 | 1.2975 | 1.7614 | 0.6562 | 0.8052 | 0.8039 |
| melbourne_airbnb | RankUp · range_limited | MAE 1.3241 | 1.3037 | 1.7734 | 0.6515 | 0.8036 | 0.8030 |
| melbourne_airbnb | RankUp · scarf | MAE 1.3189 | 1.2884 | 1.7339 | 0.6669 | 0.8101 | 0.8054 |
| melbourne_airbnb | RankUp · tabcutmix | MAE 1.3195 | 1.2954 | 1.7616 | 0.6561 | 0.8058 | 0.8045 |
| melbourne_airbnb | RankUp · tabcutmix_plus | MAE 1.3185 | 1.2933 | 1.7574 | 0.6578 | 0.8060 | 0.8036 |
| melbourne_airbnb | RankUp · zero_out | MAE 1.3206 | 1.3057 | 1.7651 | 0.6548 | 0.8038 | 0.8011 |
| melbourne_airbnb | SSLAE | — | 1.4168 | 2.0797 | 0.5208 | 0.7626 | 0.7701 |
| melbourne_airbnb | STUNT | MAE 2.6617 | 2.6510 | 3.3618 | -0.2523 | 0.2515 | 0.2667 |
| melbourne_airbnb | VIME | MAE 1.8247 | 1.8217 | 2.5445 | 0.2826 | 0.6589 | 0.6692 |
| nhanes | GFTab | MAE 0.7207 | 0.7283 | 0.9904 | -0.0053 | 0.2305 | 0.1782 |
| nhanes | RankUp · gaussian | MAE 0.7431 | 0.7482 | 0.9352 | 0.1037 | 0.3444 | 0.2814 |
| nhanes | RankUp · range_limited | MAE 0.7413 | 0.7473 | 0.9314 | 0.1109 | 0.3528 | 0.2927 |
| nhanes | RankUp · scarf | MAE 0.7415 | 0.7445 | 0.9315 | 0.1107 | 0.3439 | 0.2745 |
| nhanes | RankUp · tabcutmix | MAE 0.7446 | 0.7522 | 0.9416 | 0.0913 | 0.3416 | 0.2873 |
| nhanes | RankUp · tabcutmix_plus | MAE 0.7445 | 0.7492 | 0.9375 | 0.0993 | 0.3434 | 0.2868 |
| nhanes | RankUp · zero_out | MAE 0.7435 | 0.7498 | 0.9380 | 0.0983 | 0.3450 | 0.2829 |
| nhanes | SSLAE | — | 0.7474 | 1.0227 | -0.0719 | 0.2740 | 0.2535 |
| nhanes | STUNT | MAE 1.0051 | 1.0064 | 1.3078 | -0.7530 | 0.1716 | 0.1631 |
| nhanes | VIME | MAE 0.7194 | 0.7143 | 0.9781 | 0.0196 | 0.2456 | 0.1723 |
| student_performance | GFTab | MAE 1.6000 | 1.7200 | 2.1954 | -0.5091 | -0.1118 | -0.1259 |
| student_performance | RankUp · gaussian | MAE 1.3591 | 1.3636 | 1.7640 | 0.0258 | 0.2274 | 0.1465 |
| student_performance | RankUp · range_limited | MAE 1.3552 | 1.3597 | 1.7605 | 0.0296 | 0.2305 | 0.1652 |
| student_performance | RankUp · scarf | MAE 1.3552 | 1.3597 | 1.7605 | 0.0296 | 0.2305 | 0.1652 |
| student_performance | RankUp · tabcutmix | MAE 1.3552 | 1.3597 | 1.7605 | 0.0296 | 0.2305 | 0.1652 |
| student_performance | RankUp · tabcutmix_plus | MAE 1.3552 | 1.3597 | 1.7605 | 0.0296 | 0.2305 | 0.1652 |
| student_performance | RankUp · zero_out | MAE 1.3591 | 1.3636 | 1.7640 | 0.0258 | 0.2274 | 0.1465 |
| student_performance | SSLAE | — | 1.5400 | 1.9900 | -0.2399 | 0.1973 | 0.1582 |
| student_performance | STUNT | MAE 2.7500 | 2.8800 | 3.4409 | -2.7071 | 0.0820 | 0.0790 |
| student_performance | VIME | MAE 1.4000 | 1.4500 | 1.9261 | -0.1616 | 0.1333 | 0.0902 |

Catatan: untuk RankUp, nilai validation adalah skor utama yang dipakai Optuna (Macro-F1 untuk klasifikasi, MAE untuk ordinal). Baseline bisa menyimpan metrik validasi berbeda, misalnya loss atau accuracy; nama metrik dicantumkan di sel.
