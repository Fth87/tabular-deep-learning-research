# Temuan screening conditional — 10 November 2026

Semua 24 job selesai. Tiga varian diuji pada tiga dataset dan 1%, 10%, 30% label; varian soft-rank hanya pada ordinal dan regresi karena klasifikasi tidak memiliki ranking head. Satu seed 42, tanpa Optuna, maksimal 120 epoch, 100 langkah per epoch, patience 15. Pilot baru tidak membuka test. Hasil dari 24 job dibandingkan dengan 63 job screening donor bersyarat yang sudah ada di workspace. Checksum manifest serta labeled, unlabeled dan validation dalam dua kumpulan data cocok untuk setiap kondisi. Masing-masing screening memakai budget maksimal yang sama; awal acak dan implementasi operator tidak identik. Skrip pembanding hanya memakai skor validation dari screening lama.

| Dataset (metrik) | Label | Random conditional | Group conditional | Group + soft rank | Terbaik dalam kumpulan pembanding |
|---|---:|---:|---:|---:|---|
| NHANES (QWK ↑) | 1% | 0,300668 | 0,224653 | 0,199060 | Gibbs k5 s50: 0,309909 |
| NHANES (QWK ↑) | 10% | 0,293529 | 0,292713 | 0,300525 | SCARF / Gibbs k20 s50: 0,300610 |
| NHANES (QWK ↑) | 30% | 0,331696 | 0,331696 | 0,331106 | none: 0,348463 |
| Pen Digits (macro-F1 ↑) | 1% | 0,940328 | 0,931197 | — | Gibbs k5 s30: 0,945551 |
| Pen Digits (macro-F1 ↑) | 10% | 0,987364 | 0,986727 | — | none: 0,988051 |
| Pen Digits (macro-F1 ↑) | 30% | 0,994678 | 0,995377 | — | Gibbs k5 s30: 0,996016 |
| Superconductivity (MSE ↓) | 1% | 338,877035 | 336,737284 | 333,868944 | Gibbs k20 s50: 327,002111 |
| Superconductivity (MSE ↓) | 10% | 185,452484 | 184,668657 | 186,281382 | group conditional: 184,668657 |
| Superconductivity (MSE ↓) | 30% | 141,314062 | 137,955703 | 137,822461 | group + soft rank: 137,822461 |

Group conditional menjadi terbaik pada Superconductivity 10%; group + soft rank pada 30%. Pada 30%, selisih group + soft rank dan group hard hanya 0,133 MSE; dibanding SCARF di screening lama selisihnya sekitar 2,09%. Di NHANES 1%, grup + soft rank lebih rendah dari group hard dan random conditional. Karena hanya satu seed dan validasi sudah berulang kali dipakai untuk pemilihan, ini **bukan bukti** stabilitas, signifikansi atau keunggulan universal. Pembobotan lunak tidak dapat diklaim sebagai penyebab peningkatan terlepas dari interaksi training dan keacakan.

Nama `none` pada screening lama berarti bobot loss unlabeled nol; ia bukan kontrol consistency identity dengan bobot unlabeled aktif. Screening lama menyimpan hasil test, sehingga test itu sudah terekspos dan perlu holdout baru untuk konfirmasi. Angka pada tabel seluruhnya validation; angka dari pilot 30 epoch sebelumnya tidak boleh dibandingkan langsung dengan eksperimen ini sebagai efek metode karena budget berbeda.

## Detail desain dan artefak

Definisi random/group donor dan bobot pseudo-rank, inspirasi serta keterbatasan dibahas di [catatan riset](../../eksperimen/riset_lanjutan_20261110.md). Implementasi ada di [conditional_aug.py](../conditional_aug.py). Pada eksperimen soft, field `pseudo_accept_rate` dalam history mengandung rerata **bobot**, bukan rasio diterima; JSON menandainya dengan `history_pseudo_accept_rate_is_mean_soft_weight`. Hasil lengkap serta epoch ada di [CSV perbandingan](conditional-screen-perbandingan.csv); skor dan history tiap run tersedia di `conditional-screen-20261110/`. Jalankan `uv run python usulan/ringkas_conditional.py` untuk membangun ulang CSV. Tiga tes operator dan gradien lulus di Modal CPU; training memakai maksimal tiga T4, tanpa instal PyTorch lokal.

SHA-256 kode saat diluncurkan:

```text
dc072fa7d1c9233b496c30197a7c9e84ccafba0280a52f002d7394895e8c6984  usulan/conditional_aug.py
9a5b71dab212f5d7bf75afd8a7bdcae0c381d54a608cf7041606877ec4253524  usulan/pilot_conditional.py
d3aaabd9b86bbf9bf770aea67ce1b2f4e08a9a7783ab83e0c4d767af7da31a80  usulan/modal_conditional.py
```

Untuk penelitian konfirmasi, kunci kandidat dan hiperparameter, sertakan kontrol identitas dan supervised serta pembanding SSR-RCUS yang benar, gunakan beberapa split dan seed berpasangan, serta evaluasi sekali pada holdout baru yang tidak dipakai untuk menyaring kandidat. Investigasi disagreement antar-head TabM sebagai proksi risiko merupakan ide berikutnya, bukan komponen yang sudah diuji atau confidence terkalibrasi.
