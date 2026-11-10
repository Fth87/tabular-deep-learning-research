# Augmentasi feature-guided pada RankUp

Percobaan menambahkan **satu operator augmentasi**, tanpa teacher pohon dan tanpa mengubah objective RankUp. Semua 9 job (3 dataset × 1%, 10%, 30% label) sukses; seed 42; tanpa Optuna; maksimum 120 epoch, 100 langkah/epoch, patience 15. Weak view identitas, strong view mengganti 30% fitur dari donor bersyarat; maksimal 2 T4 paralel. Tidak membaca test. Klasifikasi nominal tetap memakai adaptasi SSL multiclass proyek tanpa auxiliary ranking classifier, bukan RankUp regresi asli.

| Dataset / metrik | Label | Feature-guided | Conditional random | Terbaik dari pembanding eksploratif |
|---|---:|---:|---:|---|
| NHANES / QWK ↑ | 1% | 0,289418 | 0,300668 | Gibbs k5 s50 0,309909 |
| NHANES / QWK ↑ | 10% | 0,296198 | 0,293529 | SCARF/Gibbs k20 s50 0,300610 |
| NHANES / QWK ↑ | 30% | 0,331696 | 0,331696 | Kontrol supervised `none` 0,348463 |
| Pen Digits / macro-F1 ↑ | 1% | 0,939805 | 0,940328 | Gibbs k5 s30 0,945551 |
| Pen Digits / macro-F1 ↑ | 10% | **0,989356** | 0,987364 | Feature-guided 0,989356 |
| Pen Digits / macro-F1 ↑ | 30% | **0,996709** | 0,994678 | Feature-guided 0,996709 |
| Superconductivity / MSE ↓ | 1% | 337,106559 | 338,877035 | Gibbs k20 s50 327,002111 |
| Superconductivity / MSE ↓ | 10% | 185,491557 | 185,452484 | Conditional group 184,668657 |
| Superconductivity / MSE ↓ | 30% | 141,054573 | 141,314062 | Conditional group + soft rank 137,822461 |

Feature-guided terbaik dalam pembanding eksploratif pada 2/9 kondisi dan selisihnya relatif kecil; turun dari conditional random pada NHANES 1%, Pen Digits 1% dan Superconductivity 10%. Perbandingan group + soft rank juga mengubah loss pseudo-rank, jadi bukan ablation augmentasi murni. Kontrol `none` mengatur bobot loss unlabeled ke nol, bukan identity consistency. Kesamaan skor yang dicetak hingga 6 desimal belum tentu bit-identik. Hasil ini tidak membuktikan keunggulan, kestabilan, atau signifikansi: validation yang sama berulang dipakai menyaring kandidat, hanya satu seed, dan kekuatan antaroperator tidak terkalibrasi sama.

## Cara kerja

Hitung asosiasi fitur dengan target **hanya pada labeled train**: nilai absolut korelasi Pearson untuk ordinal/regresi, dan rasio variasi antarkelas terhadap total untuk klasifikasi. Jika asosiasi fitur `s_j ∈ [0,1]`, bobot pengambilan fitur untuk dikorupsi ialah `w_j = 1 - 0,75 × n/(n+100) × s_j`. Semakin kecil sampel berlabel, semakin bobot mendekati pemilihan seragam; ini bentuk shrinkage heuristik, bukan estimasi uncertainty yang terkalibrasi. Secara acak pilih 30% fitur tanpa pengembalian berdasarkan bobot tersebut. Dari pool train hingga 2.048 baris, pilih acak salah satu dari 5 donor terdekat berdasarkan fitur yang dipertahankan, lalu salin fitur yang dipilih. Jika tidak ada donor sah, gunakan sampel asal.

Asosiasi univariat tidak mengukur interaksi, kausalitas, atau semua hubungan nonlinier. Fitur yang tampak lemah masih bisa penting; bobot yang lebih kecil bukan larangan korupsi. Donor berasal dari fitur train labeled/unlabeled tanpa menggunakan label donor. Metode tidak menjamin target tetap sama, distribusi gabungan terjaga, atau kombinasi fisik valid. Khusus NHANES, 36 label pada 1% membuat asosiasi rentan noise meski shrinkage digunakan.

Menghindari perubahan pada fitur yang dianggap penting bukan ide baru: [AGATa, workshop NeurIPS 2024](https://openreview.net/pdf?id=l5VGHbRVW0) memakai perhatian untuk memandu augmentasi fitur pada pembelajaran kontrasif. Prototipe ini menggunakan asosiasi label dan conditional donor pada pipeline RankUp, bukan implementasi AGATa. Kebaruan dan manfaat kombinasinya perlu dibuktikan terhadap prior art serta diuji dengan ablation dan holdout baru.

Data, epoch, pseudo-rank acceptance dan diagnostik operator ada dalam [CSV perbandingan](feature-guided-perbandingan.csv) serta direktori `feature-guided-20261110/`. Checksum input seluruh metode yang dibandingkan pada setiap kondisi cocok. Operator dan tes ada di [augmentasi_terarah.py](../augmentasi_terarah.py) dan [test_augmentasi_terarah.py](../test_augmentasi_terarah.py); tes Modal CPU lulus. Ringkasan dapat direproduksi dengan `uv run python usulan/ringkas_terarah.py`. Kode training tidak diubah untuk mencoba operator ini; checkpoint tidak disimpan dalam pilot.

SHA-256 saat run:

```text
10ae128528bd38ac204a649ab8be1466094f1591f1ddcdfd304fb76a7a03078f  usulan/augmentasi_terarah.py
f17dfe486e748081e5a301b693d8c862d2e049c639777b35d3afb3aae8fade2b  usulan/pilot_terarah.py
4ca34b492151d4a09fda1e3ba1939520d7ce698a6a5a7bf3fe4ca6b192fae88a  usulan/modal_terarah.py
```
