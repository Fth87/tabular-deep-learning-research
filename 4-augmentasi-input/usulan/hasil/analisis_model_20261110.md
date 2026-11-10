# Analisis model dan eksplorasi SSL berbasis pohon

Eksperimen validation-only, split data seed 42. Tidak ada Optuna dan tidak menggunakan label test; hasil adalah eksplorasi satu split yang validation-nya telah dipakai berkali-kali. Dua model supervised Extra Trees dan HistGradientBoosting dilatih pada subset label; model pohon SSL memakai teacher Extra Trees yang dilatih hanya dari label training. Training CPU lokal ini bukan RankUp.

## Yang ditemukan

- NHANES pada 1% memiliki 36 sampel berlabel dengan distribusi level `{0: 4, 1: 10, 2: 14, 3: 7, 4: 1}`. Ini adalah kendala observasi; kita tidak dapat menyimpulkan satu sampel langka adalah penyebab tunggal QWK yang rendah. Pseudo-label yang salah, variance split, pembulatan prediksi dan ketidakcocokan antara MSE training dan QWK adalah hipotesis yang membutuhkan pengujian, bukan temuan kausal.
- Pen Digits label 30% sudah memiliki macro-F1 validation sekitar 0,996 pada screening sebelumnya, jadi peningkatan absolutnya sempit.
- Superconductivity label 10%: Extra Trees supervised MSE 169,053; pada 30% MSE 123,722. Kandidat neural conditional-group yang diuji sebelumnya memiliki MSE 184,669 dan 137,956. Semua memakai validation dan split yang sama, tetapi training dan kapasitas model berbeda; angka tersebut bukan bukti efek augmentasi. Baseline ini menunjukkan pemilihan backbone/learning objective patut diperiksa.
- Pada Superconductivity 1%, teacher-student pohon dengan pseudo-label MSE 313,653; Extra Trees supervised 328,016. Pada 10% pseudo-label MSE 172,267 memburuk dari supervised 169,053, dan pada 30% 124,812 memburuk dari 123,722. Ini adalah screening terpilih satu seed, bukan hasil yang mengonfirmasi metode stabil.

## Rancangan pseudo-label pohon

Teacher Extra Trees 80 pohon dilatih pada label yang ada. Ia memprediksi training unlabeled; untuk klasifikasi, urutan kepercayaan berasal dari probabilitas kelas terbesar; untuk ordinal/regresi, dari simpangan baku prediksi pohon (pohon saling bergantung dan bukan uncertainty terkalibrasi). Sebanyak 20% sampel unlabeled, dibatasi 1.200, dengan peringkat kepercayaan tertinggi dipilih. Student Extra Trees dilatih pada label asli dengan bobot 1 dan pseudo-label dengan bobot 0,2. Varian kedua menambahkan interpolasi `0,85 x + 0,15 x_neighbor` jika teacher menyetujui prediksi; ini tidak menjamin validitas data atau label. Target pseudo dari teacher yang sama dapat memperkuat kesalahannya sendiri. Hasil lengkap di [baseline_cpu.csv](baseline_cpu.csv) dan [tree_ssl_explorasi.csv](tree_ssl_explorasi.csv); kode [baseline_cpu.py](../baseline_cpu.py), [tree_ssl_explorasi.py](../tree_ssl_explorasi.py).

## Prototipe integrasi RankUp

[pilot_teacher.py](../pilot_teacher.py) memanfaatkan [pseudo_teacher.py](../pseudo_teacher.py) dan menambahkan pseudo-label teacher sebanyak maksimal 50% jumlah label asli, atau 1.200 sampel, ke bagian yang diperlakukan sebagai labeled oleh loop RankUp. Sampel yang dipromosikan tetap ada dalam pool unlabeled; karena pseudo-label diperlakukan seperti label asli untuk supervised dan ranking loss, metode ini dapat memperkuat kesalahan teacher. Prototipe membandingkan terhadap conditional-group pada setting train yang sama. Ini adalah modifikasi RankUp, bukan implementasi RankUp paper, dan bobot confidence tidak terkalibrasi. Prototipe perlu diuji sebelum klaim performa; status dan hasil eksperimen harus diperiksa di folder hasil. Tes dasar fungsi pemilihan teacher lulus di Modal CPU.

Pengujian konfirmasi memerlukan data baru yang belum dipakai memilih metode, split dan seed berulang yang berpasangan, kontrol supervised dan identity, serta pelaporan interval ketidakpastian. Perbandingan lintas dataset dan metrik tidak dapat dijamin selalu unggul.
