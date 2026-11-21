# Sumber asli dan perbandingan metode

**Status:** Catatan di bawah merekam audit sebelum perbaikan. SCARF, pengulangan labeled batch, normalisasi loss, loss MSE, dan clustering TabCutMixPlus sudah diperbarui di 4-augmentasi-input. TabCutMix dan RLA tetap adaptasi; pada Satimage clustering average-linkage tetap menghasilkan dua grup masing-masing 18 fitur. Pemilihan grup kini probabilistik sehingga sebagian sampel dapat tetap tanpa pertukaran, tetapi saat satu grup terpilih 18 fitur dapat berubah sekaligus.

Catatan ini membandingkan implementasi di `experiments/rankup/` dan `experiments/rankup_classification/` dengan paper dan kode asli. Pemeriksaan berupa pembacaan kode, bukan validasi training penuh. Praktik di domain tabular adalah adaptasi; hasil tidak otomatis sama dengan hasil paper.

## RankUp

**Paper:** Huang, Fu, Tsao (NeurIPS 2024), [RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier](https://proceedings.neurips.cc/paper_files/paper/2024/file/c26a8494fe31695db965ae8b7244b7c1-Paper-Conference.pdf). **Kode resmi:** [pm25/Semi-Supervised-Regression](https://github.com/pm25/Semi-Supervised-Regression), terutama [`rankup_net.py`](https://github.com/pm25/Semi-Supervised-Regression/blob/main/semilearn/algorithms/rankup/rankup_net.py), [`rankup.py`](https://github.com/pm25/Semi-Supervised-Regression/blob/main/semilearn/algorithms/rankup/rankup.py), dan [`cls_consistency.py`](https://github.com/pm25/Semi-Supervised-Regression/blob/main/semilearn/core/criterions/cls_consistency.py). Salinannya juga ada di `../../repos/Semi-Supervised-Regression` dari root proyek `4-augmentasi-input`.

- Paper menggabungkan objective regresi dan auxiliary ranking classifier (ARC), menerapkan pseudo-label FixMatch untuk prediksi ranking pasangan weak/strong. RDA opsional; tidak adanya RDA saja bukan kesalahan. Implementasi ordinal memiliki komponen tersebut, tetapi menggunakan TabM dan augmentasi tabular, jadi adalah adaptasi.
- Dalam paper Algorithm 1 dan kode resmi, loss unlabeled yang di-mask dirata-ratakan atas **seluruh pasangan**. Implementasi TA `pseudo_rank_loss` memakai `losses[accepted].mean()`: dirata-ratakan hanya atas pasangan diterima. Jika acceptance rate rendah, loss efektif dapat jauh lebih besar daripada formulasi asli. Ini perbedaan yang perlu diselesaikan atau dinyatakan jelas sebelum mengklaim reproduksi RankUp.
- RankUp-CLS untuk Satimage tidak memiliki ARC dan memakai weak-to-strong pseudo-label classification. Secara konsep ini adaptasi FixMatch-style, **bukan RankUp regresi asli**; jangan klaim metode yang sama untuk kedua task.
- Penulis mengevaluasi beberapa seed. MSE untuk TA adalah pilihan metrik tersendiri, bukan metrik utama dalam paper RankUp, yang melaporkan MAE, R² dan korelasi Spearman.

## SCARF

**Paper:** Bahri et al. (ICLR 2022), [SCARF](https://openreview.net/pdf?id=CuV_qYkmKb3); [halaman penulis Google Research](https://research.google/pubs/scarf-self-supervised-contrastive-learning-using-random-feature-corruption/). Repo resmi penulis belum terverifikasi; [pytorch-scarf](https://github.com/clabrugere/pytorch-scarf) adalah implementasi pihak ketiga, bukan bukti resmi penulis.

- Paper memilih fitur lalu mengambil pengganti **independen dari empirical marginal masing-masing fitur**. Paper §4.4 secara eksplisit membedakan ini dari joint sampling.
- `scarf.py` memilih **satu donor per baris** kemudian memakai seluruh fitur terpilih dari donor tersebut, baik untuk ordinal maupun klasifikasi. Ini sesuai konsep joint sampling pada ablation SCARF, bukan SCARF marginal asli. Solusinya untuk reproduksi: pilih indeks donor berbeda dan independen untuk setiap sel fitur yang dikorupsi.
- Paper SCARF menggunakan kontrastif pretraining; proyek ini hanya memasukkan operator korupsi dalam RankUp/FixMatch-style, tidak mereproduksi keseluruhan SCARF.

## Range-Limited Augmentation (RLA)

**Paper:** Lee et al., [naskah Range-Limited Augmentation / FeSTA](https://openreview.net/pdf?id=kTH3bEH6hW); [repo resmi FeSTA](https://github.com/kyungeun-lee/festa), khususnya [`libs/transform.py`](https://github.com/kyungeun-lee/festa/blob/main/libs/transform.py) dan [`libs/data.py`](https://github.com/kyungeun-lee/festa/blob/main/libs/data.py).

Paper membedakan *range-limited shuffling* (memilih nilai teramati dalam rentang) dan *range-limited sampling* (uniform dalam rentang). Implementasi TA membuat empat bin berdasarkan kuantil train dan mengambil uniform di antara batas kuantil, sehingga mendekati **sampling**, bukan shuffling. Paper memilih sel dengan masker Bernoulli; TA memilih jumlah fitur tetap per baris, sedikitnya satu, dan mengoptimasi strength weak/strong. Ini modifikasi, bukan salinan penuh pipeline kontrasif FeSTA.

Batas kuantil yang berulang dapat menghasilkan interval nol dan tidak memberi perubahan; sampling uniform untuk fitur diskret dapat membuat nilai baru non-diskret. Perlu audit tipe/semantik fitur terlebih dahulu; nama fitur NHANES anonim. Pengukuran lokal oleh pemeriksa proyek menemukan 16/30 fitur NHANES dengan batas kuantil berulang; ini perlu diperhitungkan dalam laporan.

## TabCutMix dan TabCutMixPlus

**Paper:** Fang et al. (ICML 2025), [Understanding and Mitigating Memorization in Diffusion Models for Tabular Data](https://proceedings.mlr.press/v267/fang25f.html). **Kode penulis:** [`cutmix_intra.py`](https://github.com/fangzy96/TabCutMix/blob/main/cutmix_intra.py), [`process_dataset_tabcutmixplus.py`](https://github.com/fangzy96/TabCutMix/blob/main/process_dataset_tabcutmixplus.py).

TabCutMix menukar subset fitur dari pasangan kelas sama menggunakan label kelas pada data training; TabCutMixPlus mengelompokkan fitur berdasarkan korelasi dan menukar sekelompok fitur. Implementasi TA mencari donor dalam mini-batch dengan pseudo-label dan cutoff; untuk ordinal, pseudo-label diperoleh dengan pembulatan prediksi regresi. Donor pseudo-label sama bukan jaminan kelas sebenarnya sama. Jika donor yang memenuhi syarat tidak ada, fitur tidak ditukar. Ini adaptasi untuk data tanpa label, bukan reproduksi langsung eksperimen data diffusion paper.

Dalam kode asli Plus, clustering memakai average-linkage pada jarak `1 - |korelasi|` dan menangani fitur campuran; TA menggunakan connected components dari korelasi Pearson absolut dengan threshold 0,5. Dua algoritme ini dapat membentuk grup sangat berbeda. Pemeriksa proyek menemukan dua grup berukuran 18+18 untuk Satimage dan 17 grup untuk NHANES; di Satimage, aturan pilih minimal satu grup berarti kekuatan lemah sekalipun dapat menukar setidaknya setengah fitur saat donor ada. Ini merupakan risiko perbandingan weak/strong dan bukan algoritme clustering resmi.

## Gaussian dan Zero Out

**Gaussian:** [paper GsRCL](https://academic.oup.com/bfg/article/23/4/441/7577350), [kode penulis](https://github.com/ibrahimsaggaf/GsRCL). Kode TA menambahkan Gaussian independen di setiap fitur numerik; ini operator Gaussian umum, bukan implementasi lengkap pipeline kontrasif GsRCL pada single-cell RNA-seq. Gaussian noise juga dibahas sebagai ablation dalam [paper SCARF](https://openreview.net/pdf?id=CuV_qYkmKb3).

**Zero Out:** [paper SubTab](https://papers.nips.cc/paper/2021/file/9c8661befae6dbcd08304dbf4dcaf0db-Paper.pdf), [repo resmi](https://github.com/AstraZeneca/SubTab). Kode TA men-set fitur terpilih ke nol; ini sejalan dengan ide noise zero-out, tetapi bukan keseluruhan metode SubTab. Paper memakai masker Bernoulli sedangkan kode TA memilih jumlah tetap sedikitnya satu fitur; setelah standardisasi, nilai nol mengacu pada sekitar rata-rata training, tidak selalu nilai hilang.

## Temuan implementasi dan rancangan eksperimen

- **Pengulangan labeled batch pada RankUp ordinal:** setiap file ordinal membuat `cycle(labeled_loader)` di luar loop epoch. `itertools.cycle` menyimpan batch dari iterasi pertama dan mengulangnya, sehingga batch yang sama dan urutannya dipakai terus, meskipun DataLoader menggunakan `shuffle=True`. Ini berlaku di keenam runner ordinal, contohnya `experiments/rankup/gaussian.py:150`. Runner klasifikasi membuat `cycle` ulang setiap epoch, tetapi masih mengulang urutan batch dalam satu epoch setelah satu putaran. Perlu ubah iterator agar DataLoader diiterasi lagi ketika habis.
- **Pemilihan versus loss MSE:** NHANES memakai MSE untuk Optuna, pemilihan checkpoint, dan metrik test, tetapi loss regresi berlabel tetap L1 (`experiments/rankup/gaussian.py:176`). Ini bukan bug apabila MSE hanya metrik evaluasi; jika rancangan TA mengharuskan training menggunakan MSE, perlu diubah dan dinyatakan sebagai perubahan metode.
- **Perbandingan augmentasi:** keenam varian memakai augmentasi; tidak ada baseline tanpa augmentasi atau baseline supervised-only pada runner ini. Karena itu, perbandingan saat ini dapat menunjukkan perbedaan antarvarian, tetapi tidak dapat mengukur apakah RankUp maupun augmentasi meningkatkan kinerja dibanding tanpa komponen tersebut.
- **Reliabilitas:** training baru untuk 4-augmentasi-input belum dijalankan dalam pemeriksaan ini. Temuan merupakan analisis kode dan perhitungan karakteristik split, bukan bukti efek pada skor. Rerun sebaiknya dilakukan setelah rancangan metode ditetapkan; simpan seed dan konfigurasi, lalu ulangi dengan beberapa seed.
