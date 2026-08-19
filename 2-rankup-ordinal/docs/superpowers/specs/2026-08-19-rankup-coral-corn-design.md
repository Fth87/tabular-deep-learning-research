# RankUp dengan Regression, CORAL, dan CORN

## Tujuan

Buat `kode/rankup_all_datasets_coral_corn.ipynb` berdasarkan
`kode/rankup_all_datasets.ipynb`. Notebook baru mempertahankan semua dataset,
backbone, skenario training, metrik, logging, CSV, dan visualisasi notebook
acuan, lalu menambahkan CORAL dan CORN sebagai objective ordinal pada setiap
backbone.

## Ruang lingkup eksperimen

- Dataset: Melbourne Airbnb, Student Performance, NHANES, ESL, dan LEV.
- Backbone: MLP, TabM, dan RealMLP-style backbone dari notebook acuan.
- Objective: Regression, CORAL, dan CORN.
- Skenario: Supervised, Fully-Supervised, RankUp + FESTA, dan RankUp +
  TabCutMix.
- Seed: satu seed (`42`), sama dengan notebook acuan.
- Total training run: `5 × 3 × 3 × 4 = 180`.

Notebook tidak melakukan hyperparameter tuning dan tidak mengubah split,
jumlah labeled data, augmentasi, epoch, optimizer, atau bobot loss dari
notebook acuan.

## Data dan target

Split train/test dan pemilihan `N_LABEL` tetap stratified dan identik dengan
notebook acuan.

Data yang disiapkan untuk setiap dataset:

- fitur hasil `StandardScaler`;
- target kelas integer `0..K-1` untuk CORAL, CORN, ranking loss, dan evaluasi;
- target hasil `StandardScaler` hanya untuk objective Regression;
- metadata `num_classes`, `class_min`, dan `class_max`.

Scaler fitur hanya di-fit pada train set. Scaler target Regression mengikuti
notebook acuan agar hasil Regression tetap dapat dibandingkan langsung.

## Model dan output head

Semua backbone memiliki prediction head dan pairwise ranking head yang
terpisah. Ranking head selalu menghasilkan dua logits dan hanya digunakan oleh
RankUp.

### Regression

Prediction head menghasilkan satu nilai. Training memakai L1 loss pada target
yang sudah distandardisasi. Inference mengembalikan nilai ke skala kelas asli
dengan target scaler.

### CORAL

Prediction head menghasilkan `K-1` cumulative threshold logits. Untuk MLP dan
RealMLP, head memakai `CoralLayer` dari `coral-pytorch`. Untuk TabM, setiap
ensemble member menghasilkan satu latent rank logit ditambah `K-1` bias
threshold terurut, sehingga weight-sharing CORAL tetap berlaku di dalam setiap
member.

Training memakai `coral_loss`. Target diubah menjadi extended binary levels.

### CORN

Prediction head menghasilkan `K-1` conditional logits. Training memakai
`corn_loss` dengan target kelas integer. CORN tidak memakai constraint
weight-sharing CORAL.

### TabM

Loss dihitung untuk setiap ensemble member secara independen dengan meratakan
loss member, bukan menghitung loss dari logits rata-rata. Saat inference,
probabilitas cumulative setiap member dirata-ratakan sebelum menghasilkan skor
akhir.

## RankUp

Mekanisme RankUp tetap sama:

- labeled data: prediction loss sesuai objective dan supervised pairwise rank
  loss;
- unlabeled data: pseudo-ranking dari weak view ke strong view;
- FESTA dan TabCutMix hanya membentuk weak/strong view dan tidak membuat
  pseudo-target kelas langsung;
- `P_CUTOFF`, `W_RANK`, dan `W_UNLABELED` tetap sama dengan notebook acuan.

Objective CORAL/CORN hanya menggantikan prediction loss. Pairwise ranking loss
dan pseudo-ranking tidak berubah.

## Inference dan metrik

Skor kontinu berada pada rentang kelas asli:

- Regression: inverse transform prediction;
- CORAL: jumlah probabilitas `P(y > k)`;
- CORN: conditional sigmoid logits diubah menjadi cumulative probabilities
  dengan cumulative product, lalu dijumlahkan.

R², Spearman, MAE, dan RMSE memakai skor kontinu. QWK memakai skor yang
dibulatkan dan di-clip ke rentang kelas. Definisi ini sama untuk semua
objective.

## Struktur notebook

Urutan notebook mengikuti acuan:

1. setup;
2. pemuatan dataset;
3. split dan preprocessing;
4. backbone dan objective head;
5. augmentasi;
6. loss dan evaluasi;
7. DataLoader dan inference;
8. penyimpanan hasil;
9. helper visualisasi;
10. eksperimen MLP;
11. eksperimen TabM;
12. eksperimen RealMLP;
13. perbandingan akhir;
14. daftar file output.

Empat skenario eksperimen tetap ditampilkan sebagai cell terpisah per
backbone. Di dalam tiap skenario, loop objective menjalankan Regression,
CORAL, lalu CORN. Pendekatan ini mempertahankan alur notebook acuan tanpa
menduplikasi 36 fungsi training.

## Hasil dan visualisasi

Setiap baris hasil menyimpan:

- `dataset`;
- `backbone`;
- `objective`;
- `experiment`;
- `seed`;
- lima metrik;
- waktu training dan inference;
- hardware.

File output disimpan di `/kaggle/working/rankup_coral_corn_results`:

- `metrics.csv`;
- `predictions.csv`;
- `history.csv`;
- plot metrik, training loss, prediksi, residual, dan t-SNE.

Plot dibuat per dataset dan backbone. Objective menjadi kelompok utama dan
skenario eksperimen menjadi hue/facet agar grafik tetap terbaca. Residual
didefinisikan sebagai `y_score - y_true`.

## Validasi dan penanganan error

Notebook melakukan self-check ringan untuk:

- bentuk CORAL/CORN logits adalah `K-1`;
- extended levels CORAL benar untuk kelas awal, tengah, dan akhir;
- cumulative probabilities CORN monoton tidak meningkat;
- expected ordinal score berada pada rentang `0..K-1`;
- bentuk output TabM mempertahankan dimensi ensemble;
- semua kombinasi dataset, backbone, objective, dan skenario menghasilkan
  satu baris metrik sebelum visualisasi dijalankan.

Dataset tetap wajib numerik, tanpa missing value, dan memiliki kelas berurutan
`0..K-1`. Error ditampilkan sebelum training jika syarat ini tidak terpenuhi.

## Dependensi dan eksekusi

Notebook memasang `tabm` dan `coral-pytorch` melalui `%pip` agar dapat langsung
dijalankan dengan Kaggle **Save & Run All**. Training menggunakan CUDA jika
tersedia. Seperti notebook acuan, satu proses PyTorch menggunakan satu GPU;
dua T4 tidak diparalelkan karena menambah distributed-training complexity tanpa
manfaat jelas untuk model tabular kecil ini.

