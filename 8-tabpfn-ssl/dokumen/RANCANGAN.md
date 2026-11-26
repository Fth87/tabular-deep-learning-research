# Rancangan tahap 8 · Pemilihan metode SSL dan augmentasi tabular

Status: **pipeline dan notebook Kaggle tersedia, hasil tahap 8 menunggu run Kaggle**. Cara menjalankan ada di [README tahap 8](../README.md).

## 1. Ringkasan

Pertanyaan utama adalah **metode SSL mana yang membantu untuk tiap dataset dan jumlah label**, serta **augmentasi mana yang menambah manfaat setelah objektif SSL dibuat tetap**. Tahap 8 membandingkan label-only, distilasi ke TabPFN, dan self-training EMA pada head yang sama. View strong ICS dibuat dengan meng-embed baris yang sama memakai subset data berlabel sebagai konteks TabPFN, mengikuti *in-context subsetting* (ICS) dari TabMDA.

Kontribusi yang diuji adalah pemakaian ICS sebagai view weak/strong pada consistency training untuk klasifikasi, ordinal, dan regresi. Efek ICS diukur terpisah untuk distilasi, self-training, dan augmentasi supervised.

| Tahap | Pertanyaan yang diteruskan |
|---|---|
| [1 · Eksplorasi RankUp](../../1-eksplorasi-rankup/README.md) | Pada Energy Efficiency, bandingkan supervised, RankUp, dan fully supervised untuk melihat apakah data unlabeled membantu regresi. |
| [2 · RankUp ordinal](../../2-rankup-ordinal/README.md) | Pindahkan pertanyaan ke lima dataset ordinal. Bandingkan backbone, objektif regresi/ordinal, kekuatan weak/strong, dan fraksi label. |
| [3 · Benchmark SSL](../../3-pembanding-ssl/README.md) | Perluas pembanding ke SSLAE, VIME, STUNT, GFTab, dan RankUp-CLS. Pada NHANES, VIME memimpin di MAE validation dan RankUp punya hasil Spearman/QWK test lebih tinggi. [Skor tahap 3](../../3-pembanding-ssl/results/TEST_VALIDATION_SCORES.md) |
| [4 · Augmentasi input](../../4-augmentasi-input/RINGKASAN_SEMUA_EKSPERIMEN.md) dan [5 · Augmentasi latent](../../5-augmentasi-latent/README.md) | Tetapkan tiga dataset utama dan uji pengaruh augmentasi input/latent terhadap supervised dan RankUp tanpa augmentasi. |
| [6 · SSR-RCUS](../../6-ssr-rcus/README.md) | Tambahkan pembanding SSL RankUp dan SSR-RCUS dalam pipeline tahap 6. SSR-RCUS tetap kuat di Pen Digits saat label sedikit. |
| [7 · TabPFN](../../7-tabpfn-latent/RINGKASAN_2_5_DAN_3_5.md) | Ukur nilai tambah head dan operator di atas teacher label-only yang kuat. Periksa ranking ordinal dan error regresi selain QWK mentah. |
| [8 · Metode SSL + augmentasi](../README.md) | Pilih per dataset dan fraksi label dengan kontrol yang menahan head, batch, dan target konsistensi tetap saat operator diganti. |

## 2. Kenapa: hasil tahap 7

Tahap 7 kini punya dua grid berpasangan, masing-masing 270 run. Teacher TabPFN 3.5 menaikkan QWK NHANES setelah kalibrasi dari 0,215 / 0,329 / 0,376 ke 0,274 / 0,338 / 0,424 dan menaikkan macro-F1 pada semua 9 pasangan Pen Digits. Di Superconductivity MSE 3.5 membaik pada 10% dan 30%, sementara 2.5 unggul pada 1%. Rotasi kuat tetap menaikkan MSE 3.5 pada ketiga fraksi label. Tahap 8 mengulang kontrol dengan paket `tabpfn==10.2.0` dan bobot 3.5 yang sama. [Perbandingan kedua grid dan kontrol tahap 5–6](../../7-tabpfn-latent/RINGKASAN_2_5_DAN_3_5.md)

- **NHANES:** ulang angular dan SCARF dengan QWK yang dikalibrasi di validation; simpan Spearman dan MSE untuk menilai ranking dan prediksi kontinu.
- **Pen Digits:** ulang SCARF karena memimpin head tahap 7 pada semua fraksi label di validation. Pertahankan pembanding historis SSR-RCUS, terutama `rcus_input_zero_out` pada 1%.
- **Superconductivity:** sertakan TabCutMix yang memimpin validation 3.5 pada 1%, Gaussian sebagai operator ringan, angular sebagai gangguan kuat, serta log 2.5 sebagai benchmark historis pada 1%. Perbandingan internal tahap 8 memakai backbone 3.5 yang sama.

Tahap 7 (`7-tabpfn-latent/`, 270 run, TabPFN 2.5 menurut nama folder) memasang operator latent tahap 5 di head MLP di atas embedding TabPFN, dengan distilasi ke prediksi TabPFN pada data unlabeled. Skor test, rata-rata 3 seed:

| | NHANES 1/10/30% (QWK) | Pen Digits 1/10/30% (macro-F1) | Superconductivity 1/10/30% (MSE) |
|---|---|---|---|
| `tabpfn` saja | 0,089 / 0,216 / 0,320 | 0,838 / 0,948 / 0,973 | 306 / 166 / 122 |
| `head_distill` | 0,143 / 0,255 / 0,328 | 0,839 / 0,947 / 0,969 | 304 / 167 / 124 |
| `head_angular` | 0,237 / 0,298 / 0,371 | 0,842 / 0,947 / 0,970 | 831 / 228 / 150 |
| terbaik tahap 5 (TabM) | 0,280 / 0,275 / 0,319 | 0,893 / 0,953 / 0,976 | 365 / 202 / 146 |

Tiga pelajaran:

1. **Model dasar lebih menentukan daripada augmentasi.** TabPFN saja menurunkan MSE Superconductivity 16–20% dari run terbaik tahap 5. Head di atas embedding (`head_sup`, `head_distill`) hanya menyamai TabPFN, sesuai temuan Ye dkk. (2025) bahwa model linear di embedding TabPFN setara dengan prediksi in-context-nya.
2. **Operator acak yang kuat merusak regresi.** `zero_out`, `scarf`, dan `angular` menaikkan MSE Superconductivity 2–3 kali lipat. Operator ringan (`gaussian`, `dropout`, `fnoise`) setara dengan tanpa augmentasi.
3. **Kenaikan QWK NHANES itu semu.** Spearman `head_angular` sama dengan `head_distill` (0,35 di 10%), sedangkan MSE naik (0,93 → 1,31). Augmentasi kuat hanya membuat prediksi lebih menyebar, dan QWK menghargai sebaran. TabPFN saja dengan batas kelas yang disetel di validation mencapai 0,215 / 0,329 / 0,376, setara atau lebih baik dari `head_angular` tanpa augmentasi apa pun.

## 3. Dasar literatur

Ringkasan lengkap ada di `referensi/papers/DAFTAR_PAPER.md` §6 dan tabel relevansinya di `referensi/papers/README.md` §6.

- **TabMDA** (Margeloiu dkk., ICML 2024 Workshop on In-Context Learning). ICS: baris di-embed K kali dengan subset data latih stratified sebagai konteks, dan hasilnya dianggap berlabel sama. Ukuran konteks 0,7–0,9 lebih baik dari 0,5 pada data kecil. Batasannya untuk kita: rata-rata hanya 3% dari peningkatannya berasal dari ICS, dan classifier terbaiknya hanya setara dengan TabPFN sendiri. **Ekspektasi yang jujur: efek ICS kecil.**
- **A Closer Look at TabPFN v2** (Ye dkk., 2025). Embedding baris berlabel harus diambil *leave-one-fold-out* agar setara dengan embedding baris tanpa label. Aturan ini juga berlaku untuk view ICS pada baris berlabel.
- **Realistic Evaluation of Deep SSL** (Oliver dkk., 2018). Manfaat SSL harus diukur terhadap baseline supervised yang kuat. Di sini baseline-nya `tabpfn` dan `head_sup`.
- **Onishi (2023), SDAT (2022), IFMatch (2024)** dari referensi tahap 5. Augmentasi acak di tabular sensitif terhadap hyperparameter, gangguan latent harus kecil, dan operator yang tidak bergantung pada pseudo-label lebih aman. ICS memenuhi syarat terakhir: konteksnya dari data berlabel asli, bukan dari pseudo-label.
- Pencarian 26 November 2026 untuk "embedding TabPFN + consistency regularization + data unlabeled" tidak menemukan paper yang sama. Pencarian ini tidak menyeluruh, jadi bukan bukti kebaruan.

## 4. Metode

Semua di atas embedding TabPFN yang dibekukan, dengan head MLP yang sama dengan tahap 7 (2 × 256, ReLU, dropout 0,1, AdamW 1e-3, batch berlabel 32 dan unlabeled 224, maksimal 3.000 langkah, early stopping di validation).

- **View weak:** embedding dengan konteks penuh (semua data berlabel), sama dengan tahap 7.
- **View strong (ICS):** embedding dengan konteks berupa subset ρ dari data berlabel, diambil stratified. Untuk klasifikasi stratified per kelas, untuk ordinal per level, dan untuk regresi per desil target. Tiap baris punya K = 4 view yang dihitung sekali dan disimpan di cache, lalu tiap langkah training memilih satu view secara acak.
- **Baris berlabel (varian `ics_sup`, `ics_both`):** view ICS diambil out-of-fold. Untuk baris di fold f, subset konteksnya hanya dari fold lain, jadi baris itu tidak pernah ada di konteksnya sendiri.
- **Target konsistensi, dua pilihan:**
  - *distill*: prediksi TabPFN dengan konteks penuh, sama dengan tahap 7;
  - *match*: self-training dengan EMA head (decay 0,99) pada view weak, dengan stop-gradient. Klasifikasi memakai pseudo-label dengan confidence ≥ 0,95; regresi dan ordinal memakai MSE ke prediksi view weak. EMA memberi target tertunda pada kontrol view identik. Fraksi pseudo-label yang lolos disimpan untuk klasifikasi.
- **Loss:** loss berlabel (CE atau MSE) ditambah 1,0 × loss konsistensi pada view strong.
- **Kekuatan augmentasi = ukuran konteks ρ.** ρ = 1 berarti identitas, dan ρ kecil berarti gangguan besar. Ini menggantikan "strength" operator lama, dan diukur dengan cara yang sama: cosine antara view strong dan view weak dicatat per run.

## 5. Varian

| Nama | Isi | Pertanyaan |
|---|---|---|
| `tabpfn` | Prediksi TabPFN saja | Patokan yang harus dilewati |
| `head_sup` | Head, data berlabel saja | Baseline supervised (Oliver dkk.) |
| `head_distill` | + distilasi ke teacher di unlabeled, tanpa augmentasi | Sumbangan data unlabeled |
| `head_match` | Self-training EMA pada view identik | Efek EMA dan patokan bagi `ics_match_70` |
| `head_gaussian`, `head_dropout`, `head_fnoise`, `head_tabcutmix`, `head_angular`, `head_scarf` | Distilasi yang sama dengan operator latent tahap 7 | Bandingkan ICS dengan operator ringan, operator terpilih, dan rotasi kuat |
| `head_match_gaussian` | Self-training EMA dengan Gaussian | Pembanding augmentasi pada objektif EMA |
| `ics_sup` | ICS ρ = 0,7 pada data berlabel saja | Apakah ICS membantu augmentasi supervised? |
| `ics_distill_90`, `ics_distill_70`, `ics_distill_50` | `head_distill` dengan view strong ICS ρ = 0,9 / 0,7 / 0,5 | **Pertanyaan utama** dan ablasi kekuatan |
| `ics_distill_ta` | ρ ~ U[0,5; 0,99] per view (gaya TrivialAugment, seperti TabMDA) | Bisakah tanpa men-tuning ρ? |
| `ics_match_70` | Self-training EMA dengan view strong ICS | Efek ICS vs `head_match` dan `head_match_gaussian` |
| `ics_both_70` | `ics_sup` + `ics_distill_70` | Augmentasi di data berlabel dan unlabeled sekaligus |

Grid utama berisi 15 varian × 27 grup. Ablasi opsional menambah 63 run dengan ρ = 0,9 dan 0,5 pada 1% dan 10% label, serta ρ acak pada semua fraksi. Kontrol inti memakai sampel batch berpasangan per seed dan pool unlabeled yang sama.

## 6. Protokol evaluasi (perbaikan dari tahap 7)

1. Split, seed (42, 123, 456), metrik utama, dan aturan "pilih di validation, test hanya dilaporkan" sama dengan tahap 5–7.
2. **QWK NHANES dihitung dua kali**: dengan pembulatan ke level terdekat (seperti sebelumnya) dan dengan batas kelas yang disetel di validation (kuantil prediksi validation dicocokkan dengan proporsi kelas validation). QWK terkalibrasi dipakai untuk memilih checkpoint dan membandingkan varian. Spearman dan MSE NHANES juga dilaporkan.
3. **Prediksi validation dan test disimpan** per run, supaya analisis ulang tidak butuh run ulang.
4. Versi paket `tabpfn`, nama file bobot, dan SHA-256 dicatat di setiap baris hasil. Log 3.5 tahap 7 sudah mencatat dua kolom pertama.
5. Kontrol (`tabpfn`, `head_*`) dijalankan ulang di notebook yang sama, supaya perbandingannya berpasangan per seed dan semua prediksinya tersimpan.
6. Untuk setiap sel dataset × fraksi label, pilih baseline supervised, SSL tanpa augmentasi, dan SSL dengan augmentasi berdasarkan rerata validation dari tiga seed. Laporkan test dari metode yang sama. Sel yang belum lengkap tidak dipakai untuk pemilihan.
7. Laporan Kaggle menyertakan pembanding tahap 5–7 dan metode historis terpilih menurut validation. Tabel NHANES lintas tahap memakai QWK mentah di kedua sisi. Spearman dan MSE NHANES tersedia pada ringkasan tahap 7 dan hasil tahap 8. Arsitektur dan protokol historis berbeda, sehingga efek augmentasi diukur dari pasangan terkontrol tahap 8.

## 7. Diagnostik: apakah ICS benar-benar mempertahankan label?

Untuk sampel unlabeled yang dipakai semua varian (maksimum 5.000 per grup secara default), dihitung:

- cosine antara view strong dan view weak;
- perubahan prediksi teacher kalau baris di-embed dengan konteks subset, dibandingkan konteks penuh: |Δ prediksi| untuk regresi dan ordinal, perubahan kelas untuk klasifikasi.

Uji dengan rotasi `angular` memakai **cosine yang dipasangkan per baris dan view**. Embedding weak, ICS, dan angular dibaca oleh probe supervised yang sama, dilatih dari embedding out-of-fold berlabel. Drift teacher diukur terpisah dari prediksi TabPFN pada konteks penuh dan subset. Jika drift ICS lebih kecil, hasilnya mendukung H1 pada probe ini. Label sebenarnya pada baris unlabeled tidak dipakai untuk diagnostik.

## 8. Hipotesis dan kriteria

- **H1 (diagnostik):** pada cosine yang sama, ICS mengubah prediksi probe lebih sedikit dari `angular`.
- **H2 (SSL):** distilasi atau self-training EMA dengan view identik memperbaiki skor validation terhadap `head_sup`. Kandidat terpilih kemudian dibandingkan dengan patokan `tabpfn`.
- **H3 (augmentasi):** ICS memperbaiki validation terhadap kontrol identik pada metode SSL yang sama. Periksa `ics_distill_70` vs `head_distill` dan `ics_match_70` vs `head_match`. Bandingkan dengan Gaussian dan operator tahap 7 untuk melihat apakah perbaikan spesifik pada ICS.
- **H4 (kelangkaan label):** periksa apakah manfaat SSL atau augmentasi paling besar pada 1% label, per tugas. Laporkan perbedaan per seed karena tiga seed hanya memberi bukti deskriptif.

Jika H1 didukung oleh probe sementara H3 tidak didukung oleh skor, laporkan stabilitas prediksi dan hasil SSL sebagai temuan yang berbeda. Drift rendah pada probe saja belum menunjukkan bahwa label sebenarnya dipertahankan.

## 9. Biaya komputasi (perkiraan)

Waktu persiapan TabPFN 3.5 di tahap 7 per grup, dalam detik, adalah NHANES 13 / 16 / 21, Pen Digits 14 / 20 / 35, dan Superconductivity 54 / 111 / 247 pada 1 / 10 / 30% label. Tahap 8 menambah persiapan untuk beberapa set view per grup. [Waktu tahap 7](../../7-tabpfn-latent/RINGKASAN_2_5_DAN_3_5.md)

Tiap set view ICS (K = 4) kira-kira butuh 4 kali biaya embedding satu split. Biaya terbesar ada di Superconductivity 30% (4.501 baris konteks, 10.505 baris unlabeled). Penghematan:

- view ICS untuk data unlabeled dibatasi maksimal 5.000 baris acak (batch training memang mengambil sampel acak);
- ablasi ρ = 0,9 dan 0,5 hanya di 1% dan 10%;
- smoke mencatat waktu persiapan pada Superconductivity 30% dengan konteks penuh dan anggaran query yang diperkecil. Pilot dengan `FULL_ARGS` dapat mengukur satu grup pada anggaran final.

Durasi akhir akan diukur di Kaggle. Cache view dan hasil run bisa dipakai saat sesi dilanjutkan.

## 10. Risiko dan keterbatasan

- Efek ICS di TabMDA kecil (3%), jadi kemungkinan besar selisihnya di bawah noise 3 seed. Kalau H2 terlihat menjanjikan, tambah seed sebelum menarik kesimpulan.
- Dengan 36 label (NHANES 1%), subset ρ = 0,5 hanya 18 baris, dan kelas paling jarang (1 baris) bisa hilang dari konteks. Stratifikasi harus memaksa setiap kelas tetap ada.
- Pen Digits 1% masih menyisakan gap terhadap SSR-RCUS. Teacher 3.5 mencapai 0,873 pada test, sementara `rcus_input_zero_out` mencapai 0,894. Keduanya tiga seed dengan protokol masing-masing.
- TabMDA tidak menguji regresi, jadi stratifikasi per desil untuk regresi adalah pilihan kita sendiri.
- Log tahap 7 versi 3.5 mencatat versi paket dan bobot. Log lama 2.5 memakai identitas dari nama folder. Tahap 8 mencatat versi paket, nama bobot, dan SHA-256.

## 11. Urutan kerja

1. Kalibrasi ulang prediksi teacher tahap 7 yang tersedia di cache dan periksa pembanding. Ini tercatat di ringkasan tahap 7.
2. Buat view ICS dan ulang kontrol di Kaggle. Prediksi validation/test dan diagnostik disimpan oleh pipeline tahap 8.
3. Jalankan smoke tiga grup, termasuk pengukuran waktu Superconductivity 30%.
4. Jalankan grid penuh dan ringkasan dengan QWK terkalibrasi, Spearman, dan MSE.
