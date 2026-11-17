---
title: "Catatan Murni Ide Dosen dari Meeting TA/Paper"
source: "GMT20261113-091341_Recording.transcript.vtt dan rekaman meeting 13 November 2026"
scope: "Hanya gagasan dan arahan dosen yang disampaikan dalam meeting, terutama pembahasan metode di awal"
research_paradigm: "Semi-supervised learning (RankUp tetap dipakai)"
status: "Catatan rapat, bukan spesifikasi metode yang sudah diuji"
---

# Ide Dosen untuk TA/Paper: Augmentasi pada Level Embedding

> **Batas dokumen:** Catatan ini hanya merangkum **ide dan arahan dosen yang diucapkan dalam rekaman meeting**. Tidak memuat usulan eksperimen buatan AI, hasil eksperimen lama, rujukan paper tambahan, penilaian kebaruan, atau jadwal eksperimen buatan AI. Kalimat dirapikan dari transkrip otomatis; bagian yang ambigu ditandai, bukan ditebak.

## 1. Inti ide dosen

Dosen mengusulkan agar pendekatan augmentasi *weak* dan *strong* yang sebelumnya diterapkan **langsung pada data tabular** dipindahkan ke **vektor representasi (embedding/latent) keluaran encoder**. Dengan demikian, data tabular terlebih dahulu masuk ke encoder, lalu representasi numeriknya dipecah menjadi dua jalur augmentasi. **[01:00–01:34; 03:05–04:29]**

Dosen menyebut ide ini sebagai arah metode yang ingin dicoba untuk **TA/paper**, tetapi tidak mengatakan bahwa hasil eksperimennya sudah terbukti. **[10:57–11:11; 13:59–14:08]**

## 2. Pendekatan lama yang dijelaskan dosen

- Data memiliki **baris sebagai sampel** dan **kolom sebagai fitur**. **[00:14–00:32]**
- Data masuk ke model, misalnya **MLP/encoder**, kemudian menghasilkan prediksi untuk **klasifikasi atau regresi**. Dosen mencontohkan *cross-entropy* dan *MSE* sebagai loss sesuai jenis tugas. **[00:34–00:56]**
- Pada pendekatan sebelumnya, *weak augmentation* dan *strong augmentation* diterapkan **pada data/fitur input**. Weak sedikit mengubah input, sedangkan strong mengubahnya lebih jauh, misalnya menggunakan random noise atau perubahan nilai fitur. **[01:00–01:34]**

## 3. Perubahan yang diusulkan: augmentasi setelah encoder

Dosen mengusulkan alur berikut. **[02:29–05:03; 06:12–06:35]**

```text
DATA TABULAR
     |
     v
ENCODER / BACKBONE
(contoh yang dibicarakan: MLP, TabM, model tabular berbasis Transformer)
     |
     v
EMBEDDING NUMERIK h
(contoh dimensi dari dosen: 64, 128, atau 256)
     |
     +----------------------------+
     |                            |
     v                            v
WEAK AUGMENTATION          STRONG AUGMENTATION
(pada embedding)            (pada embedding)
     |                            |
     v                            v
WEAK LATENT VIEW           STRONG LATENT VIEW
     |                            |
     +-------------+--------------+
                   |
                   v
       HEAD / BAGIAN PREDIKSI
       (jenisnya disesuaikan tugas)
```

**Makna yang dijelaskan dosen:** model awal berfungsi **mengekstraksi fitur/representasi**. Output encoder berbentuk vektor numerik, sehingga augmentasi berikutnya bekerja pada vektor numerik dan **tidak lagi secara langsung menangani kolom kategorikal dari input asli**. **[06:12–06:35; 10:28–10:39]**

Dosen memberi contoh angka dimensi embedding **64, 128, 256** dan ilustrasi head kecil, misalnya **128 → 32 → 1**, atau output sejumlah kelas. Angka ini merupakan **contoh penjelasan**, bukan keputusan arsitektur final. **[03:10–03:39; 04:36–05:03]**

**Catatan kesetiaan sumber:** Dalam pembicaraan, istilah *head* dan *projection head* digunakan berdekatan dengan contoh output regresi/klasifikasi. Rekaman **tidak menetapkan secara tegas** apakah projection head dan task head adalah modul terpisah. Karena itu, diagram ini tidak mengarang hubungan loss ataupun penggabungan kedua jalur.

## 4. Dua kemungkinan cara menggunakan encoder

Dosen menyebut **dua skenario**. **[05:07–05:30]**

1. **Training from scratch:** encoder dan bagian sesudahnya dilatih dari awal.
2. **Menggunakan model yang sudah dilatih:** gunakan encoder yang memiliki bobot hasil pelatihan sebelumnya, lalu latih bagian head yang diperlukan.

Untuk skenario kedua, dosen menyebut kemungkinan encoder **di-freeze** maupun **di-fine-tune**, dan menyarankan mencoba sesuai kecocokan model. **[05:45–06:04]**

Keberadaan model *pretrained* yang sesuai **belum dipastikan di meeting**; dosen meminta agar dicari. **[01:40–02:27; 12:04–12:17]**

## 5. Pilihan model dan head yang disebut dosen

- **TabM** muncul ketika membicarakan model yang pernah disurvei; dosen lalu membahas kemungkinan encoder MLP dan model tabular berbasis Transformer. **[01:40–01:54; 03:05–03:25]**
- **AdaptFormer** disebut sebagai opsi modul/head yang dapat **disesuaikan dengan pasangan backbone berbasis Transformer**, dalam pembicaraan dosen disebut "TabFormer". **[05:34–06:06; 12:17–12:53]**
- Jika tidak menggunakan pendekatan tersebut, dosen menyebut kemungkinan memakai **MLP biasa** sebagai alternatif head. **[12:55–13:01]**

**Batas kepastian istilah:** Penamaan model pada transkrip otomatis tidak seluruhnya jernih. **TabM, TabFormer, dan AdaptFormer** dicatat sebagai istilah yang muncul/dikonfirmasi dalam konteks percakapan; dokumen ini **tidak menyatakan ketiganya model yang sama atau kompatibel secara otomatis**.

## 6. Gagasan weak/strong berdasarkan cosine similarity

Dosen mengusulkan kemungkinan lain untuk **mengendalikan tingkat perubahan embedding** dengan melihat *cosine similarity* terhadap vektor asli. **[06:43–10:03]**

- Ada **vektor asli**, kemudian dibuat **dua vektor baru**.
- **Weak view** lebih dekat arah vektornya ke embedding asli, sehingga cosine similarity-nya lebih tinggi.
- **Strong view** lebih jauh dari embedding asli, sehingga cosine similarity-nya lebih rendah.
- Dosen memberikan contoh **weak sekitar 0,7–0,95** dan **strong sekitar 0,3–0,5**. Dosen juga mencontohkan weak sekitar 0,8 dan strong sekitar 0,4. **Rentang tersebut contoh eksplorasi, bukan parameter final.** **[07:35–08:01; 08:54–09:50]**
- Dosen menyebut pendekatan **trigonometri/rumus cosinus** sebagai cara untuk membangun vektor baru ketika vektor asal dan target kedekatannya telah ditentukan. **[08:04–10:01]**

```text
                   WEAK VIEW
                  /  lebih dekat
EMBEDDING ASLI --<
                  \  lebih jauh
                   STRONG VIEW

cosine(original, weak) > cosine(original, strong)
```

Dosen secara eksplisit mengatakan rentang strong dan weak perlu dilihat kecocokannya dan menyebutnya sebagai bahan **ablasi**. **[09:34–09:51]**

## 7. Operator augmentasi yang dapat dicoba pada embedding

Dosen tidak membatasi augmentasi hanya ke konstruksi cosinus. Ia juga menyebut untuk memeriksa apakah operator yang telah dikenal, misalnya **TabCutMix** atau **Gaussian augmentation**, cocok diterapkan pada representasi numerik. Kecocokannya **belum dipastikan** di rapat. **[04:20–04:32; 10:13–10:24; 14:48–15:06]**

## 8. Urutan pengerjaan yang diminta dosen

Arahan dosen adalah **memastikan bagian dasar bekerja lebih dulu**, dengan metode augmentasi yang sudah ada, lalu baru mencoba pendekatan trigonometri/cosine similarity. Ia meminta agar bagian yang belum menjadi fokus **tidak diubah dahulu**. **[13:33–14:08; 14:48–15:06]**

Dosen juga berkata **tidak menggunakan Optuna dulu** agar pengerjaan tidak memakan waktu lama. **[13:07–13:14]**

Ini **urutan yang disampaikan di meeting**, bukan roadmap tambahan buatan AI.

## 9. Posisi TA/paper dan metode yang sudah dipakai

- Dosen mengarahkan agar ide ini dikerjakan sebagai **metode untuk paper** dan meminta pencarian literatur dengan kata kunci sejenis **embedding-level augmentation**, **latent-level augmentation**, dan **vector augmentation**, untuk dimasukkan pada tinjauan pustaka/Bab 2. **[10:49–11:47; 14:33–14:40]**
- Saat dosen menyebut ide tersebut sebagai "metode kita yang baru", itu merupakan **pernyataan ide saat diskusi**, bukan pembuktian kebaruan ilmiah. **[13:59–14:08]**
- Menjelang akhir meeting, ketika ditanya apakah RankUp tetap dipakai, dosen menyatakan **"RankUp tetap, yang lainnya tetap semuanya"**. Jadi gagasan augmentasi pada embedding ini ditempatkan dalam **konteks semi-supervised learning/RankUp yang sudah digunakan**, bukan mengubah penelitian menjadi self-supervised learning murni. **[15:10–15:14]**

## 10. Hal yang belum ditetapkan secara tegas dalam meeting

Agar model AI berikutnya tidak menganggap contoh sebagai keputusan final:

- Backbone final: **belum dipilih secara definitif**.
- Keberadaan checkpoint pretrained yang cocok: **belum dipastikan**.
- Head final (AdaptFormer, MLP, atau bentuk lain): **belum diputuskan**.
- Dimensi embedding/head: **hanya contoh**.
- Operator augmentasi latent: **masih pilihan yang dibahas**.
- Rentang cosine weak/strong: **contoh untuk dieksplorasi**.
- Rumus implementasi lengkap, desain loss yang menghubungkan dua view, serta hasil performa: **tidak ditetapkan dalam penjelasan dosen ini**.

---

**Sumber tunggal:** Rekaman Zoom dan transkrip VTT meeting **13 November 2026**, terutama menit **00:14–14:08**; konfirmasi bahwa **RankUp tetap dipakai** pada **15:10–15:14**. Tidak ada sumber luar maupun rancangan eksperimen tambahan yang digunakan.
