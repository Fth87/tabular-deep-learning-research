# Plan: baseline SSR-RCUS, angular distance, L2 norm, dan augmentasi latent lain

**Update 21 Nov:** E1–E3 sudah diimplementasi dengan kode resmi yang di-clone utuh (bukan di-port), lihat [`6-ssr-rcus/`](../../6-ssr-rcus/README.md). Fase A (normalisasi) dan E4 belum.

Disusun 21 November 2026 dari catatan pembimbing (ditempel user), kode resmi SSR-RCUS, dan kode eksperimen sendiri. Belum ada yang dijalankan. Bagian yang saya tebak ditandai **[asumsi]** dan perlu dikonfirmasi ke pembimbing.

Nama "SSR-CR" di catatan saya baca sebagai **SSR-RCUS** (Li dkk., AAAI; satu-satunya "SSR" yang kodenya dibagikan dan dibangun di atas RankUp). Kalau yang dimaksud paper lain, plan bagian 2 berubah.

## 1. Yang sudah dicek

**Kode resmi SSR-RCUS** sudah di-clone untuk dibaca: https://github.com/changchunli/SSR-RCUS (commit `e79e53f`, MIT, basis kode RankUp).

| Temuan | Artinya |
|---|---|
| Hanya ada dataset UTKFace, Yelp, BVCC dan jaringan WRN, BERT, Whisper. **Tidak ada TabM atau data tabular.** | Tidak bisa dijalankan langsung di datasetku. Yang bisa dipakai: algoritmanya, yaitu `semilearn/algorithms/rcus/rcus.py`. |
| Mixup dibuat di `arc_manifold_mixup`: ambil fitur **view strong** `feat`, campur semua pasangan dalam batch (`λ·z_j + (1−λ)·z_i`, λ = max(λ, 1−λ), λ ~ Beta(0,5; 0,5)), lalu lewatkan ke head ARC. | Mixup terjadi di output encoder, sebelum head. Sama posisinya dengan hook augmentasi latent-ku. |
| Target urutan campuran = pseudo-rank pasangan asli (view weak), hanya yang confidence ≥ 0,95. Urutannya dijamin oleh asumsi monotonik: sampel campuran terletak di antara dua sampel asal. | Ini inti SSR-RCUS. |
| **Tidak ada normalisasi L2** di encoder, ARC, maupun head. `feat` dipakai mentah. | Jadi "cek normalisasi" belum ada pembandingnya di kode resmi; harus dibuat sendiri. |
| `reg_loss = sup_loss + ulb_loss_ratio` (bukan dikali loss unlabeled). Baris yang sama ada di kode RankUp resmi. | Suku konsistensi regresi tidak aktif di kode resmi, hanya ARC yang bekerja. Cek apakah loop-ku juga begitu supaya perbandingan adil. |
| Head ARC resmi terpisah (`nn.Linear(feat, 2)`). Di TabM-ku head regresi dan ARC satu `LinearEnsemble` (`d_out=3`: 1 regresi + 2 ARC). | Untuk baseline, **jangan diubah** (permintaan: "yang lain jangan dirubah"). Mixup dilewatkan ke head yang sama, ambil 2 logit ARC. |

Catatan kecil: README repo menulis "AAAI 2025", sedangkan DOI paper-nya AAAI-26 (vol. 40). Di README utama kupakai AAAI 2026; cek sekali lagi sebelum sitasi final.

## 2. Arti catatan pembimbing

**a. "Angular distance minimum 0, maksimum 1"** **[asumsi]**
Saya baca: jarak sudut dinormalkan, d = arccos(cos) / π. d = 0 berarti sama, d = 0,5 tegak lurus, d = 1 berlawanan arah. Ini beda dari cosine distance 1 − cos yang rentangnya 0–2. Konversi dari rentang yang sudah dipakai:

| cos | sudut | d (angular distance) |
|---|---|---:|
| 0,95 | 18,2° | 0,101 |
| 0,8 | 36,9° | 0,205 |
| 0,7 | 45,6° | 0,253 |
| 0,5 | 60,0° | 0,333 |
| 0,4 | 66,4° | 0,369 |
| 0,3 | 72,5° | 0,403 |

Weak 0,7–0,95 menjadi d ≈ 0,10–0,25; strong 0,3–0,5 menjadi d ≈ 0,33–0,40. Karena h setelah ReLU ≥ 0, dua h asli selalu punya d ≤ 0,5. Hanya sampel yang sudah diganggu yang bisa lewat 0,5. Perlu ditanyakan: apakah yang dimaksud memang arccos/π, atau (1 − cos)/2?

**b. "Sebelum di head normalisasi L2 norm"**
Setelah (dan tepat sebelum) head, `h ← h / ‖h‖₂` per vektor, dihitung di sumbu terakhir (512) untuk tiap submodel (bukan lintas 32 submodel dan bukan lintas batch). Efeknya:
- Norma dibuang, head hanya melihat arah. Maka jarak sudut menjadi ukuran gangguan yang sebenarnya bagi head. Sekarang `angular` menjaga norma, jadi head masih melihat norma asli dan ukuran gangguan tidak murni.
- Gaussian, zero-out, dropout, dan mixup mengubah norma. Setelah L2 norm, efek skala itu hilang, jadi hasil operator lama bisa berubah **hanya karena normalisasi**. Inilah alasan perlu diuji.
- Mixup konveks mengecilkan norma; dengan L2 norm sampel campuran kembali ke bola satuan.
- Risiko (NormFace, Wang dkk. 2017): fitur dan bobot yang dinormalkan sulit dilatih tanpa **skala** `s`, karena logit jadi terlalu kecil. Perlu `s` (tetap atau dapat dilatih); nilainya jadi keputusan desain (bagian 4).

Dua posisi normalisasi, keduanya diuji: **N2** = augmentasi dulu, lalu normalisasi, lalu head (sesuai kalimat pembimbing, jadi utama); **N1** = normalisasi dulu, lalu augmentasi di bola satuan, lalu head.

**c. "Cek fungsi normalisasi, ini apakah pengaruh?"** Diuji sebagai ablasi (bagian 3, fase A), bukan diasumsikan.

## 3. Eksperimen

Protokol sama dengan yang sudah ada, supaya hasilnya langsung dibandingkan dengan 945 run sebelumnya: NHANES, Pen Digits, Superconductivity; label 1/10/30%; seed 42, 123, 456; weak identitas dan strong 0,5 (kecuali angular); **tanpa Optuna**; pilih konfigurasi dengan **validation**, test hanya dilaporkan.

Baris pembanding yang **sudah ada** (tidak diulang): `supervised`, `rankup_noaug`, `input_*`, `latent_*`, `angular*`.

| Kode | Isi | Mixup SSR-RCUS | View weak/strong |
|---|---|---|---|
| **E1** | Baseline SSR-RCUS | ya | tanpa augmentasi tambahan **[asumsi]** |
| **E2** | SSR-RCUS + angular | ya | angular di latent (weak cos 0,7–0,95; strong 0,3–0,5), di-grid juga dalam angular distance |
| **E3** | SSR-RCUS + augmentasi sebelumnya | ya | operator input (gaussian, zero_out, scarf, tabcutmix) |
| **E4** | Augmentasi latent lain | tidak | lihat daftar di bawah |

Tiga hal yang perlu dikonfirmasi karena catatan bisa dibaca dua cara:

1. **E1:** di kode resmi, perturbasi ada pada mixup dan pada view strong (augmentasi gambar/teks). Untuk tabular tidak ada augmentasi resmi, jadi E1 = mixup saja. Kalau pembimbing mengharapkan E1 sudah memakai augmentasi lain, itu sama dengan E3.
2. **E2: angular tidak bisa menggantikan mixup di suku ranking SSR-RCUS.** Target ranking sampel campuran sah karena dibuat dari dua sampel (i, j) dan terletak di antaranya. Sampel hasil rotasi acak dari satu sampel tidak punya urutan terhadap sampel mana pun. Jadi "augmentasi diubah jadi angular" saya artikan: angular menggantikan **view weak/strong** (bukan mixup). Kalau pembimbing memaksudkan angular menggantikan mixup, perlu argumen urutan yang baru; saya belum punya.
3. **Normalisasi** bukan eksperimen terpisah; tiap E1–E4 dijalankan **dengan dan tanpa L2 norm**.

**Daftar kandidat E4** (urut prioritas; alasan dari tinjauan literatur di `../../referensi/papers/README.md`):

| Prioritas | Kandidat | Sumber | Catatan |
|---|---|---|---|
| 1 | Complementary dropout (dua view strong, dua himpunan fitur yang saling melengkapi) | UniMatch V2, UniMatch | Murah. Dropout biasa sudah ada. |
| 2 | Ekstrapolasi / difference transform tanpa kelas: `z + λ(z_j − z_k)` dari dua baris acak | MODALS, DeVries | Versi tanpa label kelas, jadi aman dari pseudo-label salah. |
| 3 | Slerp antar dua baris (interpolasi di bola satuan) | turunan dari L2 norm + mixup | Hanya bermakna bersama normalisasi. Tidak ada di literatur yang kubaca; ini ideku. |
| 4 | Gangguan bobot layer terakhir (truncated SVD) | RaTab | Bukan gangguan vektor, tapi prior art terdekat. |
| 5 | VAT di latent | Osada dkk. | Mahal (gradien per langkah). Terakhir, hanya jika sisanya tidak menonjol. |
| Tidak | ISDA, FeatMatch, MODALS versi per kelas | | Butuh label kelas atau memori; di SSL berarti pseudo-label (berisiko). |

Yang sudah diuji dan **tidak diulang**: gaussian, zero-out, SCARF, TabCutMix, dropout, F-noise, angular (cosine).

### Urutan kerja

| Fase | Pekerjaan | Keluar |
|---|---|---|
| **A. Cek normalisasi** | Di kode sendiri: ukur sebaran ‖h‖ per dataset (rata-rata, simpangan baku). Jalankan `rankup_noaug` dan `angular` dengan/tanpa L2 norm (N1 dan N2, `s` tetap vs dapat dilatih), 3 seed × 3 dataset × label 10%. | Keputusan: normalisasi dipakai atau tidak, posisi, nilai `s`. Gerbang untuk fase berikutnya. |
| **B. Port SSR-RCUS** | Tambahkan suku ARC-mixup ke loop TabM (satu fungsi, dari `arc_manifold_mixup`). Uji ekuivalensi: dengan fitur dan head acak yang sama, loss-nya harus sama dengan kode resmi sampai toleransi numerik. | Unit test + E1 jalan 1 epoch. |
| **C. E1, E2, E3** | 27 kondisi per konfigurasi, dengan dan tanpa L2 norm (sesuai hasil A). | Tabel dibanding 945 run lama. |
| **D. E4** | Kandidat prioritas 1–3 dulu; 4–5 hanya jika waktunya ada. | Pemenang latent per dataset. |

**Perkiraan biaya** (kasar): E1 = 1 konfigurasi, E2 = 3 (grid angular), E3 = 4, E4 = ±3, jadi sekitar 11, dikali 2 untuk normalisasi = 22 konfigurasi × 27 = ±600 run. Rata-rata run lama 35 detik, jadi ±6 jam GPU T4 berurutan; run dengan mixup semua pasangan (B×B) akan lebih lambat, dan angka ini belum diukur. Profil Modal tetap `lombaaaa`.

## 4. Keputusan terbuka (perlu jawaban sebelum fase B)

1. Arti "angular distance 0–1": arccos/π? (bagian 2a)
2. E1 = mixup saja, tanpa augmentasi tambahan? (bagian 3)
3. Arti "augmentasi diubah jadi angular": angular sebagai view weak/strong (saya kerjakan), atau sebagai pengganti mixup (butuh argumen urutan)? (bagian 3)
4. Nilai skala `s` setelah L2 norm: usul dimulai dari **dapat dilatih**, diinisialisasi ke rata-rata ‖h‖ awal, dan dicatat. Fase A akan membandingkan dengan `s` tetap.
5. Boleh tidak memakai Optuna lagi (arahan 13 November) untuk semua E1–E4? Saya anggap ya.

## 5. Risiko dan batasan

- 3 seed dan satu split: selisih antar operator di 10–30% label sebelumnya lebih kecil dari simpangan baku antar seed. E1–E4 bisa kembali tidak tegas; jangan klaim keunggulan tanpa seed tambahan pada 2–3 kandidat teratas.
- SSR-RCUS dirancang untuk gambar, teks, audio. Asumsi monotonik (sampel campuran terletak di antara dua sampel) belum terbukti untuk fitur tabular TabM; hasilnya bisa lebih buruk dari RankUp biasa. Itu juga temuan.
- Normalisasi bisa mengubah hasil **semua** baris lama; perbandingan dengan 945 run lama hanya adil untuk kondisi tanpa normalisasi.
- Jangan klaim "pertama" untuk augmentasi latent di SSL tabular (SDAT, RaTab) maupun rotasi fitur bersudut (Kobayashi, ECCV 2022).

## 6. Sumber

- Kode SSR-RCUS: https://github.com/changchunli/SSR-RCUS (MIT)
- Li dkk., SSR-RCUS: `../../referensi/papers/2_latent_ssl_umum/Li2026_SSR-RCUS_Ranking_Close_Unlabeled_Samples_AAAI.pdf`
- NormFace (L2 hypersphere, skala `s`): https://arxiv.org/abs/1704.06369
- CosFace: https://arxiv.org/abs/1801.09414
- FeatDistLoss / CR-Match (kemiripan cosine antar view): https://arxiv.org/abs/2112.05825
