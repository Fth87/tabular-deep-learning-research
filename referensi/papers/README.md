> Daftar lengkap semua paper beserta tautan sumbernya ada di [`../README.md`](../README.md). Dokumen ini berisi tabel relevansi dan best practice untuk eksperimen latent.

# Rujukan: augmentasi di ruang latent / embedding untuk SSL (fokus tabular)

Dikumpulkan 17 November 2026. Semua PDF di folder ini sudah dicek: file-nya benar PDF dan judul di halaman pertamanya cocok. RaTab, SDAT, dan MODALS diunduh manual lalu dibaca utuh; SDAT berupa hasil scan sehingga teksnya diambil dengan OCR.

Penjelasan lengkap per paper (judul lengkap, inti, metode) ada di **`DAFTAR_PAPER.md`**.

Penanda di kolom "Bukti":
- ✔ isi dicek langsung dari PDF;
- ◐ hanya dari abstrak atau hasil pencarian.

## 1. Augmentasi latent khusus data tabular (`1_latent_tabular/`)

| Paper | Venue | Apa yang dilakukan di latent | Relevansi untuk kita | Bukti |
|---|---|---|---|---|
| **Eo dkk., RaTab: Representation Space Augmentation for Effective Self-Supervised Learning on Tabular Data** | AAAI 2025 | View augmentasi = forward ulang dengan **bobot layer terakhir encoder diganti versi truncated SVD** (rank 50/60/70% dari penuh; 70% paling sering terbaik) + dropout 0,1 di encoder. Loss InfoNCE + rekonstruksi (λ = 0,5), lalu fine-tune. 13 dataset OpenML (klasifikasi + regresi), backbone MLP / FT-Transformer / T2G-Former | **Prior art terdekat (self-supervised).** Klaim "pertama augmentasi latent tabular" tidak bisa dipakai. Detail berguna: (1) gangguan di **layer terakhir** terbaik, di banyak layer tidak menambah; (2) Contrastive-mixup di latent mengalahkan augmentasi input (masking, shuffling, CutMix); (3) caranya mengganggu **bobot**, bukan langsung vektor z, jadi bisa dijadikan baseline "RaTab-style" di TabM | ✔ |
| Darabi dkk., Contrastive Mixup | arXiv 2021 | Mixup di latent berdimensi rendah (asumsi manifold), karena mixup di input menghasilkan baris tabel yang tidak mungkin ada. Unlabeled dipakai lewat label propagation | Contoh **semi-supervised tabular** dengan augmentasi latent | ✔ |
| Somepalli dkk., SAINT | arXiv 2021 | Pre-training kontrastif: CutMix di input + **mixup di ruang embedding** | Gabungan augmentasi input dan latent | ✔ |
| Levin dkk., Transfer Learning with Deep Tabular Models | ICLR 2023 | Pasangan positif: CutMix di input + Mixup di embedding (mengikuti SAINT) | Sama seperti di atas | ✔ |
| Wu dkk., SwitchTab | AAAI 2024 | Autoencoder asimetris yang memisahkan fitur "mutual" dan "salient" di latent, lalu menukarnya antar pasangan | Contoh manipulasi latent tabular (swap) | ✔ |
| **Onishi, Rethinking Data Augmentation for Tabular Data in Deep Learning** | arXiv 2023 | Membandingkan augmentasi untuk self-supervised tabular | **Temuan penting:** "learning with Manifold Mixup had the only negative effect". Interpolasi di latent tidak otomatis membantu di tabular | ✔ |

## 2. Augmentasi fitur/latent untuk semi-supervised (umum, sebagian besar visi) (`2_latent_ssl_umum/`)

| Paper | Venue | Inti | Pelajaran untuk desain kita | Bukti |
|---|---|---|---|---|
| **Yang dkk., UniMatch** | CVPR 2023 | FixMatch + **aliran tambahan perturbasi fitur**: `feat_fp = Dropout(0.5)(g(x_weak))`. Pseudo-label dari weak tanpa gangguan; perturbasi fitur dan augmentasi input dianggap **saling melengkapi** | Perturbasi latent dipakai sebagai **tambahan**, bukan pengganti augmentasi input. Dropout 0,5 sudah cukup kuat sebagai baseline | ✔ |
| **Wu & Cui, Image-Feature Weak-to-Strong Consistency (IFMatch)** | ECCV 2024 | Perturbasi fitur **weak dan strong di dua posisi berbeda**; strategi "movement", "dropout", "value"; sengaja **sample-agnostic** supaya tidak terkena pseudo-label yang salah; struktur tiga cabang | **Paling dekat dengan arahan pembimbing** (weak/strong di level fitur). Operator latent yang bergantung pseudo-label (TabCutMix latent) berisiko | ✔ |
| Ouali dkk., Cross-Consistency Training (CCT) | CVPR 2020 | Gangguan pada output encoder. F-Noise: z + z·U(−0,3; 0,3). F-Drop: drop dengan ambang ∈ [0,6; 0,9]. Dropout 0,5, VAT. Konsistensi antara decoder utama dan decoder tambahan | Daftar operator latent standar + nilai default yang sudah teruji | ✔ |
| Kuo dkk., FeatMatch | ECCV 2020 | Augmentasi fitur yang dipelajari dari prototipe intra-/antar-kelas (memory bank), dipakai dalam loss konsistensi | Varian yang lebih rumit; bergantung pseudo-label kelas | ✔ |
| Verma dkk., Manifold Mixup | ICML 2019 | Interpolasi hidden state | Dasar teori augmentasi latent; hasilnya buruk di tabular (lihat Onishi) | ✔ |
| DeVries & Taylor, Dataset Augmentation in Feature Space | ICLR-W 2017 | Noise, interpolasi, **ekstrapolasi** antar tetangga di ruang fitur: c′ = (cⱼ − cₖ)λ + cⱼ, λ = 0,5 | Ekstrapolasi patut diuji selain interpolasi | ✔ |
| Osada dkk., Latent Space VAT | arXiv 2020 | VAT di latent (dari model generatif); sampel adversarial lebih fleksibel dibanding di input | Opsi gangguan latent yang adversarial (fase lanjut) | ✔ |
| Wang dkk., ISDA | NeurIPS 2019 | Geser fitur sepanjang arah semantik dari kovarians per kelas | Arah gangguan bisa dipilih secara terstruktur, tidak harus acak | ✔ |
| Cheung & Yeung, MODALS | ICLR 2021 | **Supervised**, bukan SSL. 4 operasi latent: interpolasi ke sampel sulit sekelas, ekstrapolasi menjauhi pusat kelas, noise Gaussian dengan σ per kelas, difference transform z + λ(zⱼ − zₖ) dari dua sampel sekelas. Probabilitas/magnitudo dicari dengan PBA (λ ∈ [0; 0,9]). Ditambah triplet loss + adversarial loss agar latent kompak. Tabular: 5 dataset UCI kecil, MLP 2 layer, latent 128, 3 trial | Semua operasinya **butuh label kelas**; di SSL kita itu berarti pseudo-label (berisiko, lihat IFMatch). Versi tanpa kelas (σ dari batch, difference transform dari dua baris acak) bisa dipakai. Gain di tabular kecil (mis. Abalone 64,6 → 65,7) dan tanpa uji statistik. Augmentasi latent saja memberi +1,6 poin rata-rata; sisanya dari loss tambahan | ✔ |
| **Fang dkk., SDAT** | CIKM 2022 | **Semi-supervised tabular** (klasifikasi). Encoder VAE dipakai bersama classifier; view augmentasi = sampel z ~ N(μ, σ) yang **σ-nya diperbesar** lewat regularisasi β, lalu didekode jadi x̂. Konsistensi KL(SG(h(x)) ‖ h(x̂)): **prediksi data asli tanpa gangguan + stop-gradient** jadi target. 6 dataset (64/128 label), 5 trial + t-test berpasangan | **Prior art semi-supervised terdekat** (letaknya di folder 1_latent_tabular). Temuan relevan: (1) noise Gaussian di **input** (γ = 0,5·std) hampir tidak membantu (UDA, ReMixMatch ≈ DNN); (2) VAE yang dilatih terpisah ≈ noise Gaussian; manfaatnya datang dari **training bersama**, karena latent jadi terkait tugas; (3) β terbaik sangat kecil (0,002): gangguan latent harus kecil. Hanya klasifikasi, tidak ada ordinal/regresi | ✔ (OCR) |
| **Li dkk., SSR-RCUS** | AAAI 2026 | Pengembangan **RankUp** (kode dan ambang 0,95 sama). Mixup di latent z = e(x) untuk membuat pasangan ranking antar sampel yang berdekatan; urutannya terjamin lewat asumsi monotonik. Data: UTKFace, Yelp Review, BVCC (bukan tabular) | Paling dekat dengan pekerjaan kita. Memakai mixup di latent sebagai pengganti konsistensi weak/strong, jadi bisa jadi operator latent tambahan dan pembanding | ✔ |
| Fan dkk., FeatDistLoss | IJCV 2023 | Konsistensi di classifier, **ekuivarians** (menjauhkan jarak fitur) di level fitur; augmentasi tetap di input | Menjauhkan fitur antar view bisa membantu; rujukan ablasi kemiripan weak/strong | ◐ |
| Kurtulus dkk., Tied-Augment | arXiv 2023 | Suku loss kemiripan fitur antar dua view bobot bersama; termasuk Tied-FixMatch (gambar) | Kemiripan dikendalikan lewat loss, bukan lewat konstruksi vektor seperti angular | ◐ |
| Yang dkk., UniMatch V2 | arXiv 2024 | Aliran input dan fitur digabung; Complementary Dropout (dua himpunan fitur yang saling lengkap untuk dua view strong); encoder ViT pretrained | Dua view strong dengan gangguan fitur yang saling melengkapi; mendukung skenario encoder pretrained | ◐ |
| Wang dkk., CLFP | Algorithms 2026 | Augmentasi weak/strong di input + dua dropout fitur (0,5) + complementary label; CIFAR-10 dan STL-10 | Hibrida input + fitur, mirip kondisi input + latent kita | ◐ |

## 3. Augmentasi dan SSL tabular umum (`3_tabular_ssl_augmentasi/`)

| Paper | Venue | Relevansi |
|---|---|---|
| Wang dkk., A Survey on SSL for Non-Sequential Tabular Data | ACML 2024 | Peta metode SSL tabular (prediktif, kontrastif, hibrida). Untuk Bab II |
| Ren dkk., Deep Learning within Tabular Data | arXiv 2025 | Survei 127 paper, termasuk augmentasi |
| Cui dkk., Class-Conditioned and Feature-Correlation Based Augmentation | arXiv 2024 | Korupsi input bersyarat pseudo-label + masker berbasis korelasi; penulis tidak menemukan bukti kuat masker korelasi membantu |
| Lawson dkk., Mutual Information-Guided Corruption | ICML 2026 Workshop | Korupsi input per kelompok fitur ber-MI tinggi, nilai baru dari conditional VAE. Sejalan dengan ide TabCutMix+ |
| Thimonier dkk., T-JEPA | ICLR 2025 | SSL tabular **tanpa augmentasi**: memprediksi representasi latent. Alternatif konseptual |
| Bahri dkk., SCARF | ICLR 2022 | Baseline augmentasi input yang sudah kita pakai |

## 4. Backbone, adapter, dan metode dasar (`4_backbone_adapter/`)

| Paper | Relevansi |
|---|---|
| Huang dkk., RankUp (NeurIPS 2024) | Kerangka SSL yang dipakai |
| Gorishniy dkk., TabM (ICLR 2025) | Encoder saat ini |
| Chen dkk., AdaptFormer (NeurIPS 2022) | Adapter untuk Vision Transformer pretrained. Disebut pembimbing; hanya bermakna untuk backbone Transformer |
| Zhu dkk., XTab (ICML 2023) | Kandidat **encoder tabular pretrained** (Transformer lintas tabel) untuk Skenario 2 |

## 5. Prior art untuk ide augmentasi sudut / cosine (`5_angular_prior_art/`)

| Paper | Inti | Dampak ke klaim |
|---|---|---|
| **Kobayashi, Rotation Regularization Without Rotation** | ECCV 2022. Memutar vektor fitur ke arah acak dengan **sudut α tetap**, membentuk kerucut di sekitar fitur yang memberi margin sudut. Dipakai sebagai regularisasi classifier (supervised) | **Prior art terdekat untuk ide pembimbing.** Rotasi fitur dengan sudut terkontrol sudah ada. Pembeda yang mungkin: (a) dipakai sebagai view **weak/strong** dalam konsistensi SSL, (b) rentang cosine berbeda untuk weak dan strong, (c) di RankUp tabular untuk ordinal/regresi. Wajib disitasi |

Pencarian "angular / cosine-controlled perturbation + consistency + tabular" **tidak** menemukan paper yang identik. Tapi pencarian ini belum exhaustive, jadi ini bukan bukti kebaruan.

## 6. Status unduhan manual

| Paper | Status |
|---|---|
| RaTab (AAAI 2025) | Sudah ada: `1_latent_tabular/Eo2025_RaTab_Representation_Space_Augmentation_AAAI.pdf` |
| SDAT (CIKM 2022) | Sudah ada: `1_latent_tabular/Fang2022_SDAT_SSL_Data_Augmentation_Tabular_CIKM_scan.pdf` (scan, tanpa teks) |
| MODALS (ICLR 2021) | Sudah ada: `2_latent_ssl_umum/Cheung2021_MODALS_Latent_Space_Augmentation_ICLR.pdf` |
| Mutual Information-Guided Corruption (ICML 2026 Workshop) | Sudah ada: `3_tabular_ssl_augmentasi/Lawson2026_MI_Guided_Corruption_Tabular_SSL.pdf`. Augmentasi **input** (kelompok fitur dari mutual information + CVAE), self-supervised |
| SSR-RCUS, UniMatch V2, FeatDistLoss, Tied-Augment, CLFP | Sudah ada di `2_latent_ssl_umum/` (diunduh 18 Nov 2026). Dua file ganda (IFMatch dan TabM) dihapus karena identik dengan yang sudah ada |

---

## 7. Best practice yang bisa diambil (dan dampaknya ke plan)

1. **Augmentasi latent paling sering dipakai sebagai tambahan, bukan pengganti augmentasi input.**
   - UniMatch: aliran fitur + aliran input, keduanya dibimbing satu weak view.
   - IFMatch: perturbasi input dan fitur digabung di setiap cabang.
   - SAINT dan Levin: CutMix input + mixup embedding.

   → Plan perlu kondisi **"input + latent"** selain "latent saja".
2. **Pseudo-label diambil dari weak view tanpa gangguan fitur** (UniMatch). Gangguan fitur hanya ada di cabang yang dilatih.
   → "Weak = identitas di latent" adalah default yang didukung literatur. Weak latent ringan (arahan pembimbing) tetap diuji sebagai ablasi.
3. **Gunakan operator yang tidak bergantung pseudo-label** (IFMatch). Operator berbasis kelas bisa terdorong pseudo-label yang salah.
   → Prioritaskan noise, dropout, dan rotasi sudut. TabCutMix versi latent (butuh pseudo-label) menjadi pembanding saja.
4. **Operator latent standar beserta nilai defaultnya sudah ada:**
   - Dropout 0,5 (UniMatch, CCT);
   - noise multiplikatif z + z·U(−0,3; 0,3) (CCT F-Noise);
   - F-Drop (CCT);
   - noise Gaussian, interpolasi/ekstrapolasi (DeVries, MODALS);
   - VAT latent (Osada).

   → Masukkan **dropout 0,5 dan F-Noise** sebagai baseline latent, karena keduanya sudah teruji di literatur.
5. **Interpolasi/mixup di latent belum tentu membantu di tabular** (Onishi: satu-satunya yang berefek negatif).
   → Prioritas rendah. Kalau diuji, sertakan juga ekstrapolasi (DeVries).
6. **Variasi intensitas dan bentuk perturbasi fitur** (IFMatch) mendukung ablasi rentang cosine weak/strong dari pembimbing.
7. **Novelty harus dibatasi:**
   - RaTab: augmentasi representasi tabular (self-supervised);
   - SDAT: augmentasi latent untuk **semi-supervised** tabular (klasifikasi) sudah ada, jadi klaim "pertama augmentasi latent di SSL tabular" juga tidak bisa;
   - Kobayashi: rotasi fitur dengan sudut tetap;
   - UniMatch/IFMatch: weak-to-strong di level fitur.

   Kontribusi yang masih mungkin: **augmentasi sudut dengan cosine terkontrol sebagai view weak/strong di RankUp tabular (ordinal, klasifikasi, regresi)**, dibandingkan dengan operator latent standar pada besar gangguan setara. Pembeda dari SDAT: tidak ada model generatif, gangguan langsung di latent backbone, besar gangguan diukur dengan cosine, dan diuji untuk ordinal/regresi.
8. **Gangguan latent sebaiknya kecil dan terkait tugas** (SDAT: β terbaik 0,002; VAE terpisah ≈ noise Gaussian). Gangguan di latent backbone yang dilatih bersama RankUp memang sudah "terkait tugas". Ini mendukung pengukuran cos(h, h′) aktual dan grid strength yang mencakup nilai kecil.
9. **Lokasi gangguan: layer terakhir encoder** (RaTab, ablasi layer). Ini cocok dengan pilihan kita (output backbone TabM).
10. **Operator dari MODALS yang aman tanpa label:** ekstrapolasi menjauhi rata-rata batch, difference transform z + λ(zⱼ − zₖ) dari dua baris acak, noise Gaussian dengan σ dari batch (sudah ada sebagai `gaussian`). Versi per kelas hanya untuk pembanding.
11. **Mixup di latent sudah dipakai di dalam RankUp** (SSR-RCUS, AAAI 2026, pada gambar, teks, dan audio). Penulisnya mengganti konsistensi weak/strong dengan ranking pada pasangan mixup di latent.
   → Mixup antar dua baris di latent bisa dijadikan operator pembanding. Berbeda dari butir 5, di sini mixup dipakai untuk membuat pasangan ranking, bukan sebagai view strong.
12. **Menjauhkan fitur antar view bisa membantu** (FeatDistLoss). Ini memberi alasan untuk mengujinya (bukan bukti) bahwa view strong dengan cosine rendah (0,3–0,5) tidak selalu merugikan; di FeatDistLoss augmentasinya di input, bukan di latent.
