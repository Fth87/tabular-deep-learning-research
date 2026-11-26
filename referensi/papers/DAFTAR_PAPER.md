# Tinjauan paper: augmentasi di ruang latent untuk SSL tabular

Dokumen ini merangkum 42 paper di folder `referensi/papers/` (subfolder 1–6), dikelompokkan sesuai subfolder. Daftar lengkap dengan sumbernya ada di `../README.md`; tiga paper awal tentang ordinal ada di `0_ordinal_regresi_ssl/`. Ringkasan dibuat dari teks PDF masing-masing, kecuali dua survei (§3.5 dan §3.6) serta lima paper terbaru (§2.10 sampai §2.14) yang hanya dibaca abstrak, pendahuluan, dan bagian metode yang relevan. Untuk §6, TabMDA dibaca lengkap; paper lainnya dibaca abstrak dan bagian yang relevan saja (ditandai di tiap ringkasan). Angka yang akan dikutip di laporan perlu dicek ulang ke PDF.

Istilah yang dipakai: *latent* adalah representasi hasil encoder, bukan nilai kolom tabel. *View weak* dan *view strong* adalah dua versi data dengan gangguan ringan dan berat; pada FixMatch dan RankUp, prediksi view weak menjadi pseudo-label untuk view strong. *Self-supervised* berarti pre-training tanpa label lalu fine-tune, sedangkan *semi-supervised* berarti data berlabel dan tak berlabel dilatih bersamaan.

---

## 1. Augmentasi latent untuk data tabular

Folder `1_latent_tabular/`.

### 1.1 RaTab

*Representation Space Augmentation for Effective Self-Supervised Learning on Tabular Data*. Eo dkk. (LG AI Research), AAAI 2025.
File: `Eo2025_RaTab_Representation_Space_Augmentation_AAAI.pdf`

Augmentasi di input tabel (masking, shuffling) berisiko merusak pola penting dan menghasilkan baris yang tidak realistis, sehingga RaTab memindahkan augmentasi ke ruang representasi. Yang diganggu adalah bobot, bukan vektor latent. Bobot layer terakhir encoder didekomposisi dengan SVD dan hanya disimpan sekitar 70% rank teratas. Input yang sama di-forward ulang dengan bobot tersebut untuk menghasilkan view kedua, ditambah dropout 0,1. Loss-nya gabungan InfoNCE dan rekonstruksi dengan bobot 0,5 : 0,5, lalu encoder di-fine-tune dengan label.

Pada 13 dataset OpenML, RaTab lebih baik dari masking, shuffling, CutMix, dan Contrastive-mixup. Contrastive-mixup di latent juga umumnya lebih baik dari augmentasi input. Ablasi menunjukkan SVD paling efektif di layer terakhir, dan menerapkannya di banyak layer tidak menambah performa. RaTab adalah prior art terdekat untuk setting self-supervised, sehingga klaim "augmentasi latent pertama untuk tabular" tidak bisa dipakai.

### 1.2 SDAT

*Semi-Supervised Learning with Data Augmentation for Tabular Data*. Fang dkk. (Ant Group), CIKM 2022.
File: `Fang2022_SDAT_SSL_Data_Augmentation_Tabular_CIKM_scan.pdf` (PDF hasil scan, teks dibaca dengan OCR)

SDAT membuat augmentasi untuk semi-supervised tabular di latent sebuah VAE yang encoder-nya dipakai bersama classifier. Untuk tiap sampel, z diambil dari N(μ, σ) lalu didekode menjadi x̂ sebagai sampel augmentasi. Regularisasi β dipakai untuk memperbesar σ. Loss konsistensinya KL antara prediksi x dan x̂, dengan prediksi x diberi stop-gradient sebagai target. VAE dan classifier dilatih bersamaan.

Eksperimen memakai 6 dataset dengan 64 atau 128 label, 5 kali ulangan, dan t-test. SDAT lebih baik dari UDA, VIME, ReMixMatch, MeanTeacher, dan EntMin. Noise Gaussian di input (0,5·std) hampir tidak membantu. VAE yang dilatih terpisah hasilnya setara noise Gaussian, jadi manfaatnya datang dari training bersama. Nilai β terbaik sangat kecil (0,002). SDAT adalah prior art terdekat untuk setting semi-supervised. Desain targetnya (data asli tanpa gangguan) sesuai dengan pilihan weak = identitas di latent. Perbedaannya, SDAT memakai model generatif, hanya untuk klasifikasi, dan tidak mengukur besar gangguan.

### 1.3 Contrastive Mixup

*Contrastive Mixup: Self- and Semi-Supervised Learning for Tabular Domain*. Darabi dkk. (UCLA), arXiv 2021.
File: `Darabi2021_Contrastive_Mixup_tabular_arXiv2108.12296.pdf`

Mixup di input tabel bisa menghasilkan baris yang tidak mungkin ada, jadi mixup dilakukan di latent dengan asumsi manifold. Encoder dilatih dengan rekonstruksi, lalu dua sampel sekelas dicampur di latent untuk membentuk pasangan positif pada supervised contrastive loss. Data tak berlabel diberi label lewat label propagation agar bisa ikut membentuk pasangan. Metode ini bergantung pada label atau pseudo-label kelas.

### 1.4 SAINT

*SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training*. Somepalli dkk., arXiv 2021.
File: `Somepalli2021_SAINT_arXiv2106.01342.pdf`

SAINT adalah Transformer tabular dengan attention antar kolom dan antar baris. Pre-training kontrastifnya membuat view kedua dengan CutMix di input lalu mixup di embedding, dengan loss InfoNCE dan denoising. SAINT menjadi contoh penggabungan augmentasi input dan latent.

### 1.5 Transfer Learning with Deep Tabular Models

*Transfer Learning with Deep Tabular Models*. Levin dkk., ICLR 2023.
File: `Levin2023_Transfer_Learning_Deep_Tabular_ICLR_arXiv2206.15306.pdf`

Pada data diagnosis medis dengan data hilir sedikit, model tabular yang di-pretrain lalu di-fine-tune mengungguli GBDT. Paper ini membandingkan pre-training supervised dan self-supervised (termasuk kontrastif ala SAINT), dan mengusulkan pseudo-feature untuk kolom yang tidak ada di data hulu. Relevan untuk skenario encoder pretrained.

### 1.6 SwitchTab

*SwitchTab: Switched Autoencoders Are Effective Tabular Learners*. Wu dkk., AAAI 2024.
File: `Wu2024_SwitchTab_AAAI_arXiv2401.02013.pdf`

SwitchTab memisahkan latent menjadi fitur *mutual* (dimiliki bersama) dan *salient* (khas tiap sampel) dengan autoencoder asimetris. Untuk sepasang sampel, fitur mutual ditukar dan decoder tetap harus merekonstruksi sampel aslinya. Embedding salient bisa dipakai sebagai fitur tambahan untuk XGBoost atau regresi logistik. Ini contoh manipulasi latent tabular untuk pre-training, bukan untuk consistency training.

### 1.7 Rethinking Data Augmentation for Tabular Data

*Rethinking Data Augmentation for Tabular Data in Deep Learning*. Onishi & Meguro (Meiji University), arXiv 2023.
File: `Onishi2023_Rethinking_Data_Augmentation_Tabular_arXiv2305.10308.pdf`

Paper ini membandingkan augmentasi untuk self-supervised tabular dengan arsitektur dan metode SSL yang disamakan, dan mengusulkan Mask Token Replacement (MTR) untuk Transformer tabular. Temuan yang relevan: Manifold Mixup adalah satu-satunya augmentasi yang berdampak negatif. Karena itu interpolasi di latent diberi prioritas rendah dalam rencana eksperimen.

---

## 2. Perturbasi fitur pada SSL umum

Folder `2_latent_ssl_umum/`. Sebagian besar dari bidang visi komputer.

### 2.1 UniMatch

*Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation*. Yang dkk., CVPR 2023.
File: `Yang2023_UniMatch_feature_perturbation_CVPR_arXiv2208.09910.pdf`

UniMatch menambahkan dua hal ke FixMatch. Pertama, aliran perturbasi fitur: fitur encoder dari gambar weak diberi Dropout(0,5), lalu prediksinya dilatih mengikuti pseudo-label dari gambar weak tanpa gangguan. Kedua, dua view strong yang dibimbing oleh satu view weak. Loss tak berlabel adalah rata-rata dari ketiga aliran. Penulis menunjukkan perturbasi fitur dan augmentasi gambar saling melengkapi. Karena itu augmentasi latent juga diuji sebagai tambahan augmentasi input, dan Dropout 0,5 dipakai sebagai baseline.

### 2.2 IFMatch

*Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning*. Wu & Cui, ECCV 2024.
File: `Wu2024_Image_Feature_Weak_to_Strong_Consistency_ECCV_arXiv2408.12614.pdf`

Gangguan yang hanya di level gambar menyisakan banyak sampel yang sudah diprediksi sangat yakin, sehingga loss-nya hampir nol. IFMatch menambahkan perturbasi fitur dengan intensitas weak dan strong, dibedakan dari posisinya dalam blok residual. Strukturnya tiga cabang: teacher (gambar weak, menghasilkan pseudo-label), student 1 (gambar weak + fitur strong), dan student 2 (gambar strong + fitur weak, hanya untuk sampel yang terlalu mudah). Jenis perturbasinya pergeseran, dropout, dan perubahan nilai, semuanya sample-agnostic. Penulis juga melaporkan bahwa gambar strong digabung fitur strong secara langsung justru menurunkan performa. Ini paper yang paling dekat dengan ide weak/strong di latent dari pembimbing.

### 2.3 Cross-Consistency Training (CCT)

*Semi-Supervised Semantic Segmentation with Cross-Consistency Training*. Ouali dkk., CVPR 2020.
File: `Ouali2020_Cross_Consistency_Training_CVPR_arXiv2003.09005.pdf`

CCT berangkat dari asumsi cluster: wilayah berdensitas rendah lebih terlihat di representasi tersembunyi daripada di input. Output encoder diganggu dan dikirim ke beberapa decoder tambahan, yang prediksinya harus konsisten dengan decoder utama. Perturbasi yang dipakai antara lain F-Noise (z + z·u, u ~ U(−0,3; 0,3)), F-Drop (membuang wilayah dengan aktivasi di atas ambang acak 0,6–0,9), Dropout 0,5, dan VAT. Paper ini menjadi sumber operator latent standar beserta nilai default-nya; F-Noise dan Dropout 0,5 dipakai sebagai baseline.

### 2.4 FeatMatch

*FeatMatch: Feature-Based Augmentation for Semi-Supervised Learning*. Kuo dkk., ECCV 2020.
File: `Kuo2020_FeatMatch_ECCV_arXiv2007.08505.pdf`

FeatMatch menyimpan fitur dari iterasi sebelumnya di memory bank, mengelompokkannya per kelas (berdasarkan pseudo-label) menjadi prototipe, lalu memodifikasi fitur tiap sampel lewat attention ke prototipe intra- dan antar-kelas. Hasilnya dipakai dalam loss konsistensi bersama augmentasi gambar. Metode ini bergantung pada pseudo-label kelas, sehingga berisiko kalau pseudo-label belum akurat.

### 2.5 Manifold Mixup

*Manifold Mixup: Better Representations by Interpolating Hidden States*. Verma dkk., ICML 2019.
File: `Verma2019_Manifold_Mixup_ICML_arXiv1806.05236.pdf`

Pada tiap iterasi dipilih satu layer secara acak. Hidden state dua sampel di layer itu dicampur dengan λ ~ Beta, label dicampur dengan λ yang sama, lalu forward dilanjutkan. Hasilnya batas keputusan lebih halus dan representasi kelas lebih rata. Ini dasar teori augmentasi latent, tapi di tabular hasilnya negatif (lihat §1.7).

### 2.6 Dataset Augmentation in Feature Space

*Dataset Augmentation in Feature Space*. DeVries & Taylor, ICLR 2017 Workshop.
File: `DeVries2017_Dataset_Augmentation_in_Feature_Space_arXiv1702.05538.pdf`

Augmentasi dilakukan di context vector hasil autoencoder sequence-to-sequence dengan tiga operasi: noise, interpolasi ke tetangga sekelas, dan ekstrapolasi c′ = (cⱼ − cₖ)λ + cⱼ dengan λ = 0,5. Ekstrapolasi paling efektif, sedangkan interpolasi diduga mempersempit batas kelas. Kalau operator berbasis pasangan sampel diuji, ekstrapolasi perlu disertakan.

### 2.7 MODALS

*MODALS: Modality-agnostic Automated Data Augmentation in the Latent Space*. Cheung & Yeung (HKUST), ICLR 2021.
File: `Cheung2021_MODALS_Latent_Space_Augmentation_ICLR.pdf`

MODALS memakai empat operasi latent: interpolasi ke sampel sulit sekelas, ekstrapolasi dari pusat kelas z + λ(z − μ_c), noise Gaussian dengan σ per kelas, dan difference transform z + λ(zⱼ − zₖ). Probabilitas dan magnitudo tiap operasi dicari dengan Population Based Augmentation, dan model diberi triplet loss serta adversarial loss agar latent sekelas lebih kompak. Settingnya supervised dan semua operasinya memakai label kelas. Di 5 dataset tabular UCI gain-nya kecil (misalnya Abalone 64,6 → 65,7) tanpa uji statistik. Untuk SSL, versi tanpa label (statistik batch, pasangan acak) lebih aman.

### 2.8 Latent Space Virtual Adversarial Training

*Regularization with Latent Space Virtual Adversarial Training*. Osada dkk., ECCV 2020.
File: `Osada2020_Latent_Space_Virtual_Adversarial_Training_arXiv2011.13181.pdf`

VAT di input hanya bisa menghasilkan gangguan di sekitar titik data. LVAT mencari arah adversarial di latent model generatif (VAE atau Glow), lalu mendekode hasilnya. Sampel adversarial yang dihasilkan lebih bervariasi dan regularisasinya lebih kuat pada SVHN dan CIFAR-10, baik supervised maupun semi-supervised. Ini opsi untuk tahap lanjut, tapi butuh model generatif tambahan.

### 2.9 ISDA

*Implicit Semantic Data Augmentation for Deep Networks*. Wang dkk., NeurIPS 2019.
File: `Wang2019_ISDA_Implicit_Semantic_Data_Augmentation_NeurIPS_arXiv1909.12220.pdf`

ISDA menggeser fitur sepanjang arah semantik yang diestimasi dari kovarians fitur per kelas (diperbarui online). Augmentasi tidak dibuat eksplisit; ISDA meminimalkan batas atas cross-entropy terhadap augmentasi tak hingga, sehingga biayanya hampir nol. Ini menunjukkan arah gangguan latent bisa dipilih secara terstruktur, meskipun versi aslinya butuh label kelas.

### 2.10 SSR-RCUS

*Semi-supervised Regression by Preserving Ranking Relationships between Close Unlabeled Samples*. Li dkk. (Jilin University), AAAI 2026.
File: `Li2026_SSR-RCUS_Ranking_Close_Unlabeled_Samples_AAAI.pdf`

SSR-RCUS adalah pengembangan RankUp yang memakai kode RankUp yang sama, termasuk ambang confidence pseudo-rank 0,95. Pseudo-rank yang yakin pada RankUp biasanya muncul pada pasangan dengan selisih target besar, sehingga informasi ranking antar sampel yang berdekatan tidak terpakai. Untuk itu dibuat sampel interpolasi yang dekat ke salah satu anggota pasangan, dengan λ = max(λ′, 1 − λ′) dan λ′ ~ Beta(α, α). Di bawah asumsi monotonik (target berupa fungsi monoton dari indeks linear fitur), urutan antara sampel interpolasi dan sampel terdekatnya terjamin, sehingga pasangan itu bisa dipakai sebagai target ranking tanpa label.

Interpolasinya dilakukan di ruang latent, z = e(x), dengan manifold mixup, dan ranking predictor menerima pasangan (z_i, z_ij). Loss unsupervised tampaknya menggantikan konsistensi weak/strong RankUp, lalu ditambah loss regresi dan loss ranking pada data berlabel. Eksperimennya memakai UTKFace (ResNet-28-2 tanpa pretrain), Yelp Review (Bert-Small pretrained), dan BVCC (Whisper-base pretrained) dengan 10, 50, 250, dan 2000 data berlabel; hasilnya mengungguli RankUp terutama pada label sedikit. Tidak ada data tabular. Ini paper terdekat dengan pekerjaan kita (augmentasi latent di dalam RankUp), tetapi yang diubah adalah cara membuat pasangan ranking, bukan view weak/strong. Kode ada di github.com/changchunli/SSR-RCUS.

### 2.11 UniMatch V2

*UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation*. Yang, Zhao, dan Zhao, arXiv 2410.10777 (v2, Januari 2025).
File: `Yang2025_UniMatch_V2_arXiv2410.10777.pdf`

Kelanjutan UniMatch untuk segmentasi semantik. Kontribusi utamanya adalah mengganti encoder ResNet dengan ViT pretrained (DINOv2), yang menurut penulis memberi peningkatan lebih besar daripada desain metode. Dua aliran augmentasi V1 (input dan fitur) digabung menjadi satu aliran, dan dropout di level fitur diubah menjadi Complementary Dropout: fitur dibagi menjadi dua himpunan yang saling lengkap untuk dua view strong. Weak-to-strong consistency tetap menjadi dasarnya.

Relevansinya terbatas pada gagasan bahwa dua view strong dengan gangguan fitur yang saling melengkapi bisa dipakai pada pseudo-label yang sama. Domainnya segmentasi gambar dengan encoder berskala besar, dan bagian pretrain-nya mendukung skenario 2 (encoder pretrained) pada rencana kita.

### 2.12 Revisiting Consistency Regularization (FeatDistLoss)

*Revisiting Consistency Regularization for Semi-Supervised Learning*. Fan dkk. (MPI Informatics dan ETH Zurich), IJCV 2023.
File: `Fan2023_Revisiting_Consistency_Regularization_FeatDistLoss_IJCV.pdf`

Paper ini menemukan bahwa mendekatkan fitur dari dua augmentasi yang berbeda memang membantu, tetapi **menjauhkan** fitur (ekuivarians) memberi hasil yang lebih baik lagi. FeatDistLoss menerapkan konsistensi pada classifier dan ekuivarians pada level fitur, dan unggul terutama pada label sedikit dan data tidak seimbang. Augmentasinya tetap di level input; yang ditambahkan adalah loss pada jarak fitur.

Hubungannya dengan kita ada pada ukuran kemiripan fitur antara view weak dan strong, yang berdekatan dengan ide cosine dari pembimbing. Hasil ini menunjukkan bahwa menurunkan kemiripan antar view tidak selalu merugikan. Ini rujukan untuk ablasi, bukan untuk operator latent.

### 2.13 Tied-Augment

*Tied-Augment: Controlling Representation Similarity Improves Data Augmentation*. Kurtulus, Li, Dauphin, dan Cubuk, arXiv 2305.13520.
File: `Kurtulus2023_Tied-Augment_Feature_Similarity_arXiv2305.13520.pdf`

Tied-Augment menjalankan dua view hasil augmentasi melalui model dengan bobot bersama, lalu menambahkan satu suku loss yang mengatur kemiripan fitur kedua view (jarak L2) di samping loss klasifikasi. Dengan suku ini, augmentasi yang tadinya butuh banyak epoch menjadi efektif lebih cepat, termasuk pada FixMatch (Tied-FixMatch). Augmentasinya tetap di level input pada gambar.

Relevansinya sebagai pembanding untuk kontrol kemiripan antara dua view: di sini kemiripan dikendalikan lewat loss, sedangkan pada angular kita mengendalikannya lewat konstruksi vektor. Tidak ada data tabular.

### 2.14 CLFP

*Unified Complementary Learning with Feature Perturbation for Semi-Supervised Learning*. Wang dkk. (Hangzhou Dianzi University), Algorithms 2026.
File: `Wang2026_CLFP_Complementary_Learning_Feature_Perturbation_Algorithms.pdf`

CLFP mempertahankan augmentasi weak dan strong di level gambar, lalu menambah perturbasi di ruang fitur: dua dropout (laju 0,5) yang independen diterapkan pada representasi fitur, ditambah complementary label untuk memanfaatkan sampel yang confidence-nya rendah. Pada CIFAR-10 dengan 4 label per kelas akurasinya naik 2,58 poin dibanding FixMatch, dan pada STL-10 naik 2,59 poin.

Ini contoh hibrida: perturbasi fitur ditambahkan di atas augmentasi input, seperti UniMatch dan IFMatch, dan yang dipakai hanya dropout. Bagi kita posisinya mirip dengan kondisi input + latent.

---

## 3. Augmentasi dan SSL tabular

Folder `3_tabular_ssl_augmentasi/`.

### 3.1 SCARF

*SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption*. Bahri dkk. (Google), ICLR 2022.
File: `Bahri2022_SCARF_ICLR_arXiv2106.15147.pdf`

SCARF membuat view dengan mengganti sebagian fitur yang dipilih acak dengan nilai dari distribusi marginal kolom yang sama, lalu melatih encoder dengan InfoNCE. Diuji pada 69 dataset OpenML-CC18, termasuk setting semi-supervised dan label noise. SCARF sudah dipakai di eksperimen sebelumnya sebagai augmentasi input, dan versi latent-nya masuk rencana Fase 1.

### 3.2 Class-Conditioned and Feature-Correlation Based Augmentation

*Tabular Data Contrastive Learning via Class-Conditioned and Feature-Correlation Based Augmentation*. Cui dkk. (Layer 6 AI), arXiv 2024.
File: `Cui2024_Class_Conditioned_Feature_Correlation_Augmentation_arXiv2404.17489.pdf`

Perbaikan dari SCARF: nilai pengganti diambil hanya dari baris dengan kelas yang sama (pseudo-label untuk data tak berlabel), dan fitur yang dikorupsi dipilih berdasarkan korelasi. Korupsi bersyarat kelas membantu, sedangkan pemilihan berdasarkan korelasi tidak terbukti kuat. Idenya sejajar dengan TabCutMix dan TabCutMix+.

### 3.3 Mutual Information-Guided Corruption

*Mutual Information-Guided Corruption for Improved Self-Supervised Representation Learning in Tabular Data*. Lawson dkk., ICML 2026 Workshop on Foundation Models for Structured Data.
File: `Lawson2026_MI_Guided_Corruption_Tabular_SSL.pdf`

SCARF memperlakukan fitur seolah saling bebas. Paper ini mengelompokkan fitur berdasarkan mutual information (estimator k-NN, lalu hierarchical clustering), kemudian melatih conditional VAE untuk tiap kelompok. Saat membuat view, satu kelompok fitur diganti sekaligus dengan nilai yang digenerate CVAE, dengan kondisi yang sedikit diubah agar hasilnya realistis tapi berbeda. Encoder dilatih dengan InfoNCE seperti SCARF. Augmentasinya di level input, tapi idenya mirip TabCutMix+. Ini paper workshop, jadi bobot buktinya lebih rendah dari main track.

### 3.4 T-JEPA

*T-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data*. Thimonier dkk., ICLR 2025.
File: `Thimonier2024_T-JEPA_augmentation_free_arXiv2410.05016.pdf`

T-JEPA tidak memakai augmentasi. Context encoder melihat sebagian fitur, target encoder (EMA) melihat semua fitur, dan predictor menebak representasi latent fitur yang disembunyikan dengan loss ℓ2. Regularization token ditambahkan untuk mencegah representasi kolaps. Paper ini menegaskan bahwa augmentasi tabular masih menjadi masalah terbuka.

### 3.5 Survei SSL untuk data tabular non-sekuensial

*A Survey on Self-Supervised Learning for Non-Sequential Tabular Data*. Wang dkk., ACML 2024.
File: `Wang2024_Survey_SSL_NonSequential_Tabular_arXiv2402.01204.pdf`

Survei yang membagi SSL tabular menjadi predictive, contrastive, dan hybrid learning, serta membahas transfer antar tabel dan benchmark. Dipakai sebagai peta literatur untuk Bab II.

### 3.6 Survei deep learning untuk data tabular

*Deep Learning within Tabular Data: Foundations, Challenges, Advances and Future Directions*. Ren dkk., arXiv 2025.
File: `Ren2025_Deep_Learning_Tabular_Survey_arXiv2501.03540.pdf`

Survei umum tentang representasi kolom heterogen, augmentasi, pre-training, dan model fondasi tabular. Dipakai sebagai rujukan latar belakang di Bab II.

---

## 4. Backbone, adapter, dan kerangka SSL

Folder `4_backbone_adapter/`.

### 4.1 RankUp

*RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier*. Huang dkk., NeurIPS 2024.
File: `Huang2024_RankUp_NeurIPS.pdf`

FixMatch tidak bisa langsung dipakai untuk regresi karena butuh confidence kelas. RankUp menambahkan Auxiliary Ranking Classifier yang memprediksi, untuk tiap pasangan sampel, mana yang targetnya lebih besar. Karena berbentuk klasifikasi biner, tugas ini bisa dilatih dengan FixMatch. Regression Distribution Alignment (RDA) menyesuaikan distribusi pseudo-label regresi dengan data berlabel. Ini kerangka SSL yang dipakai; yang diubah di eksperimen latent hanya cara membentuk view weak dan strong.

### 4.2 TabM

*TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*. Gorishniy dkk. (Yandex), ICLR 2025.
File: `Gorishniy2025_TabM_ICLR_arXiv2410.24210.pdf`

TabM adalah MLP yang meniru ensemble k MLP (default 32) dengan gaya BatchEnsemble: submodel berbagi hampir semua bobot dan masing-masing hanya punya vektor skala sendiri. Prediksi akhirnya rata-rata k prediksi. Di benchmark besar, TabM menjadi model deep tabular terbaik. Ini encoder yang dipakai; latent yang diganggu adalah output backbone berukuran B × 32 × 512.

### 4.3 AdaptFormer

*AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition*. Chen dkk., NeurIPS 2022.
File: `Chen2022_AdaptFormer_NeurIPS_arXiv2205.13535.pdf`

AdaptFormer menyisipkan modul bottleneck kecil (down-projection, ReLU, up-projection) paralel dengan MLP di setiap blok ViT. Bobot asli dibekukan dan hanya modul ini (< 2% parameter) yang dilatih. Disebut oleh pembimbing, tapi hanya relevan kalau encoder pretrained yang dipakai berbasis Transformer.

### 4.4 XTab

*XTab: Cross-table Pretraining for Tabular Transformers*. Zhu dkk., ICML 2023.
File: `Zhu2023_XTab_pretrained_tabular_ICML_arXiv2305.06090.pdf`

Tiap tabel diberi featurizer dan head sendiri, sedangkan backbone Transformer dipakai bersama dan di-pretrain dengan federated learning pada 84 tabel. Untuk tabel baru, backbone ini dipakai sebagai inisialisasi lalu di-fine-tune. Kandidat encoder pretrained untuk skenario kedua.

---

## 5. Prior art augmentasi sudut

Folder `5_angular_prior_art/`.

### 5.1 Rotation Regularization Without Rotation

*Rotation Regularization Without Rotation*. Kobayashi, ECCV 2022.
File: `Kobayashi2022_Rotation_Regularization_Without_Rotation_ECCV.pdf`

Vektor fitur sebelum classifier diputar ke arah acak dengan sudut α tetap. Kumpulan hasil putarannya membentuk kerucut di sekitar fitur asli, dan melatih classifier agar seluruh kerucut tetap benar memberi margin sudut. Karena rotasi eksplisit di dimensi tinggi mahal, efek rotasi dirumuskan secara statistik langsung di loss. Diuji pada klasifikasi gambar dengan CNN, supervised.

Ini prior art terdekat untuk ide cosine terkontrol dari pembimbing. Perbedaan yang masih bisa diajukan: rotasi dipakai untuk membentuk view weak dan strong dalam consistency training, dengan rentang cosine berbeda untuk keduanya, di RankUp untuk data tabular ordinal, klasifikasi, dan regresi.

---

## 6. Augmentasi dengan tabular foundation model

Folder `6_augmentasi_foundation_model/`. Ditambahkan 26 November 2026, setelah tahap 7 menunjukkan bahwa operator latent acak di embedding TabPFN tidak membantu dan merusak regresi.

### 6.1 TabMDA

*TabMDA: Tabular Manifold Data Augmentation for Any Classifier using Transformers with In-context Subsetting*. Margeloiu, Bazaga, Simidjievski, Liò, Jamnik (Cambridge), ICML 2024 Workshop on In-Context Learning.
File: `Margeloiu2024_TabMDA_ICML_arXiv2406.01805.pdf` (dibaca lengkap)

TabMDA melakukan augmentasi di ruang latent TabPFN. Embedding TabPFN bergantung pada konteks (data berlabel yang diberikan saat fit), jadi baris yang sama menghasilkan embedding berbeda kalau konteksnya berbeda. Teknik *in-context subsetting* (ICS) memanfaatkan sifat ini: untuk tiap baris dibuat K konteks berupa subset data latih yang diambil secara stratified, lalu baris itu di-embed dengan tiap konteks. Hasilnya K versi embedding yang dianggap tetap berlabel sama. Classifier apa pun (KNN, logistic regression, decision tree, random forest, XGBoost) lalu dilatih di embedding yang jumlahnya K kali lipat. Data test di-embed dengan seluruh data latih sebagai konteks.

Eksperimen memakai 6 dataset klasifikasi kecil dengan 20–500 sampel latih, 10 ulangan, dan balanced accuracy. Ukuran konteks 0,5; 0,7; 0,9; dan 1 dari data latih, dengan K = 5, 20, atau 50, dipilih per dataset lewat validation. Peningkatannya sampai 15%, dan variasi antar classifier menjadi jauh lebih kecil. Ada dua catatan penting dari paper ini sendiri. Pertama, rata-rata hanya 3% dari peningkatan itu yang berasal dari ICS; sisanya karena classifier dilatih di ruang TabPFN. Kedua, di Tabel 1 untuk dataset vehicle dan steel, classifier terbaik dengan TabMDA hanya setara dengan TabPFN sendiri (misalnya vehicle 500 sampel: 85,1 vs 84,1). Ukuran konteks 0,7–0,9 lebih baik daripada 0,5 untuk data kecil, karena konteks yang terlalu kecil menambah variance. Ukuran acak U[0,5; 0,99] (gaya TrivialAugment) selalu lebih baik dari konteks penuh, tapi sedikit di bawah ukuran tetap yang di-tuning. Keterbatasannya: hanya klasifikasi supervised, dan model in-context harus tersedia saat inferensi.

TabMDA menunjukkan bahwa augmentasi di latent bisa dibuat dari pengetahuan model itu sendiri, bukan dari noise acak. Paper ini belum menguji ICS untuk semi-supervised (view weak/strong pada data unlabeled) maupun untuk regresi.

### 6.2 TabEBM

*TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models*. Margeloiu, Jiang, Simidjievski, Jamnik, NeurIPS 2024.
File: `Margeloiu2024_TabEBM_NeurIPS_arXiv2409.16118.pdf` (dibaca abstrak dan pendahuluan)

Model generatif tabular yang mempelajari p(x, y) atau p(x | y) mudah overfit pada data kecil, sehingga data sintetisnya sering menurunkan performa classifier. TabEBM mengubah TabPFN menjadi energy-based model yang terpisah untuk tiap kelas, lalu mengambil sampel baris sintetis dari masing-masing. Data sintetis ini dilaporkan konsisten meningkatkan performa klasifikasi, terutama pada dataset kecil, dan lebih baik dari TabPFGen. Augmentasinya di ruang input dan butuh label kelas, jadi hanya berlaku untuk klasifikasi.

### 6.3 TabPFGen

*TabPFGen – Tabular Data Generation with TabPFN*. Ma, Dankar, Stein, Yu, Caterini (Layer 6 AI), arXiv 2024.
File: `Ma2024_TabPFGen_arXiv2406.05216.pdf` (dibaca abstrak)

TabPFGen memakai TabPFN sebagai fungsi energi dan mengambil sampel baris sintetis dengan SGLD, tanpa training tambahan. Dipakai untuk augmentasi, class balancing, dan imputasi. Menurut TabMDA, peningkatan dari augmentasi TabPFGen kecil, dan metode ini mewarisi batas TabPFN (maksimal 10 kelas).

### 6.4 A Closer Look at TabPFN v2

*A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities*. Ye, Liu, Chao, arXiv 2025.
File: `Ye2025_Closer_Look_TabPFN_v2_arXiv2502.17361.pdf` (dibaca abstrak dan §6)

Di TabPFN, baris berlabel (support set) dan baris tanpa label (query) punya peran berbeda: label embedding baris berlabel berasal dari label asli, sedangkan baris query memakai label dummy. Akibatnya embedding keduanya tidak bisa dibandingkan langsung. Paper ini mengusulkan ekstraksi *leave-one-fold-out*: baris berlabel dibagi ke beberapa fold, dan tiap fold di-embed sebagai query dengan fold lain sebagai konteks. Dengan cara ini TabPFN v2 menjadi feature extractor yang menghasilkan ruang hampir linearly separable, dan model linear di atasnya setara dengan prediksi in-context TabPFN sendiri. Paper ini juga menunjukkan TabPFN tetap bekerja dengan token atribut acak, dan mengusulkan strategi divide-and-conquer saat inferensi untuk data berdimensi tinggi, banyak kelas, atau berukuran besar.

### 6.5 TabPFN v2

*Accurate predictions on small data with a tabular foundation model*. Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, Hutter, Nature 637, 319–326 (2025).
File: `Hollmann2025_TabPFN_v2_Nature.pdf` (open access, CC BY; dibaca abstrak)

TabPFN adalah tabular foundation model berbasis Transformer yang di-pretrain pada data sintetis dan memprediksi lewat in-context learning. Pada dataset sampai 10.000 sampel, TabPFN mengungguli semua metode sebelumnya dengan selisih besar dan waktu training jauh lebih singkat. Dalam 2,8 detik, TabPFN mengalahkan ensemble baseline terkuat yang di-tuning selama 4 jam pada setting klasifikasi. Model ini juga mendukung fine-tuning, pembangkitan data, dan density estimation.

### 6.6 TabPFN-2.5

*TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models*. Grinsztajn dkk. (Prior Labs, 26 penulis), arXiv 2025.
File: `Grinsztajn2025_TabPFN_2.5_arXiv2511.08667.pdf` (dibaca abstrak)

Penerus TabPFN v2 untuk dataset sampai 50.000 baris dan 2.000 fitur. Menjadi metode terdepan di TabArena, mengungguli tree-based model yang di-tuning dan setara dengan AutoGluon 1.4. Pada dataset kecil-menengah (≤ 10.000 sampel, 500 fitur), TabPFN-2.5 default menang 100% melawan XGBoost default. Paper ini juga memperkenalkan distillation engine yang mengubah TabPFN-2.5 menjadi MLP atau tree ensemble yang ringkas.

### 6.7 NVIDIA Kumo Tabular

*NVIDIA Kumo Tabular: Pushing the Accuracy–Efficiency Frontier for Tabular Prediction*. Qu dkk. (NVIDIA, 15 penulis), arXiv 2026.
File: `Qu2026_Kumo_Tabular_arXiv2611.23001.pdf` (dibaca abstrak)

Keluarga tabular foundation model untuk klasifikasi dan regresi dalam tiga ukuran (27M, 61M, 214M parameter), di-pretrain hanya dengan tabel sintetis dari Kumo prior. Table encoder memadatkan tiap baris menjadi embedding, lalu Transformer in-context memprediksi baris tanpa label dari baris berlabel. Kumo Tabular-L peringkat pertama di delapan benchmark, termasuk TabArena dan BeyondArena, dan lebih cepat dari runner-up-nya (36× dari LimiX-2 di TabArena, 1,7× dari TabPFN-3.5 di BeyondArena). Bobot dan kode pretraining dirilis dengan lisensi OpenMDW-1.1 dan Apache 2.0.

### 6.8 TabDistill

*TabDistill: Distilling Transformers into Neural Nets for Few-Shot Tabular Classification*. Dissanayake, Dutta (University of Maryland), arXiv 2025.
File: `Dissanayake2025_TabDistill_arXiv2511.05704.pdf` (dibaca abstrak dan metode)

Model Transformer tabular seperti TabPFN kuat di few-shot, tapi besar dan mahal. TabDistill memindahkan pengetahuannya ke MLP kecil dengan cara men-fine-tune Transformer itu sebagai hypernetwork yang menghasilkan bobot MLP dari representasi internalnya. MLP hasil distilasi mengungguli MLP biasa, XGBoost, dan logistic regression dengan data latih yang sama, dan kadang mengungguli model asalnya.

### 6.9 Realistic Evaluation of Deep Semi-Supervised Learning

*Realistic Evaluation of Deep Semi-Supervised Learning Algorithms*. Oliver, Odena, Raffel, Cubuk, Goodfellow (Google Brain), NeurIPS 2018.
File: `Oliver2018_Realistic_Evaluation_Deep_SSL_NeurIPS_arXiv1804.09170.pdf` (dibaca abstrak)

Paper ini mengimplementasikan ulang beberapa metode SSL dalam satu kerangka dan mengujinya di kondisi yang lebih realistis. Temuannya: performa baseline sederhana tanpa data unlabeled sering dilaporkan terlalu rendah, sensitivitas metode SSL terhadap jumlah data berlabel dan unlabeled berbeda-beda, dan performa bisa turun jauh kalau data unlabeled berisi sampel di luar distribusi. Kesimpulannya, manfaat SSL harus diukur terhadap baseline supervised yang kuat.

### 6.10 Well-tuned Simple Nets Excel on Tabular Datasets

*Well-tuned Simple Nets Excel on Tabular Datasets*. Kadra, Lindauer, Hutter, Grabocka, NeurIPS 2021.
File: `Kadra2021_Well_tuned_Simple_Nets_NeurIPS_arXiv2106.11189.pdf`

MLP biasa yang diberi kombinasi regularisasi yang tepat ("regularization cocktail") mengungguli arsitektur deep tabular khusus dan XGBoost. Kombinasi dipilih per dataset dari 13 teknik regularisasi, termasuk satu teknik augmentasi (Mix-Up, Cut-Mix, Cut-Out, atau FGSM adversarial). Kombinasi terbaik berbeda di tiap dataset, jadi augmentasi hanya salah satu regularizer dan pilihan terbaiknya tergantung data.
