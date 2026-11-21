# Plan eksperimen: augmentasi weak/strong di ruang latent untuk RankUp tabular

Disusun 17 November 2026 dari:
- rapat pembimbing 13 November 2026 (catatan di `rapat/2026-11-13/`; rekaman dan transkrip hanya disimpan lokal);
- paket konteks AI (hanya lokal);
- hasil eksperimen sebelumnya dengan augmentasi di input (`../../4-augmentasi-input/`; ringkasan Optuna 5 seed disalin ke `../hasil/pembanding/`);
- studi literatur (`../../referensi/`).

Status: **Fase 0, 1, 2a (17 Nov, 594 run), 2b, 2c, dan 2e (18 Nov, 351 run) selesai; total 945 run.** Angka lengkap di `hasil/ringkasan.md`; ringkasan temuan di §5a dan §5b.

---

## 1. Arahan pembimbing (rapat 13 November)

| Timestamp | Arahan |
|---|---|
| 01:00–01:35 | Saat ini weak/strong dilakukan di **level dataset** (input). |
| 02:29–03:44 | Input → encoder (TabM / "TabFormer") → **vektor embedding** (contoh 64 / 128 / 256) → head kecil → output (MSE atau cross-entropy). |
| 03:44–04:36 | **Augmentasi dipindah ke level embedding.** Weak = perturbasi kecil, strong = perturbasi besar. Coba operator yang ada (TabCutMix, dll.) apakah cocok. |
| 04:36–05:05 | Setelah embedding ada **MLP kecil** (contoh 128 → 32 → 1 untuk MSE, → jumlah kelas untuk CE). |
| 05:07–06:04 | **Dua skenario:** (1) training from scratch semuanya; (2) encoder pretrained, hanya head yang dilatih (AdaptFormer / MLP biasa; encoder freeze atau fine-tune). |
| 06:48–10:02 | **Ide pembimbing:** kontrol cosine similarity terhadap embedding asli dengan trigonometri dibalik. Weak cos ≈ 0,7–0,95, strong cos ≈ 0,3–0,5. Angka ini untuk **ablasi**. |
| 11:04–11:41 | Survei kata kunci *embedding-level*, *latent*, *feature augmentation* untuk Bab II. |
| 13:07 | **Jangan pakai Optuna dulu.** |
| 13:33–14:00, 14:48 | **Urutan:** operator existing di latent dulu sampai bekerja, baru cosine. |
| 15:11 | **RankUp tetap**, yang lain tetap semuanya. |
| 14:33, 14:48 | Target **paper**, progres **tiap minggu**. |

## 2. Keputusan yang sudah diambil (17 November)

| Hal | Keputusan |
|---|---|
| Kerangka | **Semi-supervised RankUp** (bukan VICReg self-supervised seperti di paket konteks). Yang berubah hanya posisi augmentasi. |
| Weak view | **Dua-duanya diuji:** weak = identitas (sesuai SDAT, UniMatch) dan weak = gangguan latent ringan (sesuai arahan pembimbing). |
| Skenario 2 | "Pretrained" = encoder dengan bobot hasil training di **dataset lain**. Ditunda sampai Skenario 1 selesai. |
| Novelty | Klaim dipersempit: weak/strong dengan **cosine terkontrol** di RankUp tabular (ordinal, klasifikasi, regresi). Bukan "pertama augmentasi latent tabular" (RaTab) atau "pertama di SSL tabular" (SDAT). |
| Modal | GPU T4, profil Modal pribadi. |

**Catatan TabFormer:** model yang disebut pembimbing kemungkinan IBM TabFormer (Padhi dkk., ICASSP 2021). Model ini untuk **deret transaksi** (tiap baris jadi token, dilatih seperti BERT/GPT pada urutan transaksi kartu kredit sintetis). Repositorinya hanya memberi kode dan dataset, **tanpa bobot pretrained**. Jadi tidak cocok untuk data i.i.d. kita dan tidak bisa langsung dipakai sebagai encoder pretrained. Perlu dikonfirmasi ke pembimbing model mana yang dimaksud.

## 3. Pertanyaan riset

- **RQ1.** Apakah memindahkan augmentasi weak/strong dari input ke latent TabM mengubah kinerja RankUp, dengan operator, data, split, seed, dan budget yang sama?
- **RQ2.** Apakah augmentasi sudut (cosine terkontrol) lebih baik dari operator latent existing pada besar gangguan yang setara?
- **RQ3.** Rentang cosine weak/strong mana yang memberi hasil terbaik?
- **RQ4 (Skenario 2).** Apakah encoder pretrained + head lebih baik dari training from scratch?

Hipotesis yang bisa dibantah: dari E7, manfaat RankUp tabular datang dari regularisasi noise, bukan dari sampel realistis. Augmentasi latent menguji ini langsung karena tidak menghasilkan baris tabel sama sekali. Catatan: SDAT justru melaporkan noise Gaussian di input hampir tidak membantu (setting klasifikasi 64 label).

## 4. Desain yang dikunci

| Komponen | Pilihan | Alasan |
|---|---|---|
| Kerangka SSL | RankUp (ordinal, regresi) / RankUp-CLS (Pen Digits), loop lama **tidak diubah** | Rekaman 15:11 |
| Encoder | TabM, latent = output backbone `h ∈ R^(B×32×512)` (habis ReLU, ≥ 0) | Sama dengan E1/E2 |
| Data | NHANES (QWK), Pen Digits (Macro F1), Superconductivity (MSE); label 1/10/30%; split `data-5seed` | Sama dengan E1/E2 |
| Training | lr 0,002, batch 32, 100 langkah/epoch, maks. 120 epoch, patience 15, bobot unlabeled 0,2, cutoff 0,95 | Sama dengan E2 |
| Tuning | Tidak ada Optuna | Rekaman 13:07 |
| Seed | 42, 123, 456 | Temuan E4: variasi seed bisa menutupi beda augmentasi |
| Pemilihan | Rata-rata **validation**; test hanya dilaporkan | Aturan sejak awal |
| Statistik | Peringkat rata-rata, Wilcoxon berpasangan per (dataset, label, seed) | Sama dengan E4 |
| Diagnostik | cos(h, h′) aktual untuk weak dan strong, acceptance pseudo-label, epoch terbaik | Besar gangguan terukur, bukan sekadar angka strength |

**Kesetaraan dengan E2 sudah dicek:** konfigurasi `input_*` di kode baru memberi angka test yang **identik** dengan `usulan.run_one` lama (dicek untuk 3 tugas, 1 epoch, CPU).

## 5. Fase

### Fase 0 — Implementasi dan audit ✔ (17 Nov)

- `kode/latent.py`: 7 operator latent + hook di output backbone, untuk weak dan strong.
- Unit test (`kode/test_latent.py`): shape dan nilai berubah, cos angular = target (galat < 1e-4), norma tetap, vektor nol aman, TabCutMix hanya mengambil dari pseudo-label sama, hanya tensor weak/strong saat training yang diganggu.
- Smoke run 2 epoch × 3 dataset di T4 lulus.

Operator yang dipakai dan alasannya:

| Operator | Versi latent | Catatan |
|---|---|---|
| Gaussian | h + s · std_batch · ε | Ada versi input |
| Zero-out | unit terpilih (proporsi s) → 0 | Ada versi input |
| SCARF | unit terpilih diambil dari baris acak lain di batch | Ada versi input |
| TabCutMix | unit terpilih dari baris dengan pseudo-label sama | Ada versi input; bergantung pseudo-label |
| Dropout | F.dropout(h, 0,5) | Baseline UniMatch/CCT |
| F-Noise | h + h · U(−0,3; 0,3) | Baseline CCT |
| Angular | h′ = ‖h‖(c·v + √(1−c²)·u), c ~ U(c_min, c_max) | Ide pembimbing |

Range-limited dan TabCutMix+ **tidak** dibuat versi latent-nya: bin kuantil dan kelompok korelasi di 512 unit latent tidak punya arti yang jelas dan mahal dihitung. Interpolasi juga tidak dipakai (Onishi: Manifold Mixup berefek negatif di tabular).

### Fase 1 — Operator existing (Skenario 1, from scratch) ✔

Semua kondisi: 3 dataset × 3 label × 3 seed = 27 run per konfigurasi. Strength strong = 0,5 (sama dengan E2), weak latent = 0,1.

| Kelompok | Konfigurasi | Jumlah |
|---|---|---:|
| Tanpa RankUp | `supervised` | 27 |
| Input saja (= E2) | `input_{gaussian, zero_out, scarf, tabcutmix}` | 108 |
| Latent saja, weak = identitas | `latent_{op}` | 108 |
| Latent saja, weak latent ringan | `latent_{op}_weak` | 108 |
| Input + latent di view strong | `both_{op}` | 108 |
| Baseline latent literatur | `latent_dropout`, `latent_fnoise` | 54 |
| Kontrol: RankUp tanpa augmentasi | `rankup_noaug` (weak = strong = data asli) | 27 |
| **Total** | 20 konfigurasi | **540** |

Output (`hasil/ringkasan.md`): skor test per kondisi, peringkat rata-rata validation/test, Wilcoxon input vs latent vs input+latent per operator, dan tabel cosine aktual.

### Fase 2 — Augmentasi angular (ide pembimbing)

| Run | Isi | Jumlah | Status |
|---|---|---:|---|
| 2a | `angular` (weak identitas, strong cos [0,3; 0,5]) dan `angular_weak` (weak cos [0,7; 0,95], strong cos [0,3; 0,5]) | 54 | ✔ 17 Nov |
| 2b | Grid ablasi: weak {identitas, [0,7; 0,95], [0,9; 0,99]} × strong {[0,1; 0,3], [0,3; 0,5], [0,5; 0,7]}; 7 kombinasi baru (`angular[_weak\|_weak90][_s10\|_s50]`) | 189 | ✔ 18 Nov |
| 2c | **Kontrol strength-matched:** `matched_{zero_out, dropout, gaussian, scarf}`, strength dari `matched_strength` (kalibrasi cos Fase 1) agar cos strong ≈ 0,4 seperti `angular`. TabCutMix dan F-Noise tidak bisa mencapai cos 0,4 | 108 | ✔ 18 Nov |
| 2e | Rotasi sebelum ReLU blok terakhir: `angular_prerelu`, `angular_weak_prerelu` | 54 | ✔ 18 Nov |
| 2d | Baseline gaya RaTab (truncated SVD bobot layer terakhir), opsional | 27 | opsional |

**Risiko:** h habis ReLU (≥ 0). Rotasi ke cos 0,3 menghasilkan komponen negatif yang tidak pernah dilihat head. Karena itu diuji varian rotasi sebelum ReLU (2e).

2c dicocokkan ke `angular`, bukan ke `angular_weak_prerelu` yang peringkat validation-nya sedikit lebih baik (5,76 vs 5,89 dari 12, praktis seri): kalibrasi strength hanya berlaku di output backbone, dan keduanya sama-sama cos strong 0,4.

### 5a. Hasil Fase 1 dan 2a (17 November)

Peringkat rata-rata di 27 blok (3 dataset × 3 label × 3 seed), dipilih dari validation. Peringkat di bagian ini dihitung dari 22 konfigurasi saat itu; `ringkasan.md` sekarang memuat 35 konfigurasi, jadi angkanya berbeda, tetapi urutannya sama:

1. **Semua konfigurasi latent (peringkat val 8,9–10,7) di atas semua konfigurasi input (13,5–16,2) dan `supervised` (13,6).** Per operator, latent > input signifikan untuk Gaussian (validation, 85% menang, p = 0,0002) dan TabCutMix (test, 89% menang, p = 0,0006). Zero-out dan SCARF searah tapi tidak signifikan.
2. **Tetapi `rankup_noaug` (RankUp tanpa augmentasi apa pun) juga berada di kelompok yang sama (peringkat val 9,6).** Tidak ada satu pun konfigurasi latent yang signifikan lebih baik dari `rankup_noaug` (semua p > 0,06). Augmentasi input cenderung **lebih buruk** dari tanpa augmentasi (Gaussian input kalah 89% di validation, p = 0,003).
3. Jadi "latent lebih baik dari input" terutama berarti **augmentasi input merugikan, sedangkan gangguan di latent netral**. Gangguan latent belum terbukti menambah manfaat di atas RankUp tanpa augmentasi.
4. **Angular (ide pembimbing) setara dengan operator latent lain dan dengan `rankup_noaug`** (p > 0,4). Weak latent (`*_weak`, `angular_weak`) tidak membantu; TabCutMix weak malah lebih buruk di test (p = 0,0055).
5. **Input + latent (`both_*`) tidak lebih baik dari input saja.**
6. **Cosine aktual:** Gaussian 0,5 → 0,90; F-Noise 0,3 → 0,99; TabCutMix → 0,73; zero-out/dropout 0,5 → 0,705 (= √0,5); SCARF → 0,63; angular → 0,40 (tepat target). Besar gangguan berbeda jauh, tapi kinerjanya mirip. Ini konsisten dengan poin 2.
7. Acceptance pseudo-label tidak kecil (Pen Digits ±0,99, Superconductivity ±0,6, NHANES ±0,35), jadi loss konsistensi memang aktif. Hasil netral bukan karena RankUp "mati".
8. RankUp sendiri membantu di label 1%: NHANES QWK 0,17 (`supervised`) → ±0,23–0,25; Pen Digits F1 0,84 → ±0,88.

Dugaan (belum diuji): pada training end-to-end, backbone bisa menyesuaikan diri sehingga gangguan di output-nya "diserap"; dan 32 submodel TabM sudah memberi keragaman prediksi. Ini alasan kuat untuk menguji gangguan latent pada **encoder yang dibekukan** (two-stage, sesuai ucapan pembimbing 06:48) sebelum menyimpulkan.

### 5b. Hasil Fase 2b, 2c, 2e (18 November)

1. **Tidak ada satu pun dari 11 varian angular yang berbeda signifikan dari `rankup_noaug`** (semua p > 0,2). Rentang cosine dan letak rotasi tidak mengubah kesimpulan §5a poin 4.
2. **Strong cos [0,1; 0,3] (`_s10`) cenderung paling buruk** di antara varian angular (peringkat val 17,7–20,9 dari 35 vs 15,6–16,6 untuk cos 0,4 dan 0,6 di output backbone). Gangguan yang terlalu besar mulai merugikan. [0,3; 0,5] dan [0,5; 0,7] setara.
3. **Weak [0,9; 0,99] tidak lebih baik dari weak [0,7; 0,95] maupun weak identitas.**
4. **Rotasi sebelum ReLU setara dengan rotasi di output backbone** (p > 0,6). Masalah nilai negatif setelah rotasi bukan penyebab angular tidak unggul.
5. **Pada cos yang sama (≈ 0,4), hasilnya berbeda per operator.** `matched_gaussian` (cos 0,43) berperingkat terbaik dari 35 konfigurasi di validation (12,3) dan test (11,3), menang atas `angular` di 74% pasangan (val p = 0,03, test p = 0,012), dan memberi Pen Digits 1% F1 0,893 vs 0,881 (`rankup_noaug`). `matched_zero_out` (cos 0,39) justru paling buruk di antara konfigurasi latent (val 21,1, hampir sama dengan `supervised` 21,5). **Tetapi tidak ada yang lolos Bonferroni (p < 0,0011, 44 pembanding).**
6. Cos aktual 2c dekat target: zero-out/dropout 0,39, Gaussian 0,43, SCARF 0,44.

Kesimpulan sementara: besar gangguan (cosine) saja tidak menentukan hasil. Konstruksi trigonometri angular tidak memberi keunggulan atas operator lama dengan cos yang sama, dan Gaussian latent kuat (s = 2,37) adalah kandidat terbaik untuk dikonfirmasi dengan 5 seed (Fase 5).

### Fase 3 — Head setelah embedding

Sketsa rapat 04:36. Memakai konfigurasi terbaik Fase 1–2: head linear bawaan TabM vs MLP kecil (512 → 32 → output); dimensi embedding 128 vs 512. ±108 run.

### Fase 4 — Encoder pretrained (Skenario 2, ditunda)

1. Cari checkpoint yang dilatih di **dataset lain** dan bisa memberi embedding untuk tabel numerik i.i.d. Kandidat: TabPFN v2, XTab, CARTE. IBM TabFormer tidak cocok (lihat §2).
2. Kalau ada: encoder freeze vs fine-tune; head MLP vs adapter gaya AdaptFormer (hanya untuk backbone Transformer); augmentasi latent terbaik dari Fase 1–2.
3. Kalau tidak ada yang cocok, dicatat sebagai "tidak layak".

### Fase 5 — Konfirmasi dan paper

2–3 konfigurasi terbaik (dari validation) dijalankan 5 seed, uji statistik, notebook visualisasi, bab metode dan hasil. Pembanding novelty di Bab II: RaTab, SDAT, Kobayashi, UniMatch, IFMatch, MODALS (lihat `papers/DAFTAR_PAPER.md`).

## 6. Biaya

Satu epoch ±1 detik di T4, jadi satu run (≤ 120 epoch, early stopping) ±30–60 detik. T4 ±$0,59/jam.

| Fase | Run | Perkiraan biaya |
|---|---:|---|
| 1 + 2a + kontrol | 594 (selesai) | ±$3 (rata-rata 30 detik/run) |
| 2b, 2c, 2e | 351 (selesai) | ±$2 |
| 2d (RaTab, opsional) | 27 | ±$0,2 |
| 3 | ±108 | ±$0,8 |

## 7. Yang sengaja tidak dikerjakan

- Optuna (rekaman 13:07).
- Mengubah loss, cutoff, lr, atau jadwal training. Masalah checkpoint di epoch awal (NHANES 30%, Superconductivity 1%) tetap ada dan dilaporkan, supaya hanya posisi augmentasi yang berubah.
- Varian self-supervised VICReg dari paket konteks.
- 10 dataset tambahan (E5); bisa ditambahkan untuk paper.

Struktur folder dan cara menjalankan ada di `README.md`.
