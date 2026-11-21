# Usulan: rotasi adversarial di latent (closed-form, sadar-head)

Status: **hipotesis, sudah diimplementasi (`kode/latent.py`, grup `fase3` di `kode/grid.py`), belum ada hasil.** Belum ada angka yang membuktikan metode ini lebih baik.

Rentang d di grid: `advrot_05` = U(0,025, 0,075), `advrot_10` = U(0,05, 0,15), `advrot_20` = U(0,10, 0,30). Setiap run baru juga mencatat `cos.shift_strong`, yaitu ‖Δ output head‖ / ‖output head‖, untuk diagnostik §1a.

Menjalankan (dari `kode/`):

```bash
MODAL_PROFILE=lombaaaa modal run modal_latent.py                                                 # unit test + smoke 2 epoch
MODAL_PROFILE=lombaaaa modal run modal_latent.py --action run --names fase3 --seeds 42 --percents 1   # pilot 15 run
MODAL_PROFILE=lombaaaa modal run modal_latent.py --action run --names fase3                       # grid penuh 135 run
```

## 1. Kenapa semua operator latent hasilnya mirip (diagnosis dari 945 run)

Di `hasil/results.jsonl`, semua operator latent selisihnya lebih kecil dari std antar seed. Ada dua penyebab yang terlihat di data.

**a. Gangguan acak hampir tidak terlihat oleh head.**
- Head TabM linear. Skor ranking ARC sampel i adalah `r_i = a · h_i`, dengan `a = W[:,2] − W[:,1]`. Peluang i > j hanya bergantung pada `r_i − r_j`, jadi ARC hanya "melihat" **1 arah** dari 512 dimensi h. Pen Digits melihat 10 arah.
- Arah acak di 512 dimensi rata-rata punya |cos| dengan `a` sebesar **0,035** (simulasi numpy, 100k sampel). Proyeksinya ke ruang 10 arah rata-rata 0,14.
- Artinya, rotasi acak 60° hanya menggeser skor ranking sekitar 3,5% dari rotasi 60° yang diarahkan ke `a`. Sebagian besar gangguan acak (Gaussian, dropout, angular) jatuh di ruang yang diabaikan head.
- Data mendukung ini: di Pen Digits, acceptance pseudo-label tetap 99% untuk semua operator latent, sedangkan input SCARF menurunkannya ke 77%. View strong hampir sama dengan view weak, jadi loss konsistensinya hampir nol.

**b. NHANES 10/30% dan Superconductivity 1% overfit sangat cepat.**
- Median best epoch-nya 2–3.
- Di NHANES 30%, acceptance hanya 8%: dengan cutoff 0,95, hampir semua data unlabeled dibuang.
- Masalah ini bukan masalah augmentasi, jadi operator apa pun tidak akan banyak membantu di situ (lihat §5).

## 2. Metode

Satu langkah di hook backbone, hanya untuk view strong. View weak tetap identity.

1. `v = h/‖h‖` per submodel TabM, h berukuran (B, 32, 512).
2. Hitung arah yang **paling mengubah output head**, dengan bobot head yang di-detach:
   - **Ordinal/regresi (ARC):** `r_i = a_k·h_i`. Arah `g_i = −sign(r_i − median_batch(r)) · a_k`. Setiap skor didorong ke median batch, sehingga urutan antar sampel jadi paling sulit dibedakan.
   - **Klasifikasi:** `g_i = W[:,c2] − W[:,c1]`, dengan c1 dan c2 kelas tertinggi pertama dan kedua dari logit h_i sendiri. Arahnya menuju batas kelas terdekat.
3. Proyeksikan ke bidang tangen supaya norma tetap: `u = g − (g·v)v`, lalu normalkan.
4. Rotasi `h' = ‖h‖ (cos θ · v + sin θ · u)` dengan `θ = π·d`, `d ~ U(d_lo, d_hi)`. Di sini **d adalah angular distance 0–1** seperti catatan pembimbing.

Biayanya satu perkalian matriks kecil. Tidak ada backward tambahan, karena untuk head linear arah terburuk orde-1 bisa dihitung langsung (closed form). Sebagai pembanding, VAT butuh power iteration.

**Kontrol penting (ablation):** `headrand`, yaitu langkah yang sama tetapi tanda `g` diacak, bukan menuju median. Kontrol ini memisahkan dua efek, yaitu "sadar-head" (gangguan masuk ke ruang yang dilihat head) dan "adversarial" (gangguan diarahkan ke sisi yang paling sulit).

## 3. Posisi terhadap literatur (jangan klaim "pertama")

Gangguan adversarial di ruang fitur sudah ada, jadi metode ini **varian**, bukan ide baru dari nol:
- VAT (Miyato dkk.) di input.
- CCT (Ouali dkk., CVPR 2020) memakai VAT di output encoder.
- LVAT (Osada dkk., ECCV 2020) memakai VAT di latent VAE/Glow.

Pembedanya:
1. Arahnya closed form dari head linear, tanpa power iteration.
2. Bentuknya rotasi di hypersphere dengan budget angular distance 0–1.
3. Targetnya objektif ranking ARC (RankUp).

Saya belum menemukan paper yang menggabungkan ketiganya. Pencarian ini tidak menyeluruh.

## 4. Eksperimen (tanpa Optuna, protokol sama dengan 945 run)

| Config | d strong | Catatan |
|---|---|---|
| `advrot_05`, `advrot_10`, `advrot_20` | 0,05 / 0,10 / 0,20 (9° / 18° / 36°) | lebih kecil dari angular acak (0,33–0,40), karena per derajat jauh lebih merusak |
| `headrand_10`, `headrand_20` | 0,10 / 0,20 | kontrol tanda acak |
| pembanding | | `rankup_noaug`, `angular`, `matched_gaussian` (peringkat validation terbaik saat ini), sudah ada di results.jsonl |

- **Pilot dulu:** seed 42, label 1%, 3 dataset, 5 config, total 15 run. Perkiraan sekitar 10 menit di T4 (rata-rata run lama 35 detik).
- **Grid penuh:** jalan hanya kalau pilot tidak lebih buruk dari `rankup_noaug` di validation. Ukurannya 5 config × 27 = 135 run, sekitar 1,3 jam GPU berurutan.
- **Pemilihan d:** dari validation saja.
- **Diagnostik yang dicatat:**
  - acceptance rate;
  - `cos` (sudah ada);
  - pergeseran skor `|Δr|` dibanding operator acak, untuk membuktikan klaim §1a.

**Varian kalau pilot bagus:**
- Gabungkan dengan L2 norm sebelum head (N2 di `PLAN_SSR_RCUS.md`). Pada fitur ternormalisasi, rotasi adalah satu-satunya perubahan yang dilihat head, jadi kedua ide pembimbing jadi satu kerangka.
- Pakai `advrot` sebagai pengganti augmentasi pada E2 SSR-RCUS.

## 5. Yang lebih mungkin jadi penentu, tetapi bukan augmentasi

- **NHANES 30% (acceptance 8%):** threshold tetap 0,95 membuang hampir semua data unlabeled. Kandidatnya **SoftMatch** (bobot per sampel dari kurva Gaussian terpotong) atau **FreeMatch** (threshold adaptif), lalu dicek di validation.
- **Overfit di epoch 2–3:** augmentasi tidak memperbaiki ini.
- Ini perlu ditanyakan ke pembimbing: apakah boleh di luar scope "augmentasi"?

## 6. Risiko

- Arah adversarial bisa benar-benar membalik ranking sebenarnya. Kalau d terlalu besar, targetnya jadi salah dan hasilnya lebih buruk dari `rankup_noaug`. Karena itu d dimulai kecil.
- Model bisa "kabur" dengan mengecilkan ‖a‖ atau menyelaraskan h dengan a, sehingga rotasi kehilangan efek. Ini perlu dipantau lewat |Δr| per epoch.
- Nama atribut head di paket `tabm` (`model.output`, dan bentuk bobot `(k, d_in, d_out)`) belum dicek, karena torch dan tabm tidak terpasang di venv lokal.
- Dengan 3 seed dan 1 split, selisih kecil tidak bisa disebut unggul. Perlu uji berpasangan per blok (dataset × label × seed).

## Sumber

- Miyato dkk., VAT: https://arxiv.org/abs/1704.03976
- Ouali dkk., CCT: https://arxiv.org/abs/2003.09005
- Osada dkk., LVAT: https://arxiv.org/abs/2011.13181
- Yang dkk., UniMatch V2: https://arxiv.org/abs/2410.10777
- Chen dkk., SoftMatch: https://arxiv.org/abs/2301.10921
- Wang dkk., FreeMatch: https://arxiv.org/abs/2205.07246
- Li dkk., SSR-RCUS: https://ojs.aaai.org/index.php/AAAI/article/view/39487
- Eo dkk., RaTab: https://ojs.aaai.org/index.php/AAAI/article/view/33265
