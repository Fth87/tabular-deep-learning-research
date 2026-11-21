# Augmentasi weak/strong di ruang latent untuk RankUp pada data tabular

Pada RankUp, view weak dan strong biasanya dibuat dengan mengganggu **data input**. Di sini gangguannya dipindah ke **ruang latent**, yaitu output backbone TabM, lalu dibandingkan dengan augmentasi di input pada tiga dataset tabular (ordinal, klasifikasi, regresi) dengan data berlabel 1%, 10%, dan 30%.

![Arsitektur](gambar/arsitektur/arsitektur_medium.png)

## Hasil singkat

945 run (3 dataset × 3 persentase label × 3 seed, banyak konfigurasi). Angka lengkap ada di [`hasil/ringkasan.md`](hasil/ringkasan.md), dan pembahasannya di [`dokumen/PLAN_EKSPERIMEN_LATENT.md`](dokumen/PLAN_EKSPERIMEN_LATENT.md) bagian 5.

- RankUp membantu terutama saat data berlabel sangat sedikit (label 1%).
- Augmentasi di input cenderung berada di bawah RankUp tanpa augmentasi, sedangkan augmentasi di latent netral atau sedikit di atasnya. Belum terbukti menambah manfaat, dan selisih antar operator lebih kecil dari simpangan baku antar seed.
- Augmentasi sudut dengan cosine terkontrol (ide pembimbing) setara dengan operator latent lain.
- Keterbatasan: 3 seed, satu split data, dan strength augmentasi tidak di-tuning.

## Struktur folder

```
.
├── kode/              kode eksperimen
│   ├── latent.py          operator latent, hook di backbone TabM, run_one
│   ├── grid.py            semua konfigurasi dan kelompoknya
│   ├── modal_latent.py    menjalankan grid di Modal (GPU T4)
│   ├── ringkas.py         membuat hasil/ringkasan.md
│   ├── test_latent.py     unit test
│   ├── visualisasi.ipynb  grafik hasil
│   └── experiments/       loop training RankUp (disalin dari `../4-augmentasi-input/eksperimen/experiments/`, tidak diubah)
├── data/              split data (npz + manifest): label_01, label_10, label_30
├── hasil/             results.jsonl (satu baris per run), ringkasan.md, pembanding/
├── gambar/            diagram (LaTeX, PDF + PNG); lihat gambar/README.md
└── dokumen/           rencana dan catatan rapat
    ├── PLAN_EKSPERIMEN_LATENT.md
    └── rapat/             catatan ide pembimbing (13 November)
```

Rekaman rapat, paket konteks AI, dan log run hanya disimpan lokal (di `../arsip/`) dan tidak ikut diunggah. Paper rujukan (PDF, sumber, dan tinjauannya) ada di [`../referensi/`](../referensi/).

## Alur

```
x_u (unlabeled) ──► augmentasi input? ──► backbone TabM ──► h (B × 32 × 512) ──► augmentasi latent? ──► head
                    (operator lama)                                              (hook, hanya view
                                                                                  weak/strong saat training)
view weak   → prediksi jadi pseudo-label (ranking untuk ordinal/regresi, kelas untuk klasifikasi)
view strong → dilatih mengikuti pseudo-label (loss konsistensi RankUp)
```

Data berlabel, validation, dan test tidak pernah diganggu. Kode baru hanya (1) mengganti fungsi `augment` dan (2) memasang hook di output backbone, sehingga konfigurasi `input_*` memberi hasil identik dengan eksperimen sebelumnya.

## Memulai

Butuh Python 3.10 atau lebih baru.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd kode

PYTHONPATH=. python test_latent.py -v     # unit test (CPU)
python ringkas.py                         # membuat ../hasil/ringkasan.md dari results.jsonl
jupyter lab visualisasi.ipynb             # grafik; jalankan semua sel
```

Menjalankan eksperimen di Modal (perlu akun Modal; ganti `<profil>` atau hapus `MODAL_PROFILE`):

```bash
MODAL_PROFILE=<profil> modal run modal_latent.py                                  # test + smoke run 2 epoch
MODAL_PROFILE=<profil> modal run modal_latent.py --action run --names fase1       # satu kelompok
MODAL_PROFILE=<profil> modal run modal_latent.py --action run --names angular --seeds 42 --percents 10
```

Run yang sudah ada di `hasil/results.jsonl` dilewati, jadi perintah yang sama aman diulang kalau terputus. Untuk grid panjang, jalankan sebagai service (misalnya `systemd-run --user`) supaya tidak ikut mati saat terminal ditutup.

## Nama konfigurasi

Format `{posisi}_{operator}[_weak]`; operator: `gaussian`, `zero_out`, `scarf`, `tabcutmix`.

| Posisi | Arti |
|---|---|
| `supervised` | Tanpa RankUp (bobot unlabeled 0) |
| `rankup_noaug` | RankUp tanpa augmentasi apa pun (kontrol) |
| `input_` | Augmentasi di input |
| `latent_` | Augmentasi di latent; weak = identitas |
| `latent_…_weak` | Augmentasi di latent; weak juga diganggu ringan (0,1) |
| `both_` | View strong diaugmentasi di input lalu di latent |
| `angular`, `angular_weak` | Cosine terkontrol: strong cos ∈ [0,3; 0,5], weak cos ∈ [0,7; 0,95] |
| `angular[_weak\|_weak90][_s10\|_s50]` | Ablasi rentang cosine |
| `angular…_prerelu` | Rotasi sebelum ReLU blok terakhir, bukan di output backbone |
| `matched_{op}` | Operator lama dengan strength yang memberi cos ≈ 0,4, sama dengan `angular` |

## Data

Tiga dataset publik: NHANES (ordinal, metrik QWK), Pen Digits (klasifikasi, Macro F1), dan Superconductivity (regresi, MSE). `data/` berisi split yang sudah jadi (data berlabel, tak berlabel, validation, dan test; seed split 42), hanya file `.npz` dan `manifest.json`.
