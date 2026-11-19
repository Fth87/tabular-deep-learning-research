# Gambar

Diagram dibuat dengan LaTeX (TikZ), jadi tajam di slide maupun di dokumen. Tiap gambar punya tiga file dengan nama sama:
`.tex` (sumber), `.pdf` (vektor, untuk dokumen), `.png` (300 dpi, untuk slide).

Menyusun ulang setelah mengubah `.tex` (perlu `pdflatex` dan `pdftoppm`):

```bash
./bangun.sh                # semua gambar
./bangun.sh cosine_skema   # satu gambar
```

## `arsitektur/` — alur RankUp dengan augmentasi di latent

| Gambar | Isi |
|---|---|
| `arsitektur` | Ringkas: data → encoder → weak/strong → head → loss. Untuk slide |
| `arsitektur_medium` | Sama, dengan keterangan di tiap kotak |
| `arsitektur_detail` | Lengkap: alur training, isi TabM, dan augmentasi angular. Untuk paper |

## `cosine/` — metode cosine dari dosen (V, V1, V2)

Urutan baca yang disarankan:

| Gambar | Isi |
|---|---|
| `cosine_dasar` | Apa itu cosine similarity `c`, dan contoh 2 dimensi |
| `cosine_hitung` | Cara menghitung V1 dari V dan `c`, langkah demi langkah |
| `cosine_skema` | Skema V → weak V1 / strong V2 → Head, tanpa angka |
| `cosine_latent` | Skema yang sama dengan contoh angka |
| `cosine_dosen` | Versi lengkap sesuai papan tulis dosen: alur, langkah, dan geometri |
| `cosine_langkah` | Empat panel (cosine, contoh 2D, kerucut di 512 dimensi, resep). Bagian kerucut hanya ada di sini |

## `paper/` — metode dari paper rujukan

| Gambar | Isi |
|---|---|
| `ssr_rcus` | Metode SSR-RCUS (Li dkk., AAAI 2026), gaya sama dengan `arsitektur_medium` untuk dibandingkan |
