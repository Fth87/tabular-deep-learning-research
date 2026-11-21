# Tahap 1 — Eksplorasi RankUp (Juni–Juli 2026)

Mempelajari RankUp (Huang dkk., NeurIPS 2024), semi-supervised regression dengan auxiliary ranking classifier, dan mencoba implementasi kecil sebelum masuk ke dataset ordinal di tahap 2. Kode resmi RankUp dipakai sebagai acuan; klonnya ada di `../referensi/Semi-Supervised-Regression` (kode pihak ketiga, tidak ikut git).

| Notebook | Isi |
|---|---|
| `rankup_minimal_energy_efficiency.ipynb` | RankUp versi minimal pada dataset Energy Efficiency (UCI): Supervised, RankUp minimal, dan Fully-Supervised |
| `rankup_energy_efficiency_clean.ipynb` | Versi yang dirapikan dari notebook di atas |
| `inspect_convnextv2_errors.ipynb` | Analisis kesalahan prediksi ConvNeXtV2 (gambar uji dengan label 0 yang diprediksi 2) |

Lanjut ke [`../2-rankup-ordinal/`](../2-rankup-ordinal/).
