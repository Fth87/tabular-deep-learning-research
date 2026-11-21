# EDA dataset eksperimen

[Notebook EDA](eda_datasets.ipynb) memuat NHANES, Pen Digits, dan Superconductivity. Notebook sudah dijalankan secara lokal; untuk menjalankan ulang dari folder `4-augmentasi-input`:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace eda/eda_datasets.ipynb
```

Di IDE, pilih interpreter `4-augmentasi-input/.venv/bin/python` sebagai kernel. EDA seluruh dataset bersifat deskriptif; pemilihan fitur dan model harus menggunakan train/validation.

| | NHANES | Pen Digits | Superconductivity |
|---|---:|---:|---:|
| Baris | 5.223 | 10.992 | 21.263 |
| Fitur numerik | 30 | 16 | 81 |
| Missing values | 0 | 0 | 0 |
| Baris duplikat lengkap | 0 | 0 | 66 |
| Fitur konstan | 0 | 0 | 0 |

**NHANES:** target memiliki lima level ordinal. Level terjarang hanya memiliki satu contoh berlabel pada split 1% saat ini.

**Pen Digits:** target terdiri atas sepuluh kelas digit. Test resmi dari UCI dipertahankan; validation dipilih dari train resmi.

**Superconductivity:** `critical_temp` adalah target kontinu. Ada 6.093 baris berulang menurut seluruh fitur di luar kemunculan pertama. Split mengelompokkan fitur identik sehingga tidak bercampur antara train pool, validation, dan test. Baris berulang tetap ada; material yang mirip tetapi tidak identik masih bisa tersebar ke partisi berbeda.

Grafik, statistik deskriptif, dan ringkasan JSON disimpan di [`hasil/`](hasil/). Hasil EDA untuk Satimage dan Grid Stability sebelumnya dipertahankan di [`hasil/arsip_dataset_sebelumnya/`](hasil/arsip_dataset_sebelumnya/).
