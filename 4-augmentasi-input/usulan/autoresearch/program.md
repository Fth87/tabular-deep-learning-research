# Program autoresearch augmentasi RankUp

Adaptasi terbatas dari [karpathy/autoresearch](https://github.com/karpathy/autoresearch): agen mengusulkan satu perubahan pada `candidate.py`, menjalankannya pada sembilan kondisi, memeriksa hasil, dan mencatat keputusan beserta kegagalan. Harness, data, metrik dan budget tetap. Ini bukan menjalankan nanochat upstream dan tidak secara otomatis menjadi agen tanpa proses Codex yang aktif.

- Scope perubahan hanya operator augmentasi, bukan arsitektur, loss atau data.
- 3 dataset × 1%, 10%, 30%; seed 42; QWK dan macro-F1 lebih tinggi lebih baik, MSE lebih rendah lebih baik; validation yang ada untuk screening, tidak membaca test.
- Budget per kondisi: max 120 epoch, 100 steps/epoch, patience 15; weak identity, strong 30%, batch 32, lr .002; maksimal 2 T4 paralel, timeout 600 detik/job.
- Pembanding awal: feature-guided yang ada (`hasil/feature-guided-20261110`), dengan split yang harus cocok berdasarkan SHA-256. Keep bila tidak kalah pada semua kondisi dan ada peningkatan pada sedikitnya satu; selebihnya discard atau perlu pemeriksaan. Ini aturan eksplorasi ketat, bukan bukti statistik. Peringkat sekunder dan setiap sel tetap ditampilkan, bukan merata-rata QWK/F1/MSE mentah.
- Log semua hasil, config, checksum kode, keputusan dan kegagalan dalam `hasil/autoresearch/`. Jangan menimpa atau menghapus kerja sebelumnya. Jika crash, catat sebabnya dan perbaiki.
- Validation sebelumnya dipakai berkali-kali dan test historis sudah dilihat. Metode yang terpilih belum bisa disebut unggul/stabil tanpa split baru, seed berulang dan data holdout baru atau eksternal.
- Gunakan Modal profil `lagi` dan PyTorch di image Modal; jangan instal PyTorch lokal. Jangan mengasumsikan jumlah GPU lain atau menutup pekerjaan Modal milik user.
