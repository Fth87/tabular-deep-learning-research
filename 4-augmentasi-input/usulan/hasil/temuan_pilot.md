# Temuan pilot bounded local transport

Pilot selesai dengan 81/81 job sukses pada 9 kondisi (3 dataset × 1%, 10%, 30% label), masing-masing dengan 9 metode, satu seed 42, tanpa Optuna, dan tanpa membuka data test. Seluruh metode pada tiap kondisi memakai checksum split input yang sama. Nilai di bawah adalah **validation**, dipilih pada epoch terbaik masing-masing; QWK dan macro-F1 lebih tinggi lebih baik, MSE lebih rendah lebih baik. Satu seed tidak cukup untuk menilai stabilitas statistik.

| Dataset | Label | Kandidat | Peringkat dari 9 | Terbaik pada kondisi ini |
|---|---:|---:|---:|---|
| NHANES (QWK) | 1% | 0,151439 | 7 | SCARF 0,241357 |
| NHANES (QWK) | 10% | 0,291023 | 9 | TabCutMix 0,313613 |
| NHANES (QWK) | 30% | 0,323800 | 1 | Kandidat 0,323800 |
| Pen Digits (macro-F1) | 1% | 0,908338 | 7 | Zero-out 0,926889 |
| Pen Digits (macro-F1) | 10% | 0,985359 | 5 | RLA 0,988700 |
| Pen Digits (macro-F1) | 30% | 0,995349 | 1 | Kandidat 0,995349 |
| Superconductivity (MSE) | 1% | 340,382400 | 2 | RLA 340,258836 |
| Superconductivity (MSE) | 10% | 200,974790 | 1 | Kandidat 200,974790 |
| Superconductivity (MSE) | 30% | 192,024778 | 3 | Gaussian 191,639323 |

Kandidat terbaik pada 3/9 kondisi dan melampaui kontrol SSL tanpa augmentasi pada 6/9 kondisi. Rerata peringkat kandidat 4,00/9; Gaussian 3,89/9. Peringkat dan selisih kecil, misalnya 0,995349 vs 0,995347 pada Pen Digits 30%, tidak menunjukkan perbedaan yang meyakinkan. Pemilihan epoch dan pelaporan kondisi terbaik menggunakan validation yang sama membuat angka ini bersifat eksplorasi; tidak ada dasar untuk menyimpulkan kandidat paling stabil atau unggul secara umum. Selisih strength antaroperator tidak setara secara semantik, sehingga ini bukan benchmark final yang dikalibrasi.

Hasil lengkap, epoch, dan diagnostik tersedia dalam [CSV ringkasan](pilot-local-transport-20261109-ringkasan.csv) dan direktori `pilot-local-transport-20261109/`. Diagnosis dan definisi matematis kandidat ada di [README](../README.md), dengan sumber literatur di [catatan riset](../../eksperimen/riset_augmentasi_usulan.md). Nomor checksum input telah diperiksa konsisten di dalam kondisi. Model pilot tidak disimpan; riwayat dan parameter ringkas tersimpan.

Modal memakai profil `lagi`, maksimal tiga T4 secara paralel. Ringkasan biaya akun sebelum dan sesudah pilot berubah sekitar US$0,35 pada metered cost (pembacaan agregat seluruh akun, tidak selalu langsung dan bisa berubah); ini bukan tagihan khusus pilot yang terverifikasi atau batas biaya. Cuplikan akhir menunjukkan billed cost US$0 setelah kredit pada waktu pemeriksaan.

Reproduksi ringkasan: `uv run python usulan/ringkas_pilot.py usulan/hasil/pilot-local-transport-20261109`. Kode yang dijalankan memiliki SHA-256:

```text
807a831df48e6d4ed538bed2ab4f6544b886fd25558959f8c3553bbfa447b877  usulan/local_transport.py
d164aae7141f7dad598b67b48cda76b01294fcc11f7a9b4921fae1bcd1efaef3  usulan/pilot.py
417afb2c67c982848d0f18712ce3150075277cf2c92200682f86e94ab75b48dc  usulan/modal_pilot.py
```

Langkah penelitian berikutnya: tentukan dan kunci hipotesis dan protokol sebelum evaluasi baru; uji ablation pembatas jarak dan gating terhadap kontrol; ukur sensitivitas strength; lakukan split dan seed berulang yang berpasangan dengan interval ketidakpastian; bandingkan juga supervised, identity, serta SSR-RCUS yang relevan. Gunakan holdout baru/eksternal yang belum dipakai memilih metode untuk konfirmasi, karena hasil test historis proyek telah dilihat. Klasifikasi memakai adaptasi SSL proyek tanpa auxiliary ranking classifier, bukan RankUp regresi asli.
