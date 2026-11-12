# Catatan revisi Bab II

## Tata letak

Keempat tabel memakai `table` dan `\columnwidth`, sehingga masing-masing menempati satu kolom IEEE. Tabel perbandingan diringkas menjadi tiga kolom, sedangkan tabel operator dan penelitian terdahulu menjadi dua kolom agar teks tidak terlalu sempit. Caption tetap di atas tabel dan rujukan dalam teks menggunakan `\ref`. Kolom pada halaman terakhir diseimbangkan dengan `\IEEEtriggeratref`; margin dan ukuran kolom IEEEtran tidak diubah.

## Bahasa dan isi

- Pengulangan penjelasan SimCLR, MAE, VIME, SubTab, SCARF, pseudo-label, dan konsistensi diringkas. Kalimat kesimpulan umum yang mengulangi paragraf sebelumnya dihapus.
- Penjelasan enam operator, tiga dataset, tiga fraksi label, Optuna, lima seed dan tiga metrik utama tetap ada. Rangkumannya lebih padat daripada naskah awal; bandingkan kembali jika ada uraian yang diperlukan dosen pembimbing.
- Perbedaan RankUp untuk regresi/ordinal dan adaptasi klasifikasi tanpa ARC tetap dinyatakan. Batasan baseline dipersempit pada rangkaian enam operator, karena workspace juga memiliki pilot lain dengan kontrol tambahan.
- Keempat tabel dan 16 referensi Bab II tetap ada. Referensi contoh dari template IEEE juga tetap ada atas permintaan sebelumnya dan bukan referensi penelitian; hapus saat naskah final disusun.
- Revisi anti-slop mengurangi pola repetitif dan kalimat generik, bukan mendeteksi apakah teks ditulis oleh AI. Uraian dan referensi belum diaudit secara menyeluruh terhadap setiap paper; pemeriksaan ilmiah dan kesesuaian dengan eksperimen final masih diperlukan.

Sumber asli Markdown tidak diubah. Versi LaTeX dan PDF sebelum revisi disimpan di `arsip_sebelum_revisi/`.

## Penyesuaian struktur lanjutan

Subbagian self-supervised learning pada citra dan tabular dipisahkan kembali. Pseudo-labeling dan consistency regularization ditempatkan pada bagian semi-supervised. Paket `flafter` dan opsi float `[!htbp]` membantu tabel muncul setelah penjelasan yang merujuknya; posisi dapat berubah saat naskah lain diubah. `\IEEEtriggeratref{16}` dipakai untuk mengatur ulang daftar pustaka pada versi saat ini dan perlu diperiksa kembali jika isi berubah. Penomoran A, B, dan seterusnya mengikuti IEEEtran; penomoran 2.1, 2.2 adalah format tesis yang berbeda.
