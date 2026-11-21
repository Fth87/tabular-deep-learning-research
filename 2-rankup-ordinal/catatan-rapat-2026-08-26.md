# Catatan rapat pembimbing, 26 Agustus 2026

Dirangkum dari transkrip otomatis (18 menit, pembimbing dan mahasiswa). Transkripnya kasar dan banyak istilah salah dengar, jadi angka dan istilah di bawah ini perlu dicek bila dikutip. Rekaman dan transkrip aslinya tidak diunggah.

## 1. Menentukan kekuatan augmentasi weak dan strong

- Untuk augmentasi yang memilih kolom (TabCutMix, SCARF, Range-Limited), jumlah kolom yang diubah dinyatakan sebagai **persen dari jumlah kolom**, bukan angka tetap. Contoh dari pembimbing: weak sampai 25% kolom (dipilih acak) dan strong sampai 75%; atau weak 10% dan strong 50%. Kuartil juga bisa dipakai untuk menentukan batas.
- Skenario dicoba berurutan, misalnya mulai dari 10% dan 25%, lalu 30–40%. Weak tidak masuk akal kalau lebih dari 50%. Kasus terburuk untuk dicoba: weak 50% dan strong 100%.
- Untuk Gaussian, besar noise weak dan strong harus ditentukan. Nilai yang disebut sebagai contoh: weak 0,1 atau 0,01, strong 0,5 atau 0,10. Pembimbing menyebutnya bahan ablasi karena belum diketahui nilai mana yang lebih baik. Mahasiswa melaporkan Gaussian memberi hasil paling buruk.
- Hasilnya dibuat dalam tabel atau grafik: makin agresif nilai weak dan strong, apakah performa naik atau turun, dan konfigurasi weak/strong mana yang terbaik.
- Kalau bisa, dicatat persentase pseudo-label yang benar pada tiap epoch.
- Optuna boleh dipakai, termasuk di akhir. Pembimbing hanya meminta mahasiswa memahami pola nilainya dulu (misalnya lewat log), dan mengatakan cara manual juga tidak masalah.

## 2. Pencatatan hasil

Setiap eksperimen disimpan ke Excel atau CSV supaya tidak perlu diulang dari nol. Mahasiswa sudah menyimpannya sebagai CSV.

## 3. Survei paper (untuk Bab II)

- Dua topik, juga dikirim lewat chat: (1) SSL for tabular data, (2) strong and weak augmentation on tabular data. Masing-masing sekitar 10 paper.
- Hasil survei dibuat sebagai tabel dengan kolom: tahun, judul, metode (khusus untuk tabular), dan kelemahan.
- Cari di jurnal atau konferensi top-tier (misalnya IEEE Xplore), jangan jurnal atau konferensi lokal. arXiv boleh dicari dulu.
- Jenis augmentasi lain selain empat yang sudah dipakai juga disurvei, sampai yang terbaru.
- Kalau ada metode yang mirip, dicoba diterapkan. Untuk sementara RankUp dipakai untuk tabular, dan metode lain dibutuhkan sebagai model pembanding.
- Setelah eksperimen saat ini selesai, pembimbing akan memberi satu **metode baru** untuk SSL tabular yang harus dieksperimenkan.

## 4. Jumlah seed

Satu seed boleh dipakai untuk mencari konfigurasi, karena satu training memakan 5–6 jam. Kalau selisih antar metode sangat kecil (misalnya 0,02), seed perlu ditambah. Kalau selisihnya jauh (misalnya RMSE 0,5 vs 0,1), tidak perlu. Amannya 2–3 seed saat mencari, lalu 5 seed untuk konfigurasi terbaik.

## 5. Cakupan dan model

- Model tetap TabM dengan empat augmentasi.
- Tahap pertama difokuskan pada data **ordinal**. Kalau berhasil, versi kedua menambah data klasifikasi atau regresi biasa.

## 6. Jadwal

Pertemuan berikutnya Kamis pukul 13.00–15.00, luring, dengan papan tulis. Target pembimbing: semester ini sudah bisa menulis paper.
