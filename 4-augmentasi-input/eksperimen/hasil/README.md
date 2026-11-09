# Hasil eksperimen lima seed

Hasil run `ta-5seed-40trial-20261105` diambil dari Modal, workspace `lagi`. Run ini mencakup 3 dataset, 6 augmentasi, 3 persentase label, maksimal 40 trial Optuna, dan 5 seed per kombinasi.

- `ringkasan-5seed-40trial.csv`: rata-rata dan standar deviasi metrik train berlabel, validation, serta test untuk 54 kombinasi.
- `ta-5seed-40trial-20261105/`: parameter terbaik, riwayat trial, konfigurasi, metrik tiap seed, riwayat training, prediksi test, dan ringkasan JSON.
- `data-5seed/`: split data dan preprocessor tiap dataset serta persentase label.

Checkpoint model `*_model.pt` dan snapshot studi Optuna `*_study.pkl` tidak disalin ke folder ini. Berkas aslinya tetap berada di Modal Volume `experiment-ta-results`. File penanda `*_complete.json` mencatat checksum hasil lengkap di Modal, termasuk checkpoint model yang tidak disalin ke lokal.
