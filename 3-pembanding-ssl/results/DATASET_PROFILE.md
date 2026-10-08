# Profil dataset Task 3

## Klasifikasi

| Dataset | Baris | Fitur (numerik/kategorikal) | Distribusi kelas (%) |
|---|---:|---:|---|
| Diabetes Pima | 768 | 8 / 0 | Negatif 65.1; positif 34.9 |
| CMC | 1,473 | 2 / 7 | Kelas 1: 42.7; 2: 22.6; 3: 34.7 |
| Satimage | 6,430 | 36 / 0 | Kelas 1: 23.8; 2: 10.9; 3: 21.1; 4: 9.7; 5: 11.0; 7: 23.5 |
| Bank Marketing | 45,211 | 7 / 9 | Kelas 1: 88.3; kelas 2: 11.7 |
| Adult Income | 48,842 | 6 / 8 | `<=50K`: 76.1; `>50K`: 23.9 |

## Ordinal regression

| Dataset | Baris | Fitur (numerik/kategorikal) | Distribusi level target (%) |
|---|---:|---:|---|
| LEV | 1,000 | 4 / 0 | Level 0–4: 9.3 / 28.0 / 40.3 / 19.7 / 2.7 |
| NHANES | 5,223 | 30 / 0 | Level 0–4: 10.7 / 28.2 / 39.7 / 17.7 / 3.7 |
| ESL | 488 | 4 / 0 | Level 0–5: 10.7 / 20.5 / 23.8 / 27.7 / 12.7 / 4.7 |
| Student Performance | 662 | 43 / 0 | Level 0–7: 6.8 / 3.6 / 11.2 / 24.2 / 21.6 / 15.6 / 11.0 / 6.0 |
| Melbourne Airbnb | 20,036 | 48 / 0 | Level 0–9: 8.3 / 9.1 / 9.6 / 8.6 / 4.8 / 12.4 / 10.6 / 8.9 / 9.5 / 18.2 |

Catatan: distribusi target ditampilkan sebagai persentase per kelas/level sesuai urutan label. Eksperimen menggunakan split train/validation/test 70/15/15; 10% dari train pool dipakai sebagai data berlabel. Adult Income memiliki 6,465 sel fitur kosong; sembilan dataset lain tidak memiliki nilai fitur kosong.
