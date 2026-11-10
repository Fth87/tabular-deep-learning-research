# Augmentasi campuran adaptif untuk pipeline RankUp

Sembilan percobaan sukses pada 3 dataset × 1%, 10%, 30% label, satu seed 42, tanpa Optuna, maksimal 120 epoch dan 100 langkah per epoch. Tidak membaca test; maksimal dua GPU T4. Augmentasi dipasang pada pipeline yang sama dengan pilot conditional sebelumnya, tanpa mengubah objective atau menambah teacher pohon. Pada klasifikasi pipeline proyek adalah adaptasi SSL tanpa ARC, bukan RankUp regresi asli.

| Dataset/metrik | Label | Campuran adaptif | Feature-guided sebelumnya | Terbaik dalam pembanding eksploratif |
|---|---:|---:|---:|---|
| NHANES / QWK ↑ | 1% | 0,293253 | 0,289418 | Gibbs k5 s50 0,309909 |
| NHANES / QWK ↑ | 10% | 0,297413 | 0,296198 | SCARF/Gibbs k20 s50 0,300610 |
| NHANES / QWK ↑ | 30% | 0,346568 | 0,331696 | Kontrol supervised none 0,348463 |
| Pen Digits / macro-F1 ↑ | 1% | 0,934627 | 0,939805 | Gibbs k5 s30 0,945551 |
| Pen Digits / macro-F1 ↑ | 10% | 0,985340 | 0,989356 | Feature-guided 0,989356 |
| Pen Digits / macro-F1 ↑ | 30% | 0,996688 | 0,996709 | Feature-guided 0,996709 |
| Superconductivity / MSE ↓ | 1% | 335,998932 | 337,106559 | Gibbs k20 s50 327,002111 |
| Superconductivity / MSE ↓ | 10% | 185,776562 | 185,491557 | Conditional group 184,668657 |
| Superconductivity / MSE ↓ | 30% | 140,061909 | 141,054573 | Group + soft rank 137,822461 |

Campuran melampaui feature-guided pada 5/9 kondisi dan tidak menjadi pemenang di antara semua kandidat pada kondisi mana pun. Pengamatan itu **belum menunjukkan** stabilitas atau signifikansi: hanya satu seed dan validation telah dipakai berkali-kali untuk menyusun serta menyaring kandidat. Metode group + soft rank juga mengubah objective, jadi perbandingan dengannya bukan ablation augmentasi murni. Skor `none` berarti bobot loss unlabeled nol dan bukan kontrol identity consistency. Kekuatan berbeda operator tidak terkalibrasi sehingga peringkat tidak membuktikan operator paling unggul dengan budget yang sepenuhnya setara.

## Algoritme

Berdasarkan train pool tanpa label validation/test, hitung `c`, rerata absolut korelasi pasangan fitur, dan `n`, jumlah sampel train berlabel. Tetapkan probabilitas operator blok korelasi `p_group = clip((c-0,25)/0,15, 0, 1) × n/(n+300)`. Pada sisa probabilitas, pilih operator fitur terarah sebesar `p_guided=(1-p_group) × n/(n+100)` dan sisanya operator fitur acak. Setiap baris pada strong view memilih satu operator; weak view identitas. Ketiganya mengambil donor pada fitur yang dipertahankan dari pool train sebanyak maksimal 2.048, dengan k=5 dan kuat 30%. Bobot fitur terarah dihitung dari labeled train. Tidak ada donor target ground truth atau label validation/test yang digunakan.

Batas dan angka probabilitas adalah heuristik yang disusun setelah meninjau eksperimen terdahulu, bukan parameter dengan jaminan optimal. Korelasi global bisa mengabaikan interaksi dan perbedaan lokal; asosiasi berlabel sedikit bisa salah. Penggantian fitur tidak menjamin label atau batas fisik terjaga. Pada Pen Digits performa turun meskipun campuran bertujuan lebih adaptif; ini alasan untuk tidak menganggap campuran sebagai perbaikan universal. Ukuran n mengubah peluang antarfraksi label sehingga metode perlu diuji pada split baru sebelum kesimpulan.

Perbandingan lengkap: [CSV](adaptive-perbandingan.csv), hasil mentah dan riwayat di `adaptive-20261110/`, implementasi [augmentasi_adaptif.py](../augmentasi_adaptif.py), desain operator awal di [temuan feature-guided](temuan_feature_guided.md). Jalankan `uv run python usulan/ringkas_adaptif.py`. Tes dasar probabilitas dan output lulus pada Modal CPU. Checksum split input identik dalam setiap kondisi. Hasil model checkpoint tidak disimpan.

SHA-256 saat run:

```text
6d7429065b776aa9d5dfec975d718ea4f09c70429fea15a56d28a592a722b50d  usulan/augmentasi_adaptif.py
2cfe147ecd2fdeaf9d099ae41ba5474c79ff159039f580245c5259ee4b137641  usulan/pilot_adaptif.py
6d83835af9ae94162cf472f6c19439bebb4cec2dbf08ad4eaad5ecc6a8ffed8a  usulan/modal_adaptif.py
```
