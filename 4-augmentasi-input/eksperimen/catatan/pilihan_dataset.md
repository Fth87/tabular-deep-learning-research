# Kandidat dataset untuk eksperimen TA

Penilaian ini untuk membandingkan enam augmentasi RankUp pada regresi kontinu, klasifikasi nominal, dan ordinal dengan label 1%, 10%, dan 30% dari **train pool**. Keputusan dataset dan perubahan kode eksperimen belum dilakukan.

## Rekomendasi awal

| Tugas | Rekomendasi | Alasan | Catatan |
|---|---|---|---|
| Regresi kontinu | [UCI Electrical Grid Stability Simulated Data](https://archive.ics.uci.edu/dataset/471/electrical%2Bgrid%2Bstability%2Bsimulated%2Bdata) | 10.000 baris simulasi, 12 fitur numerik dan target kontinu `stab`. Jika p1 yang deterministik dikeluarkan, tersisa 11 fitur untuk augmentasi; 1% train pool pada split 70% sekitar 70 label. | `stabf` adalah label turunan dari `stab` dan harus dikeluarkan untuk menghindari kebocoran target. Hasil berlaku pada simulasi, bukan bukti generalisasi ke jaringan listrik nyata. |
| Klasifikasi | Satimage ([sumber UCI Statlog](https://archive.ics.uci.edu/dataset/146/statlog%2Blandsat%2Bsatellite), [OpenML 182](https://www.openml.org/d/182)) | Sudah digunakan pada Task 3; salinan lokal memiliki 6.430 baris, 36 fitur, 6 kelas. Split lokal sekarang memiliki 45 label pada fraksi 1%. | UCI mencatat 6.435 baris, berbeda lima baris dari salinan OpenML lokal; jangan mengklaim kedua file identik sebelum verifikasi. Sampel berasal dari lingkungan piksel 3×3, sehingga generalisasi geografis tidak otomatis tercakup oleh random split. |
| Ordinal | Melbourne Airbnb ([TOC-UCO](https://www.uco.es/grupos/ayrna/materials/tocuco/)) | TOC-UCO mengelompokkannya sebagai ordinal asli; salinan Task 2 berisi 20.036 baris, 48 fitur, 10 level. Pada split 70/15/15, seed 42, 1% train memberikan 140 label dengan sedikitnya 7 untuk satu level. | Sebelumnya eksperimen memakai NHANES. Arti target dan detail preprocessing fitur anonim perlu ditelusuri sebelum interpretasi substantif. Sepuluh level dan jumlah data yang lebih besar meningkatkan kompleksitas. |

Pilihan ordinal ini menekankan reliabilitas contoh berlabel pada fraksi 1%, bukan jaminan hasil model lebih baik. Terdapat alternatif mempertahankan NHANES agar kompatibel dengan hasil sebelumnya: 5.223 baris, 30 fitur, lima level; salinan lokal pada split yang sama menghasilkan 36 label pada 1%, dan satu level hanya memperoleh satu label. [TOC-UCO](https://github.com/ayrna/tocuco) mencantumkan kedua dataset dalam kelompok ordinal asli dan menyediakan partisi untuk benchmark. Arti fitur dan label dalam file lokal tetap perlu diverifikasi.

## Pemeriksaan tambahan untuk kandidat regresi

### Electrical Grid Stability

Data resmi [UCI](https://archive.ics.uci.edu/dataset/471/electrical%2Bgrid%2Bstability%2Bsimulated%2Bdata) berupa satu berkas dengan 10.000 baris dan 14 kolom: 12 fitur, `stab` kontinu, dan `stabf` klasifikasi. Pemeriksaan langsung ZIP resmi dalam memori menemukan tidak ada missing value, tidak ada duplikat baris lengkap maupun duplikat vektor 12 fitur, dan 10.000 nilai `stab` berbeda. `stabf` sesuai tepat dengan tanda `stab`, sehingga memakainya untuk input merupakan kebocoran target. UCI menjelaskan `p1 = |p2 + p3 + p4|`; pada file, nilai maksimum `|p1+p2+p3+p4|` sekitar 5,8×10⁻¹⁵. Jika `p1` dihilangkan, 11 fitur tersisa dan hubungan deterministik ini tidak dilanggar oleh augmentasi pada variabel daya input. Namun, augmentasi masih bisa menghasilkan nilai/relasi fisik tidak realistis; hal ini perlu dievaluasi. Data berasal dari simulasi, jadi klaim hasil harus dibatasi pada sistem simulasi tersebut.

### Superconductivity

[UCI Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty%2Bdata) memiliki 21.263 baris dan 81 fitur; fraksi 1% dari perkiraan train pool 70% sekitar 149 label. Pada pemeriksaan langsung terhadap ZIP UCI tanpa mengubah workspace, `train.csv` memiliki 21.263 × 82 kolom (81 fitur dan `critical_temp`). `unique_m.csv` berjumlah sama, dan kolom `critical_temp` cocok pada urutan baris. Ditemukan **66 baris duplikat penuh**, **6.093 baris berulang berdasarkan seluruh 81 fitur** di luar kemunculan pertama, dan **2.374 kombinasi fitur** dengan lebih dari satu nilai target. Pada `unique_m.csv`, 15.542 string formula berbeda menyisakan 5.721 baris dengan formula string berulang. Hitungan ini berdasarkan file UCI yang diakses saat audit; kesamaan string bukan jaminan bahwa semua formula ekuivalen sudah teridentifikasi.

Artinya, **random split per baris dapat menaruh komposisi atau vektor fitur sama pada train dan test**, menghasilkan penilaian yang terlalu optimistis untuk material yang belum pernah dilihat. Rancang dan dokumentasikan split berdasarkan grup komposisi atau fitur serta perlakuan untuk pengukuran berulang, dan hitung ulang jumlah label aktual; jangan menyebut 21.263 baris sebagai 21.263 material unik. Group split mungkin berbeda dari pembagian acak yang digunakan pada dataset lain. Dataset ini di [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty%2Bdata) memiliki lisensi CC BY 4.0; atribusikan sumber bila digunakan.

### Alternatif yang lebih sederhana

- [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined%2Bcycle%2Bpower%2Bplant): 9.568 baris, empat fitur kontinu, target output daya. Split acak 70/15/15 memberi sekitar 66 baris berlabel pada fraksi 1%. Lebih ringan daripada Superconductivity, tetapi karena hanya ada empat fitur, perubahan jumlah fitur pada masker cenderung melompat per 25%. UCI menyebut data pengamatan per jam selama beberapa tahun dan menyediakan data yang diacak; random split tidak membuktikan generalisasi ke masa depan.
- [Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy%2Befficiency): pernah dipakai dalam notebook proyek dengan target `Y1` Heating Load. Hanya 768 konfigurasi simulasi dan delapan fitur; jika 70% menjadi train pool, 1% sekitar lima label. Beberapa prediktor berupa level desain diskret, sehingga perturbasi kontinu perlu ditafsirkan hati-hati. Cocok untuk smoke test, tetapi terlalu kecil sebagai pilihan utama untuk studi 1%.

## Protokol evaluasi yang perlu ditetapkan

- Jumlah `1%`, `10%`, dan `30%` mengacu pada train pool, bukan seluruh dataset. Semua fraksi sebaiknya memakai test dan validation yang sama pada setiap seed; untuk material berulang, kelompok juga harus tetap terpisah.
- QWK adalah metrik utama ordinal; perhitungan menggunakan bobot kuadratik berdasarkan urutan level, yang memberi penalti lebih besar untuk prediksi yang semakin jauh.
- `RankUp-CLS` pada Satimage adalah adaptasi weak-to-strong klasifikasi; klaim metodenya berbeda dari RankUp regresi asli.
- RankUp murni adalah metode semi-supervised regression. Untuk regresi kontinu, label donor TabCutMix berdasarkan pembulatan ke integer/tingkat ordinal di runner sekarang tidak otomatis tepat; perlu merancang dan menguji adaptasi donor tanpa memakai target data unlabeled, dan QWK ordinal tidak relevan pada target kontinu.
- Secara ilmiah, perbandingan enam augmentasi saja tidak mengukur manfaat RankUp atau manfaat augmentasi terhadap metode tanpa augmentasi. Baseline supervised-only relevan jika pertanyaan riset mencakup manfaat SSL; baseline tersebut di luar scope runner yang diminta pengguna saat ini.
