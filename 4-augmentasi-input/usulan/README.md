# Kandidat augmentasi: bounded local transport

Kandidat ini adalah **hipotesis eksploratif**, bukan metode yang sudah terbukti unggul atau klaim kebaruan final. Ia menggabungkan interpolasi tetangga lokal, pembatasan jarak, dan gating pseudo-label. Interpolasi dan pasangan dekat telah diteliti, antara lain pada [ICT](https://www.ijcai.org/Proceedings/2019/0504.pdf), [C-Mixup](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html), dan [SSR-RCUS](https://ojs.aaai.org/index.php/AAAI/article/view/39487). Peta literatur dan perbandingan kode terdapat di [catatan riset](../eksperimen/catatan/riset_augmentasi_usulan.md).

## Masalah yang diuji

Enam operator sebelumnya tidak memiliki pemenang validasi yang sama pada semua kombinasi. Pertukaran fitur dan noise dapat menghasilkan kombinasi yang merusak struktur, sementara donor yang kelasnya ditebak model dapat salah. Kendala ini berbeda antar dataset. Pada 270 run lama, rerata penerimaan pseudo-rank NHANES 30% di epoch terpilih hanya **0,0043**; ini pengamatan pada kode dan split saat ini, bukan bukti penyebab kinerja rendah. Jalankan `uv run python usulan/analisis_riwayat.py` untuk menghitung ulang audit.

Pemeriksaan awal **hanya pada data train berlabel** menunjukkan kedekatan Euclidean juga tidak menjamin kedekatan target. Contohnya pada subset 1%, tetangga berlabel terdekat memiliki level NHANES sama pada 16/36 anchor (44,4%), sedangkan kelas Pen Digits sama pada 48/59 anchor (81,4%). Ini hanya statistik deskriptif dari subset kecil, bukan estimasi akurasi donor pada unlabeled. Untuk Superconductivity, selisih absolut target rata-rata antara 150 anchor berlabel dan tetangga berlabel terdekat sekitar 13,04, dengan simpangan baku target berlabel 32,44. Penghitungan ini tidak memakai label validation atau test.

## Definisi metode

Untuk mini-batch `x_i ∈ R^d` yang sudah distandardisasi menggunakan train pool:

1. Hitung jarak per fitur `D(i,j) = ||x_i - x_j||₂ / √d` dan larang donor `j = i`.
2. Pada klasifikasi, cari donor terdekat dengan **pseudo-label kelas sama** dan confidence awal ≥ 0,8 untuk penerima maupun donor. Pada ordinal, gunakan label ordinal hasil pembulatan dan confidence heuristik bawaan kode proyek. Pseudo-label ini dapat salah dan confidence-nya tidak diasumsikan terkalibrasi.
3. Pada regresi kontinu, pipeline baseline tidak meneruskan prediksi atau uncertainty donor ke operator. Prototipe ini memakai donor geometris terdekat **tanpa gating target**; tidak mengklaim donor punya target atau ranking sama.
4. Jika tidak ada donor yang memenuhi syarat, gunakan identitas `x'_i = x_i`. Jika ada, pakai `x'_i = x_i + a min(1, τ / max(D(i,j), ε)) (x_j - x_i)` dengan `τ = 0,5`, `a_weak = 0,1`, dan `a_strong = 0,3`. Pencarian terbatas pada mini-batch unlabeled; anggota batch berubah saat DataLoader mengacak data.

Interpolasi seluruh vektor menjaga hubungan linear tertentu antara dua titik, tetapi **tidak menjamin** bahwa sampel berada pada manifold data atau sah secara semantik. Bilangan pecahan pada fitur diskrit, kategori/one-hot, kendala fisik nonlinier, dan tetangga geometris yang berbeda target adalah risiko; prototipe ini hanya untuk tiga dataset numerik yang diuji. Tidak ada jaminan nearest neighbor mini-batch adalah tetangga terdekat seluruh dataset. Fallback identitas juga dapat membuat perubahan terlalu kecil dan konsistensi menjadi kurang informatif.

Untuk ordinal dan regresi, operator dipasang pada weak/strong view pada pipeline RankUp tabular yang ada, termasuk auxiliary ranking classifier. Untuk klasifikasi nominal, pipeline proyek **tidak memiliki ARC**; operator diuji pada adaptasi konsistensi kelas ala FixMatch, bukan RankUp regresi asli. Implementasi: [local_transport.py](local_transport.py).

## Batas matematis yang bersyarat

Dengan `0 ≤ a ≤ 1` dan `τ > 0`, rumus di atas menjamin secara aljabar `||x'_i-x_i||₂ / √d ≤ aτ`, hingga toleransi numerik. Pada parameter pilot, batas view strong ialah `0,3 × 0,5 = 0,15` dalam RMS fitur terstandardisasi. Tes memeriksa batas ini serta fallback donor.

Jika, sebagai asumsi tambahan yang belum diverifikasi, fungsi target sejati `f*` adalah `L`-Lipschitz sepanjang segmen tersebut, maka `|f*(x'_i)-f*(x_i)| ≤ L aτ√d`. Untuk dua sampel dengan batas perubahan target sama, urutan target terjaga apabila margin target sejati melebihi `2 L aτ√d`. `L` dan label unlabeled tidak diketahui, dan asumsi tersebut bisa gagal. Batas ini **bukan bukti** bahwa augmentasi menjaga label/rank di data nyata, atau bahwa QWK, F1, maupun MSE akan membaik. Untuk klasifikasi dibutuhkan asumsi margin kelas yang terpisah.

## Pilot murah

Satu seed 42, tanpa Optuna dan tanpa membaca test: 3 dataset × 3 fraksi label × 9 metode = 81 job. Metode terdiri atas enam augmentasi lama, kontrol identitas dengan SSL, kontrol objective berlabel dengan bobot unlabeled nol, dan kandidat ini. Semua memakai maksimal 30 epoch, 40 langkah per epoch, batch 32, early stopping 15, dan kekuatan weak/strong tetap 0,1/0,3. Model supervised pada ordinal/regresi tetap memiliki ranking loss berlabel; preprocessing tetap transduktif karena sebelumnya di-fit pada seluruh train pool. Itu bukan supervised yang sepenuhnya tanpa akses ke fitur unlabeled.

Gunakan nilai validation: QWK dan macro-F1 lebih tinggi lebih baik; MSE lebih rendah lebih baik. Strength 0,1/0,3 tidak memiliki arti atau tingkat kerusakan yang sama untuk tiap operator, sehingga ini **screening dengan budget sama**, bukan perbandingan final yang sudah terkalibrasi. Metode lama sebelumnya dituning 40 trial dan dilatih dengan budget berbeda, sehingga angka pilot tidak bisa dibandingkan langsung sebagai bukti keunggulan atas tabel hasil lama. Jangan memilih desain dari test; test proyek telah dilihat sebelumnya, sehingga konfirmasi kuat perlu holdout baru/eksternal.

Per-job timeout 300 detik, maksimal 3 GPU T4 bersamaan. Ini membatasi job, bukan batas tagihan keras; startup, CPU/memori, retry infrastruktur, dan penyimpanan dapat menambah biaya. Pilot ini bukan lima seed, tidak membuktikan stabilitas statistik, dan tidak menyediakan ketidakpastian antar-split. Untuk klaim: tetapkan protokol sebelum eksperimen konfirmasi, bandingkan dengan SSR-RCUS dan metode tabular yang relevan, ablasikan gating/jarak/interpolasi, lakukan validasi berulang lintas split serta seed berpasangan, gunakan data independen yang belum dipakai memilih metode, dan laporkan selisih serta interval ketidakpastian dengan unit evaluasi yang benar.

Hasil pilot pada Modal Volume `experiment-ta-results/pilot-local-transport-20261109/` mencatat skor dan riwayat validation, epoch, acceptance rate, checksum input dan diagnostik transport (cakupan donor, batas jarak, perubahan RMS). Di dalam data lokal tugas ini, label validation digunakan untuk seleksi model; label test tidak dibuka oleh pilot. Checkpoint model tidak disimpan pada pilot; untuk penelitian akhir perlu menambah penelusuran artefak model dan lingkungan secara lengkap.
