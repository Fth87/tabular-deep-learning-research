# Riset augmentasi untuk RankUp dan data tabular

Tanggal penelusuran: 9 November 2026. Dokumen ini membedakan gagasan yang didukung literatur, hipotesis penelitian, dan hasil eksperimen. Kebaruan dan keunggulan metode tidak dapat dipastikan hanya dari rancangan; pencarian literatur juga tidak menjamin semua karya yang relevan sudah tercakup.

## Temuan utama

Tidak ada satu augmentasi yang selalu menang pada sembilan kombinasi dataset dan persentase label pada hasil yang tersedia. Hal ini sesuai dengan dugaan bahwa label, struktur fitur, kekuatan augmentasi, dan kualitas pseudo-label berinteraksi. Kemenangan tipis pada validasi bukan bukti keunggulan statistik. RankUp sendiri adalah metode semi-supervised regression dengan auxiliary ranking classifier (ARC), bukan augmentasi tunggal. Untuk klasifikasi nominal, adaptasi berbasis pseudo-label yang tidak memakai ARC perlu disebut sebagai adaptasi SSL, bukan reproduksi RankUp asli. [RankUp, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c26a8494fe31695db965ae8b7244b7c1-Paper-Conference.pdf); [kode penulis](https://github.com/pm25/Semi-Supervised-Regression).

Ada dua benturan prior art yang penting. [SSR-RCUS, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39487) telah menggabungkan pasangan unlabeled yang berdekatan, mixup, dan hubungan ranking berkepercayaan tinggi untuk memperbaiki RankUp. [Kode penulis SSR-RCUS](https://github.com/changchunli/SSR-RCUS). [Cui dkk., 2024](https://arxiv.org/html/2404.17489) telah menggunakan pseudo-label untuk pengondisian korupsi dan korelasi fitur untuk pemilihan masker dalam contrastive learning tabular; penulis melaporkan tidak ada bukti konkret peningkatan untuk masker berbasis korelasi. [Kode penulis](https://github.com/layer6ai-labs/Tabular-Class-Conditioned-SSL). Karena itu, pasangan dekat plus ranking atau pseudo-label plus korelasi tidak boleh diklaim baru sebagai ide tersendiri.

## Audit hasil lokal yang tersedia

Sumber: `hasil/ringkasan-5seed-40trial.csv`, 54 baris, masing-masing agregasi lima seed. Pemenang ditentukan hanya dari metrik validasi: QWK dan F1 lebih tinggi lebih baik, MSE lebih rendah lebih baik. Kolom selisih adalah selisih rata-rata antara pemenang dan peringkat kedua, bukan interval kepercayaan.

| Dataset | Label | Terbaik pada validasi | Skor validasi | Selisih dengan kedua | Skor test terkait |
|---|---:|---|---:|---:|---:|
| NHANES ordinal | 1% | range_limited | 0.310433 | 0.003755 | 0.277428 |
| NHANES ordinal | 10% | tabcutmix | 0.299855 | 0.001676 | 0.231107 |
| NHANES ordinal | 30% | scarf | 0.322326 | 0.000454 | 0.304119 |
| Pen Digits klasifikasi | 1% | tabcutmix | 0.942199 | 0.000796 | 0.897197 |
| Pen Digits klasifikasi | 10% | tabcutmix | 0.991554 | 0.001708 | 0.967280 |
| Pen Digits klasifikasi | 30% | gaussian | 0.997119 | 0.000567 | 0.976032 |
| Superconductivity regresi | 1% | scarf | 337.139840 | 0.571861 | 376.620 |
| Superconductivity regresi | 10% | tabcutmix_plus | 188.395470 | 0.168359 | 205.081 |
| Superconductivity regresi | 30% | tabcutmix_plus | 138.435530 | 0.643284 | 147.295 |

TabCutMix menang tiga kondisi, SCARF dan TabCutMix+ masing-masing dua, sedangkan range-limited dan Gaussian masing-masing satu. Hasil ini bersifat eksploratif: selisih kecil harus diperiksa dengan perbedaan berpasangan per seed dan ketidakpastian dari sampel test. Lima seed pada satu pembagian data tidak mengukur variasi seluruh populasi atau variasi pemilihan split. Test sudah pernah dilihat dalam eksplorasi; untuk klaim konfirmatori yang kuat, sediakan data evaluasi baru yang belum digunakan untuk memilih metode.

## Peta paper dan kode asli

| Pendekatan | Gagasan dan hubungan dengan eksperimen |
|---|---|
| [RankUp, 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c26a8494fe31695db965ae8b7244b7c1-Abstract-Conference.html) / [kode](https://github.com/pm25/Semi-Supervised-Regression) | ARC dan konsistensi pseudo-ranking untuk regresi semi-supervised. Penggunaan operator tabular tertentu adalah pilihan tambahan dan harus diidentifikasi terpisah. |
| [SSR-RCUS, 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39487) / [kode](https://github.com/changchunli/SSR-RCUS) | Sample mixup yang dekat dan pseudo-ranking; baseline langsung paling penting untuk kebaruan usulan regresi berbasis RankUp. Jaminan teorinya bergantung pada asumsi yang dinyatakan paper. |
| [TabCutMix, ICML 2025](https://proceedings.mlr.press/v267/fang25f.html) / [kode](https://github.com/fangzy96/TabCutMix) | Pertukaran fitur antarcontoh satu kelas; varian Plus memakai struktur korelasi. Paper menargetkan memorisasi model generatif; donor pseudo-label atau bin regresi adalah adaptasi, bukan reproduksi utuh. |
| [SCARF, ICLR 2022](https://openreview.net/pdf?id=CuV_qYkmKb3) | Korupsi fitur dengan sampling dari marginal empiris dalam kerangka pretraining contrastive. Memakai operator korupsi saja tidak sama dengan menjalankan keseluruhan SCARF. |
| [VIME, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) / [kode](https://github.com/jsyoon0823/VIME) | Korupsi fitur, prediksi mask dan rekonstruksi, lalu konsistensi semi-supervised. Bedakan operator korupsi dari keseluruhan pipeline VIME. |
| [Range-Limited Augmentation, KDD 2025](https://doi.org/10.1145/3711896.3737098) / [kode](https://github.com/kyungeun-lee/festa) | Membatasi nilai korupsi berdasarkan rentang fitur. Nilai yang valid secara individual belum tentu menghasilkan kombinasi fitur yang valid. |
| [SubTab, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html) / [kode](https://github.com/AstraZeneca/SubTab) | Pandangan berupa subset fitur dalam self-supervised learning; zero-out saja merupakan operator yang terinspirasi, bukan implementasi penuh SubTab. Nilai nol setelah standardisasi bukan selalu token data hilang. |
| [GsRCL, 2024](https://academic.oup.com/bfg/article/23/4/441/7577350) / [kode](https://github.com/ibrahimsaggaf/GsRCL) | Gaussian noise dalam konteks representasi single-cell; Gaussian tabular umum tidak dengan sendirinya mereproduksi paper ini. |
| [C-Mixup, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/file/1626be0ab7f3d7b3c639fbfd5951bc40-Paper-Conference.pdf) / [kode](https://github.com/huaxiuyao/C-Mixup) | Pasangan interpolasi dipilih berdasarkan kedekatan label untuk regresi supervised. Hasil teoritis yang diberikan mengikuti asumsi tertentu; tidak langsung berlaku untuk pseudo-label. |
| [ICT, IJCAI 2019](https://www.ijcai.org/Proceedings/2019/0504.pdf) / [kode](https://github.com/google-research/mixmatch/blob/master/ict.py) | Konsistensi antara prediksi pada campuran data unlabeled dan campuran prediksi teacher; ini juga prior art untuk mixup dengan pseudo-target. |
| [RaTab, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33265) | Augmentasi di ruang representasi sebagai alternatif gangguan langsung pada fitur. Interpolasi laten tetap tidak menjamin validitas semantik atau peningkatan seluruh tugas. |
| [Cui dkk., 2024](https://arxiv.org/html/2404.17489) / [kode](https://github.com/layer6ai-labs/Tabular-Class-Conditioned-SSL) | Korupsi class-conditioned memakai pseudo-label dan masker berbasis korelasi. Penelitian ini membuat klaim kebaruan generik atas kombinasi tersebut tidak memadai. |
| [Heteroscedastic pseudo-labels, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/30a129d6b8e4229d1ed6d90a65ee1a76-Abstract-Conference.html) | Ketidakpastian pseudo-label regresi dapat berbeda antarcontoh. Confidence heuristik perlu divalidasi, bukan dianggap otomatis terkalibrasi. |
| [Hierarchical confidence screening dan neighbor calibration, 2026](https://doi.org/10.1109/ACCESS.2026.3680669) | Screening confidence dan kalibrasi pseudo-label berdasarkan tetangga juga telah diteliti pada regresi semi-supervised. |
| [SeBA, 2026](https://arxiv.org/abs/2605.08519) | Alternatif alignment antar-view bertetangga tanpa membuat augmentasi sintetis, berguna sebagai pembanding konseptual. |

### Empat paper yang tersedia di folder `papers`

1. RankUp (di atas) adalah titik awal regresi, termasuk ARC; rujukan paper dan implementasi harus dibedakan dari adaptasi proyek.
2. [Chen, *A Semi-Supervised Ordinal Classification Method*, 2025](https://doi.org/10.1109/AINIT65432.2025.11035003) menggabungkan ide ordinal, ARC, konsistensi rank dan threshold kelas adaptif dalam konteks citra. Ini pembanding metodologis ordinal, bukan bukti bahwa implementasi regresi yang dibulatkan sudah mereproduksi metode tersebut.
3. [Niu dkk., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.html) merumuskan regresi ordinal sebagai beberapa klasifikasi biner. Metode supervised pada citra ini memotivasi pemodelan urutan, bukan augmentasi SSL tabular.
4. [Wang dkk., *Contrastive Learning Spectral Seriation*](https://arxiv.org/abs/2512.09267) / [kode CLSS](https://github.com/xmed-lab/CLSS) menghubungkan kesamaan fitur, anchor berlabel, dan seriation spektral. Jaminan teoretisnya mengikuti asumsi paper dan tidak otomatis menjadi jaminan RankUp tabular.

## Arah metode yang layak diuji

Hipotesis kerja: gunakan augmentasi lokal yang menjaga kelompok fitur bila dapat dipertanggungjawabkan, membatasi besarnya perubahan, dan menurunkan atau menolak bobot konsistensi ketika pasangan tidak cukup andal. Untuk regresi/ordinal, informasi urutan dari ARC dapat membantu menyaring atau memberi bobot pasangan; untuk klasifikasi gunakan kesesuaian kelas dan confidence, tanpa berpura-pura ada ARC jika memang tidak ada. Selalu sediakan fallback identitas bila tidak ada donor yang memenuhi syarat.

Ini adalah **kombinasi/adaptasi eksploratif**, bukan klaim bahwa masing-masing komponennya baru. Nilai penelitiannya harus dibuktikan lewat rumusan spesifik, perbandingan terhadap SSR-RCUS dan Cui dkk. yang relevan, ablation, dan perbaikan konsisten pada evaluasi yang belum digunakan untuk pemilihan. Korelasi bukan hubungan kausal; neighborhood dalam ruang fitur bukan otomatis neighborhood target. Ambang confidence dan kekuatan perturbasi harus dibekukan sebelum evaluasi, atau dipilih hanya melalui validasi dengan anggaran yang sama pada semua metode.

**Teori bersyarat yang dapat diperiksa:** andaikan fungsi target sebenarnya `f*` bersifat `L`-Lipschitz pada perturbasi yang diizinkan dan `||x' - x|| ≤ δ`, maka `|f*(x') - f*(x)| ≤ Lδ`. Untuk dua sampel yang masing-masing berubah paling besar `δ`, urutan target aslinya terjaga jika selisih target sebenarnya lebih besar dari `2Lδ`. Ini tidak membuktikan keberhasilan pada data nyata: `L` dan target unlabeled tidak diketahui, jarak/representasi dapat salah, serta asumsi dapat gagal pada fitur diskrit atau batas kelas. Hasil ini juga tidak menghasilkan batas langsung untuk F1, QWK, atau MSE model. Klasifikasi membutuhkan asumsi margin kelas yang terpisah. Kesepakatan model, korelasi fitur dan confidence semata bukan jaminan kebenaran; model neural dapat tidak terkalibrasi. [Guo dkk., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).

Jika variasi prediksi kepala model atau beberapa stochastic pass digunakan sebagai confidence, sebut sebagai **heuristik ketidakpastian**; komponen yang berbagi parameter tidak otomatis merupakan ensemble independen atau estimator ketidakpastian yang terkalibrasi. Ukur hubungan skor tersebut dengan kesalahan pada subset berlabel validasi melalui kurva risk–coverage dan cek tingkat penerimaan. Hindari memakai label test untuk menyetel ambang.

## Implikasi dataset

- NHANES ordinal: kelas langka pada subset label 1% hanya memiliki satu contoh berlabel dalam pembagian yang tersedia. Estimasi per kelas dan pseudo-label menjadi rapuh; laporkan distribusi kelas, confusion matrix dan QWK per seed. Nama fitur yang tidak menjelaskan semantik membatasi klaim validitas augmentasi.
- Pen Digits: 16 fitur adalah koordinat turunan lintasan tulisan; mengganti koordinat secara terpisah dapat merusak hubungan geometris. Class-conditioned donor tetap bisa salah jika pseudo-label salah. Pembagian test resmi berbeda penulis, sehingga perbandingan harus mempertahankan split resmi.
- Superconductivity: fitur rekayasa berkorelasi dan dataset lokal memiliki pengulangan vektor serta kelompok dengan target berbeda. Split berdasarkan grup vektor identik membantu mencegah vektor identik melintasi train/validasi/test, tetapi tidak mencegah semua kedekatan material lintas split atau duplikasi antara subset labeled dan unlabeled di dalam train. Tetangga fitur tidak otomatis mempunyai target identik.

## Protokol eksperimen

1. Untuk eksplorasi cepat sesuai permintaan: jalankan 3 dataset × 3 persentase label (1%, 10%, 30%) dengan satu seed yang sama, tanpa Optuna. Bekukan split, parameter awal, jumlah epoch, metode pre-processing dan anggaran komputasi. Laporkan QWK ordinal, F1 klasifikasi dan MSE regresi pada train/validasi; simpan konfigurasi, seed dan runtime.
2. Jalankan metode usulan dan keenam augmentasi dengan konfigurasi pembanding yang setara. Sertakan supervised-only, serta kontrol SSL tanpa augmentasi; tanpa augmentasi, konsistensi bisa trivial, jadi kedua kontrol menjawab pertanyaan berbeda. Pembandingan langsung satu konfigurasi usulan terhadap baseline 40-trial sebelumnya bersifat deskriptif dan bukan estimasi efek yang adil. Evaluasi baseline SSL sederhana penting karena SSL dapat gagal atau kalah dari supervised. [Oliver dkk., NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html).
3. Catat proporsi donor diterima/ditolak, fallback identitas, jarak donor, kekuatan perubahan, confidence/ARC, dan lama komputasi. Ablation terpisah perlu menguji pembatasan jarak, pengelompokan fitur, serta gating/weighting; bila semua berubah sekaligus, kontribusi tiap bagian tidak teridentifikasi.
4. Pilih kandidat dengan validasi dan tulis keputusan sebelum menghitung test. Hasil satu seed merupakan screening, bukan bukti stabil. Pada tahap konfirmasi, ulang lima seed berpasangan, tambah variasi pemilihan data berlabel/split bila memungkinkan, laporkan selisih berpasangan dan interval bootstrap per unit test yang sesuai; pertimbangkan koreksi perbandingan berganda untuk hipotesis yang telah ditetapkan. Jangan menafsirkan lima seed sebagai lima dataset independen.
5. Karena ketiga tugas hanya diwakili masing-masing satu dataset, temuan tidak membuktikan generalisasi ke seluruh regresi, ordinal, atau klasifikasi. Sediakan dataset eksternal atau test baru yang belum dipakai untuk pemilihan sebelum membuat klaim luas.
