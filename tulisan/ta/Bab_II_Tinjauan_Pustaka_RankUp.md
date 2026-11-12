# II. TINJAUAN PUSTAKA

## A. Data Augmentation

Data augmentation merupakan teknik memperluas variasi data pelatihan melalui transformasi terhadap sampel yang telah tersedia. Teknik ini bertujuan membantu model mempelajari pola yang lebih umum sehingga tidak hanya bergantung pada karakteristik data pelatihan. Shorten dan Khoshgoftaar [1] menjelaskan bahwa augmentasi dapat mengurangi risiko *overfitting* dan meningkatkan kemampuan generalisasi model, terutama ketika jumlah data pelatihan terbatas.

Augmentasi tidak selalu menghasilkan sampel dengan informasi yang sama persis seperti data asal. Perubahan yang terlalu besar dapat menghilangkan karakteristik penting atau menghasilkan data yang tidak sesuai dengan distribusi sebenarnya. Oleh karena itu, pemilihan transformasi perlu mempertimbangkan struktur data dan informasi yang berkaitan dengan target prediksi.

Dalam pembelajaran menggunakan data tanpa label, augmentasi memiliki fungsi tambahan untuk menghasilkan beberapa versi dari sampel yang sama. Versi tersebut dapat digunakan dalam pembelajaran representasi maupun konsistensi prediksi. Dengan demikian, augmentasi menjadi salah satu komponen penting dalam pendekatan *self-supervised learning* dan *semi-supervised learning*.

### 1) Image Data Augmentation

Data citra memiliki struktur spasial yang terbentuk dari hubungan antar-piksel. Karakteristik tersebut memungkinkan penerapan transformasi geometris dan perubahan tampilan visual, seperti *rotation*, *horizontal flipping*, *random cropping*, *resizing*, dan *color jittering*. Transformasi ini dapat menghasilkan variasi tampilan tanpa mengubah identitas objek yang menjadi target prediksi, selama perubahan yang dilakukan masih relevan terhadap tugas [1].

Chen et al. [2] memperkenalkan SimCLR, yaitu metode pembelajaran representasi yang menggunakan dua hasil augmentasi dari citra yang sama sebagai pasangan positif. Model dilatih untuk menghasilkan representasi yang berdekatan bagi pasangan tersebut dan membedakannya dari representasi sampel lain. Penelitian ini menunjukkan bahwa komposisi augmentasi memiliki pengaruh penting terhadap kualitas representasi yang dipelajari melalui *contrastive learning*.

Pendekatan lain diperkenalkan He et al. [3] melalui *Masked Autoencoders* (MAE). Metode ini menyembunyikan sebagian patch citra dan melatih model untuk merekonstruksi bagian yang hilang. Berbeda dari SimCLR yang mengandalkan pembandingan representasi, MAE menggunakan tugas rekonstruksi sebagai sumber pembelajaran.

Kedua penelitian tersebut menunjukkan bahwa augmentasi pada citra dapat digunakan untuk tujuan pembelajaran yang berbeda. SimCLR memanfaatkan variasi tampilan dalam contrastive learning, sedangkan MAE memanfaatkan informasi yang disembunyikan untuk mempelajari representasi visual.

### 2) Tabular Data Augmentation

Data tabular terdiri atas sejumlah baris yang merepresentasikan sampel dan kolom yang merepresentasikan fitur. Setiap fitur dapat memiliki tipe, distribusi, dan makna yang berbeda, seperti nilai numerik, kategori, atau tingkat ordinal. Berbeda dengan citra, posisi suatu kolom dalam tabel tidak secara langsung menunjukkan hubungan spasial dengan kolom lainnya.

Wang et al. [4] menjelaskan bahwa tidak adanya struktur spasial atau sekuensial yang eksplisit menjadi salah satu tantangan dalam pengembangan self-supervised learning untuk data tabular. Kondisi tersebut juga memengaruhi pemilihan teknik augmentasi karena perubahan nilai fitur berpotensi mengubah hubungan antaratribut.

Yoon et al. [5] memperkenalkan *Value Imputation and Mask Estimation* (VIME), yang menggunakan korupsi fitur untuk membentuk tugas rekonstruksi dan estimasi mask. Melalui mekanisme tersebut, model mempelajari hubungan antarfitur tanpa menggunakan label target utama pada tahap pretraining.

Pendekatan berbeda dikembangkan Ucar et al. [6] melalui SubTab. Metode ini membagi fitur menjadi beberapa subset sehingga model dapat mempelajari representasi dari bagian informasi yang berbeda. Sementara itu, Bahri et al. [7] melalui SCARF menggunakan *random feature corruption* untuk membentuk pasangan sampel dalam contrastive learning.

Penelitian-penelitian tersebut memperlihatkan bahwa augmentasi tabular dapat dilakukan melalui penghilangan nilai fitur, penggantian nilai, maupun pembentukan subset fitur. Namun, berbeda dengan transformasi geometris pada citra, modifikasi fitur tabular perlu mempertimbangkan validitas kombinasi atribut.

Sebagai contoh, penggantian nilai satu fitur secara independen dapat menghasilkan kombinasi yang jarang ditemukan pada data asli. Kondisi ini berpotensi mengubah informasi yang dibutuhkan model untuk memprediksi target. Oleh karena itu, efektivitas suatu teknik augmentasi tabular tidak hanya ditentukan oleh besarnya variasi yang dihasilkan, tetapi juga oleh kemampuannya mempertahankan informasi yang relevan.

### 3) Perbandingan Augmentasi Citra dan Tabular

**TABEL I. PERBANDINGAN DATA AUGMENTATION PADA CITRA DAN TABULAR**

| Aspek | Data Citra | Data Tabular |
|---|---|---|
| Struktur data | Hubungan spasial antar-piksel | Hubungan antarfitur yang heterogen |
| Tipe informasi | Warna, bentuk, tekstur, dan posisi | Nilai numerik, kategori, dan ordinal |
| Teknik augmentasi | Rotation, cropping, flipping, color jitter | Masking, feature corruption, noise, feature swapping |
| Informasi yang dijaga | Karakteristik visual yang relevan | Makna fitur dan hubungan antaratribut |
| Risiko transformasi | Hilangnya informasi objek | Kombinasi fitur tidak valid atau perubahan informasi target |
| Penelitian terkait | SimCLR [2], MAE [3] | VIME [5], SubTab [6], SCARF [7] |

Berdasarkan Tabel I, augmentasi citra dan tabular memiliki prinsip yang sama, yaitu menghasilkan variasi data, tetapi berbeda dalam mekanisme transformasi dan karakteristik informasi yang perlu dipertahankan. Pada citra, transformasi dapat memanfaatkan keteraturan spasial, sedangkan pada data tabular, setiap perubahan harus mempertimbangkan karakteristik fitur.

Perbedaan tersebut menjadi dasar penelitian ini dalam membandingkan beberapa teknik augmentasi data tabular. Setiap teknik mempunyai mekanisme perubahan fitur yang berbeda sehingga berpotensi menghasilkan pengaruh yang berbeda terhadap pembelajaran dari data tanpa label.

## B. Self-Supervised Learning

*Self-supervised learning* merupakan pendekatan pembelajaran yang membentuk sinyal supervisi berdasarkan informasi dari data itu sendiri, tanpa memerlukan anotasi target utama. Dalam pendekatan ini, model menyelesaikan suatu *pretext task* untuk mempelajari representasi yang dapat digunakan pada tugas prediksi berikutnya [4].

Proses self-supervised learning umumnya melibatkan dua tahap. Tahap pertama adalah *pretraining*, yaitu pembelajaran representasi menggunakan data tanpa label target utama. Tahap kedua adalah *downstream learning*, yaitu penggunaan representasi yang telah dipelajari untuk menyelesaikan tugas tertentu, seperti klasifikasi atau regresi.

Pendekatan self-supervised dapat dibedakan berdasarkan objective pembelajarannya. Salah satu pendekatan menggunakan tugas prediksi atau rekonstruksi terhadap bagian data yang dimodifikasi. Pendekatan lainnya menggunakan contrastive learning untuk mempelajari hubungan antarrepresentasi. Wang et al. [4] juga mengidentifikasi pendekatan hybrid yang menggabungkan beberapa objective pembelajaran.

Meskipun sama-sama memanfaatkan data tanpa label, self-supervised learning perlu dibedakan dari semi-supervised learning. Self-supervised learning membentuk tugas supervisi dari data, sedangkan semi-supervised learning memanfaatkan data berlabel dan tanpa label untuk mempelajari tugas prediksi utama.

### 1) Self-Supervised Learning on Image Data

Penerapan self-supervised learning pada citra memanfaatkan informasi visual dan hubungan spasial yang terkandung dalam gambar. Dua pendekatan yang umum digunakan adalah contrastive learning dan reconstruction-based learning.

Pada contrastive learning, model dilatih untuk menghasilkan representasi yang sesuai bagi dua versi augmentasi dari citra yang sama. SimCLR [2] menerapkan pendekatan tersebut dengan menghasilkan dua *augmented views* dan mengoptimalkan representasinya menggunakan contrastive objective.

Sementara itu, reconstruction-based learning memanfaatkan tugas membangun kembali informasi yang dihilangkan. MAE [3] menyembunyikan sebagian patch citra dan menggunakan encoder serta decoder untuk merekonstruksi bagian yang tidak terlihat.

Kedua pendekatan tersebut menunjukkan bahwa data citra memungkinkan pembentukan pretext task berdasarkan karakteristik visual. Namun, keberhasilan pendekatan tersebut bergantung pada jenis transformasi dan asumsi mengenai informasi yang harus dipertahankan selama pembelajaran.

### 2) Self-Supervised Learning on Tabular Data

Self-supervised learning pada data tabular bertujuan mempelajari representasi dari hubungan antarfitur tanpa menggunakan label target utama. Penerapannya menghadapi tantangan karena data tabular tidak selalu memiliki struktur eksplisit yang dapat digunakan untuk membentuk pasangan representasi atau tugas rekonstruksi.

Wang et al. [4] mengelompokkan metode self-supervised learning pada data tabular nonsekuensial menjadi tiga pendekatan utama, yaitu *predictive learning*, *contrastive learning*, dan *hybrid learning*.

Predictive learning memanfaatkan sebagian informasi fitur untuk memprediksi informasi lain yang disembunyikan atau dimodifikasi. VIME [5] menggunakan mask estimation dan value reconstruction untuk membentuk tugas pembelajaran dari fitur tabular.

Contrastive learning bertujuan mempelajari kedekatan representasi antara beberapa versi dari sampel yang sama. SCARF [7] menerapkan random feature corruption untuk menghasilkan augmented views yang digunakan selama contrastive pretraining.

SubTab [6] menggunakan pendekatan multi-view dengan membagi fitur menjadi beberapa subset. Model mempelajari representasi melalui informasi yang terkandung pada subset tersebut, termasuk melalui rekonstruksi dan penggabungan representasi.

Ketiga penelitian tersebut menunjukkan bahwa pengembangan Tabular SSL tidak hanya bergantung pada arsitektur model, tetapi juga pada cara informasi fitur digunakan untuk membentuk tugas pembelajaran.

### 3) Perbandingan Self-Supervised Learning pada Citra dan Tabular

**TABEL II. PERBANDINGAN SELF-SUPERVISED LEARNING PADA CITRA DAN TABULAR**

| Aspek | Image SSL | Tabular SSL |
|---|---|---|
| Struktur utama | Hubungan spasial antar-piksel | Hubungan antarfitur |
| Pembentukan augmented views | Transformasi visual | Korupsi, masking, dan subsetting fitur |
| Predictive learning | Rekonstruksi patch atau piksel | Rekonstruksi dan prediksi nilai fitur |
| Contrastive learning | Membandingkan representasi citra | Membandingkan representasi tabular |
| Tantangan | Pemilihan transformasi yang menjaga informasi objek | Menjaga makna dan hubungan antarfitur |
| Contoh metode | SimCLR [2], MAE [3] | VIME [5], SubTab [6], SCARF [7] |

Berdasarkan Tabel II, perbedaan utama terletak pada cara model memperoleh informasi untuk menyelesaikan pretext task. Pada citra, pembelajaran memanfaatkan hubungan spasial, sedangkan pada data tabular, pembelajaran perlu mempertimbangkan hubungan antaratribut yang lebih heterogen.

Dalam konteks penelitian ini, metode Tabular SSL menjadi landasan untuk memahami berbagai operator augmentasi yang telah dikembangkan. Namun, penelitian tidak menjalankan keseluruhan metode self-supervised pretraining tersebut. Operator augmentasi diadaptasi untuk kerangka semi-supervised learning yang memanfaatkan data berlabel dan tanpa label secara bersamaan.

## C. Semi-Supervised Learning dan Pseudo-Labeling

### 1) Semi-Supervised Learning

*Semi-supervised learning* merupakan pendekatan pembelajaran yang menggunakan data berlabel dan tanpa label untuk menyelesaikan tugas prediksi. Van Engelen dan Hoos [8] menjelaskan bahwa pendekatan ini memungkinkan pemanfaatan informasi dari data tanpa label ketika jumlah data berlabel terbatas.

Dalam supervised learning, model dilatih menggunakan pasangan fitur dan target yang diketahui. Sebaliknya, pada semi-supervised learning, sebagian sampel tidak memiliki label target yang dapat digunakan secara langsung untuk menghitung supervised loss.

Untuk memanfaatkan data tersebut, berbagai metode menggunakan pseudo-labeling dan consistency regularization. Pseudo-labeling menghasilkan target sementara berdasarkan prediksi model, sedangkan consistency regularization mendorong model menghasilkan prediksi yang konsisten terhadap perubahan input.

Keberhasilan pendekatan ini bergantung pada kualitas prediksi awal dan kesesuaian transformasi yang digunakan. Prediksi yang keliru atau augmentasi yang mengubah informasi target dapat menurunkan kualitas pembelajaran.

### 2) Pseudo-Labeling

Pseudo-labeling merupakan teknik menggunakan prediksi model sebagai label sementara untuk data yang tidak memiliki anotasi. Label tersebut selanjutnya digunakan sebagai sumber supervisi tambahan dalam proses pelatihan.

Sohn et al. [9] memperkenalkan FixMatch, yang menggabungkan pseudo-labeling dengan consistency regularization. Metode ini menghasilkan prediksi pada data tanpa label yang diberi augmentasi lemah. Prediksi yang memenuhi ambang kepercayaan digunakan sebagai target ketika model memproses sampel yang sama dengan augmentasi lebih kuat.

Pada klasifikasi, pseudo-label umumnya berupa kelas yang diprediksi model. Namun, penggunaan pseudo-label perlu memperhatikan tingkat kepercayaan prediksi karena label yang tidak akurat dapat menyebabkan kesalahan selama pelatihan.

Dalam penelitian ini, prinsip pseudo-labeling digunakan pada adaptasi klasifikasi RankUp-CLS. Model menghasilkan pseudo-label kelas dari data tanpa label dan memanfaatkannya dalam pembelajaran konsistensi antara augmentasi lemah dan kuat.

### 3) Consistency Regularization

Consistency regularization merupakan strategi yang mendorong model menghasilkan prediksi yang relatif konsisten terhadap beberapa versi dari sampel yang sama. Pendekatan ini menggunakan asumsi bahwa transformasi tertentu tidak seharusnya mengubah informasi target yang relevan.

FixMatch [9] menunjukkan penerapan consistency regularization melalui hubungan antara augmentasi lemah dan kuat. Prediksi dari augmentasi lemah digunakan sebagai target pembelajaran untuk augmentasi kuat.

Namun, asumsi konsistensi dapat menjadi masalah apabila augmentasi mengubah informasi yang berkaitan dengan target. Risiko tersebut menjadi salah satu pertimbangan penting pada data tabular karena perubahan nilai fitur dapat menghasilkan kombinasi atribut yang tidak sesuai dengan data asal.

Oleh karena itu, teknik augmentasi berpotensi memengaruhi kualitas konsistensi prediksi, khususnya ketika data berlabel tersedia dalam jumlah terbatas.

## D. RankUp dan Arsitektur Tabular

### 1) RankUp untuk Semi-Supervised Regression

Huang et al. [10] memperkenalkan RankUp sebagai pendekatan semi-supervised regression yang mengadaptasi metode semi-supervised classification untuk tugas regresi. Permasalahan yang diangkat adalah metode klasifikasi seperti FixMatch tidak dapat langsung diterapkan pada regresi karena target regresi berupa nilai kontinu, bukan kelas diskrit.

RankUp menggunakan *Auxiliary Ranking Classifier* (ARC) untuk mempelajari hubungan urutan antara pasangan sampel. Dengan mengubah perbandingan nilai target menjadi persoalan klasifikasi ranking, komponen tambahan tersebut dapat memanfaatkan mekanisme semi-supervised classification.

ARC dilatih bersama objective regresi utama. Dengan demikian, model tidak hanya mempelajari nilai target, tetapi juga hubungan relatif antarobservasi. RankUp juga memperkenalkan *Regression Distribution Alignment* (RDA) sebagai komponen tambahan untuk memperbaiki distribusi pseudo-label regresi [10].

Dalam penelitian ini, RankUp digunakan sebagai kerangka pembelajaran untuk regresi kontinu dan diadaptasi pada tugas ordinal. Mekanisme ARC dipertahankan pada kedua tugas tersebut, sedangkan RDA tidak digunakan.

### 2) Adaptasi RankUp untuk Prediksi Ordinal

Prediksi ordinal merupakan tugas yang memiliki target berupa kategori dengan urutan alami. Berbeda dengan klasifikasi nominal, kesalahan prediksi pada kelas yang berdekatan tidak memiliki karakteristik yang sama dengan kesalahan pada kelas yang berjauhan.

Dalam penelitian ini, tugas ordinal diperlakukan sebagai regresi terhadap tingkat kelas yang berurutan. Model menghasilkan prediksi kontinu yang kemudian dipetakan ke level ordinal terdekat pada tahap evaluasi.

Pendekatan tersebut memungkinkan penggunaan objective regresi dan ARC pada RankUp. Namun, perlakuan kelas ordinal sebagai nilai numerik mengandung asumsi mengenai jarak antarkelas. Oleh karena itu, evaluasi perlu memperhatikan urutan target dan tidak hanya menggunakan ukuran kesalahan numerik.

Adaptasi ini merupakan bagian dari rancangan eksperimen penelitian, bukan metode ordinal yang secara khusus diperkenalkan oleh Huang et al. [10].

### 3) Adaptasi RankUp untuk Klasifikasi

RankUp asli dikembangkan untuk regresi sehingga tidak secara langsung memerlukan mekanisme yang sama ketika diterapkan pada klasifikasi nominal. Pada tugas klasifikasi, target telah berbentuk kelas diskrit dan dapat digunakan dalam pembentukan pseudo-label kelas.

Penelitian ini menggunakan adaptasi RankUp-CLS untuk klasifikasi multiclass. Pendekatan tersebut memanfaatkan pseudo-label kelas serta konsistensi prediksi antara augmentasi lemah dan kuat berdasarkan prinsip FixMatch [9].

Berbeda dengan penerapan RankUp pada regresi dan ordinal, RankUp-CLS tidak menggunakan ARC. Oleh karena itu, istilah RankUp-CLS dalam penelitian ini merujuk pada adaptasi implementasi untuk klasifikasi, bukan metode klasifikasi yang diperkenalkan dalam paper RankUp.

### 4) TabM sebagai Arsitektur Dasar

Gorishniy et al. [11] memperkenalkan TabM, sebuah arsitektur deep learning untuk data tabular berbasis *parameter-efficient ensembling*. Pendekatan ini memanfaatkan model multilayer perceptron dengan mekanisme pembagian parameter untuk menghasilkan beberapa prediksi secara efisien.

TabM dikembangkan untuk meningkatkan performa dan efisiensi model prediktif tabular. Berbeda dengan RankUp, TabM pada dasarnya merupakan arsitektur model, bukan algoritma semi-supervised learning.

Dalam penelitian ini, TabM digunakan sebagai arsitektur dasar yang dikombinasikan dengan kerangka RankUp atau adaptasi RankUp-CLS sesuai jenis tugas prediksi. Penggunaan arsitektur yang sama pada seluruh metode augmentasi bertujuan mengurangi pengaruh perbedaan model ketika membandingkan hasil eksperimen.

Dengan demikian, kontribusi utama yang dievaluasi bukan perubahan arsitektur TabM, melainkan perbedaan teknik augmentasi dalam kerangka pembelajaran yang telah ditentukan.

## E. Teknik Augmentasi dalam Penelitian

Penelitian ini membandingkan enam teknik augmentasi tabular, yaitu TabCutMix, SCARF, Range-Limited Augmentation (RLA), Gaussian Noise, Zero Out, dan TabCutMix Plus. Metode tersebut memiliki mekanisme transformasi fitur yang berbeda, mulai dari pertukaran nilai antarobservasi hingga penambahan gangguan numerik.

### 1) TabCutMix

Fang et al. [12] memperkenalkan TabCutMix untuk mengurangi memorisasi pada model difusi tabular. Teknik ini membentuk sampel hasil augmentasi dengan menukar sebagian fitur antara pasangan observasi dari kelas yang sama.

Pada penelitian asal, informasi kelas digunakan untuk menentukan pasangan sampel. Namun, dalam penelitian ini, TabCutMix diadaptasi untuk data tanpa label dengan memanfaatkan pseudo-label dalam pemilihan donor.

Khusus regresi kontinu, prediksi dikelompokkan berdasarkan kuantil untuk membantu pemilihan donor yang memiliki kelompok prediksi serupa. Pengelompokan ini hanya digunakan dalam mekanisme augmentasi dan tidak mengubah target regresi kontinu menjadi kelas diskrit.

Adaptasi tersebut memiliki keterbatasan karena pseudo-label dapat mengandung kesalahan. Selain itu, kesesuaian donor berdasarkan kelompok prediksi tidak menjamin bahwa pertukaran fitur selalu menghasilkan sampel yang valid.

### 2) SCARF

SCARF dikembangkan Bahri et al. [7] sebagai metode self-supervised contrastive learning pada data tabular. Metode tersebut menggunakan random feature corruption dengan mengganti sebagian nilai fitur untuk menghasilkan representasi sampel yang berbeda.

Pada penelitian asal, operator tersebut merupakan bagian dari proses contrastive pretraining. Dalam penelitian ini, hanya mekanisme korupsi fiturnya yang digunakan sebagai augmentasi dalam kerangka semi-supervised learning.

Nilai pengganti diambil melalui donor independen untuk setiap fitur yang terpilih. Mekanisme ini memberikan variasi nilai tanpa mengharuskan pertukaran seluruh kelompok fitur dari satu donor.

Namun, penggantian secara independen juga dapat mengubah hubungan antaratribut. Oleh karena itu, efektivitasnya perlu diuji pada tugas prediksi dan karakteristik data yang berbeda.

### 3) Range-Limited Augmentation

Lee et al. [13] memperkenalkan Range-Limited Augmentation untuk pembelajaran tabular dengan jumlah label terbatas. Pendekatan ini membatasi perubahan nilai fitur pada rentang tertentu agar augmentasi tidak menghasilkan perubahan yang terlalu jauh dari informasi sampel asal.

Metode asal dikembangkan dalam konteks contrastive learning untuk few-shot classification. Penelitian ini mengadaptasi gagasan pembatasan rentang nilai sebagai operator augmentasi pada data tanpa label.

Implementasi yang digunakan menghasilkan nilai fitur melalui pengambilan sampel pada rentang kuantil lokal. Tujuannya adalah memberikan variasi numerik dengan tetap membatasi perubahan berdasarkan karakteristik nilai fitur.

Meskipun demikian, pembatasan rentang tidak secara otomatis mempertahankan seluruh hubungan antarfitur. Pada atribut yang secara konseptual diskrit, pengambilan sampel kontinu juga dapat menghasilkan nilai yang tidak sesuai dengan tipe data aslinya.

### 4) Gaussian Noise

Gaussian Noise merupakan teknik augmentasi yang menambahkan gangguan acak berdistribusi normal pada nilai fitur numerik. Transformasi ini menghasilkan variasi lokal dengan mempertahankan sebagian besar karakteristik nilai asal ketika skala gangguan relatif kecil.

Besarnya gangguan menjadi faktor penting karena perubahan yang terlalu besar dapat menghilangkan informasi yang relevan terhadap target. Teknik ini juga tidak secara langsung mempertimbangkan korelasi antarfitur.

Dalam penelitian ini, Gaussian Noise digunakan sebagai operator perturbasi numerik. Besarnya gangguan dikendalikan melalui parameter kekuatan augmentasi yang ditentukan dalam eksperimen.

### 5) Zero Out

Zero Out merupakan teknik mengubah nilai sejumlah fitur terpilih menjadi nol. Teknik ini mengurangi informasi yang tersedia pada input sehingga model perlu mempelajari prediksi dengan sebagian fitur yang telah dimodifikasi.

Zero Out memiliki hubungan konseptual dengan masking, tetapi interpretasinya bergantung pada representasi data. Apabila fitur telah distandardisasi, nilai nol dapat merepresentasikan nilai di sekitar rata-rata fitur, bukan selalu menunjukkan informasi yang hilang.

Dalam penelitian ini, Zero Out digunakan sebagai operator sederhana untuk membandingkan pengaruh penghilangan nilai fitur dengan teknik augmentasi lain yang menghasilkan nilai pengganti.

### 6) TabCutMix Plus

TabCutMix Plus diperkenalkan Fang et al. [12] sebagai pengembangan TabCutMix yang mempertimbangkan hubungan korelasi antarfitur. Metode tersebut membentuk kelompok fitur sehingga atribut yang berkaitan dapat dipertukarkan secara bersama.

Pendekatan ini bertujuan mengurangi masalah yang dapat muncul ketika fitur yang memiliki hubungan kuat dipertukarkan secara terpisah. Dengan mempertahankan sebagian struktur hubungan fitur, sampel hasil augmentasi diharapkan lebih konsisten dengan karakteristik data.

Dalam penelitian ini, TabCutMix Plus diadaptasi untuk penggunaan data tanpa label. Pengelompokan fitur menggunakan hierarchical clustering dengan average linkage, sedangkan pemilihan donor memanfaatkan pseudo-label atau kelompok kuantil prediksi pada regresi kontinu.

Meskipun pengelompokan berbasis korelasi dapat mempertahankan sebagian hubungan antarfitur, teknik tersebut tidak menjamin bahwa seluruh kombinasi hasil pertukaran valid secara semantik.

### 7) Perbandingan Teknik Augmentasi

**TABEL III. RINGKASAN TEKNIK AUGMENTASI YANG DIGUNAKAN**

| Metode | Mekanisme utama | Sumber metode |
|---|---|---|
| TabCutMix | Pertukaran sebagian fitur antarsampel | Fang et al. [12] |
| SCARF | Random feature corruption | Bahri et al. [7] |
| RLA | Perturbasi dalam rentang fitur terbatas | Lee et al. [13] |
| Gaussian Noise | Penambahan gangguan Gaussian | Operator augmentasi numerik |
| Zero Out | Penggantian fitur terpilih dengan nol | Operator masking |
| TabCutMix Plus | Pertukaran kelompok fitur berdasarkan korelasi | Fang et al. [12] |

Tabel III menunjukkan bahwa keenam metode memiliki perbedaan dalam cara menghasilkan data hasil transformasi. TabCutMix dan TabCutMix Plus memanfaatkan informasi donor, SCARF mengganti nilai fitur secara independen, RLA membatasi nilai dalam rentang tertentu, sedangkan Gaussian Noise dan Zero Out menggunakan transformasi langsung terhadap nilai fitur.

Perbedaan mekanisme tersebut menjadi dasar perbandingan empiris pada penelitian ini. Penelitian tidak menjalankan keseluruhan metode pembelajaran yang terdapat pada paper asal, melainkan mengadaptasi operator augmentasinya ke dalam kerangka semi-supervised yang sama.

## F. Penelitian Terdahulu dan Posisi Penelitian

Penelitian mengenai pemanfaatan data tanpa label telah mengembangkan berbagai pendekatan pembelajaran representasi dan semi-supervised learning. Namun, setiap metode memiliki tujuan, mekanisme pembelajaran, serta kondisi evaluasi yang berbeda.

VIME [5], SubTab [6], dan SCARF [7] menunjukkan pengembangan pendekatan pembelajaran representasi untuk data tabular. Sementara itu, FixMatch [9] menggabungkan pseudo-labeling dan consistency regularization untuk klasifikasi, sedangkan RankUp [10] memperluas penerapan pendekatan semi-supervised classification ke dalam tugas regresi melalui ARC.

Penelitian TabM [11] berfokus pada rancangan arsitektur tabular, sedangkan TabCutMix [12] dan RLA [13] dikembangkan dalam konteks augmentasi untuk tujuan pembelajaran yang berbeda. Oleh karena itu, performa suatu operator pada penelitian asal tidak dapat langsung digeneralisasi ketika operator tersebut digunakan dalam kerangka RankUp.

**TABEL IV. RINGKASAN PENELITIAN TERDAHULU**

| Penelitian | Metode | Fokus utama | Hubungan dengan penelitian ini |
|---|---|---|---|
| Yoon et al. [5] | VIME | Mask estimation dan reconstruction | Landasan augmentasi tabular |
| Ucar et al. [6] | SubTab | Feature subsetting | Landasan pembelajaran dari subset fitur |
| Bahri et al. [7] | SCARF | Contrastive learning dengan feature corruption | Sumber operator SCARF |
| Sohn et al. [9] | FixMatch | Pseudo-labeling dan consistency regularization | Dasar adaptasi klasifikasi |
| Huang et al. [10] | RankUp | Semi-supervised regression melalui ARC | Kerangka utama pembelajaran |
| Gorishniy et al. [11] | TabM | Parameter-efficient ensembling | Arsitektur dasar |
| Fang et al. [12] | TabCutMix | Augmentasi pada model difusi tabular | Sumber TabCutMix dan TabCutMix Plus |
| Lee et al. [13] | RLA | Augmentasi untuk few-shot tabular learning | Sumber konsep augmentasi berbatas rentang |

Berdasarkan penelitian terdahulu, terdapat kebutuhan untuk mengevaluasi bagaimana berbagai operator augmentasi berperilaku ketika digunakan pada kerangka semi-supervised dan kondisi keterbatasan label yang sama. Kebutuhan tersebut merupakan fokus eksperimen penelitian ini, bukan klaim bahwa seluruh kombinasi metode tersebut belum pernah diteliti.

Penelitian ini membandingkan enam teknik augmentasi pada tiga jenis tugas prediksi, yaitu ordinal, klasifikasi multiclass, dan regresi kontinu. Untuk tugas ordinal digunakan dataset NHANES dari benchmark TOC-UCO [14], untuk klasifikasi digunakan Pen Digits [15], sedangkan untuk regresi kontinu digunakan Superconductivity [16].

Ketiga dataset tersebut memiliki karakteristik target yang berbeda. NHANES memiliki lima level ordinal, Pen Digits memiliki sepuluh kelas nominal, sedangkan Superconductivity memiliki target suhu kritis kontinu. Perbedaan ini memungkinkan evaluasi teknik augmentasi pada beberapa bentuk tugas prediksi.

Setiap metode augmentasi diuji pada fraksi data berlabel sebesar 1%, 10%, dan 30% dari train pool. Sisa data pada train pool digunakan sebagai data tanpa label. Dengan demikian, terdapat 54 kombinasi eksperimen yang mencakup tiga dataset, enam augmentasi, dan tiga fraksi label.

Perbandingan dilakukan dengan menggunakan kerangka pembelajaran dan arsitektur dasar yang telah ditentukan. Setiap kombinasi melalui optimasi kekuatan augmentasi menggunakan Optuna, kemudian dilakukan lima pelatihan akhir dengan seed berbeda. Evaluasi menggunakan Quadratic Weighted Kappa (QWK) untuk ordinal, macro-F1 untuk klasifikasi, dan Mean Squared Error (MSE) untuk regresi kontinu.

Penelitian ini bertujuan mengidentifikasi perbedaan performa keenam augmentasi pada variasi tugas dan jumlah data berlabel. Hasilnya juga dapat digunakan untuk mengamati kestabilan pelatihan berdasarkan variasi seed.

Namun, interpretasi hasil memiliki batasan. Karena rancangan eksperimen tidak menyertakan baseline supervised-only maupun tanpa augmentasi, hasil penelitian tidak dapat digunakan untuk menyimpulkan besarnya peningkatan dibandingkan kedua pendekatan tersebut. Kesimpulan dibatasi pada perbandingan relatif enam teknik augmentasi dalam konfigurasi yang diuji.

Dengan batasan tersebut, kontribusi penelitian diarahkan pada evaluasi empiris pengaruh strategi augmentasi terhadap pembelajaran semi-supervised berbasis RankUp dan TabM dalam kondisi keterbatasan label dan jenis target yang berbeda.

---

# DAFTAR PUSTAKA

[1] C. Shorten and T. M. Khoshgoftaar, “A survey on image data augmentation for deep learning,” *Journal of Big Data*, vol. 6, art. no. 60, 2019. https://doi.org/10.1186/s40537-019-0197-0

[2] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,” in *Proceedings of the 37th International Conference on Machine Learning*, vol. 119, pp. 1597–1607, 2020. https://proceedings.mlr.press/v119/chen20j.html

[3] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked Autoencoders Are Scalable Vision Learners,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 16000–16009, 2022. https://doi.org/10.1109/CVPR52688.2022.01553

[4] W.-Y. Wang, W.-W. Du, D. Xu, W. Wang, and W.-C. Peng, “A survey on self-supervised learning for non-sequential tabular data,” *Machine Learning*, vol. 114, art. no. 16, 2025. https://doi.org/10.1007/s10994-024-06674-0

[5] J. Yoon, Y. Zhang, J. Jordon, and M. van der Schaar, “VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020. https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

[6] T. Ucar, E. Hajiramezanali, and L. Edwards, “SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning,” in *Advances in Neural Information Processing Systems*, vol. 34, 2021. https://proceedings.neurips.cc/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html

[7] D. Bahri, H. Jiang, Y. Tay, and D. Metzler, “SCARF: Self-Supervised Contrastive Learning Using Random Feature Corruption,” in *International Conference on Learning Representations*, 2022. https://openreview.net/forum?id=CuV_qYkmKb3

[8] J. E. van Engelen and H. H. Hoos, “A survey on semi-supervised learning,” *Machine Learning*, vol. 109, pp. 373–440, 2020. https://doi.org/10.1007/s10994-019-05855-6

[9] K. Sohn et al., “FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020. https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html

[10] P.-Y. Huang, S.-W. Fu, and Y. Tsao, “RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/c26a8494fe31695db965ae8b7244b7c1-Abstract-Conference.html

[11] Y. Gorishniy, A. Kotelnikov, and A. Babenko, “TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling,” in *International Conference on Learning Representations*, 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html

[12] Z. Fang, Z. Jiang, H. Chen, X. Li, and J. Li, “Understanding and Mitigating Memorization in Diffusion Models for Tabular Data,” in *Proceedings of the 42nd International Conference on Machine Learning*, vol. 267, pp. 15976–16005, 2025. https://proceedings.mlr.press/v267/fang25f.html

[13] K. Lee et al., “Range-limited Augmentation for Few-shot Learning in Tabular Data with Comprehensive Benchmark,” in *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 1262–1273, 2025. https://doi.org/10.1145/3711896.3737098

[14] R. Ayllón-Gavilán, D. Guijo-Rubio, A. M. Gómez-Orellana, F. Bérchez-Moreno, V. M. Vargas-Yun, and P. A. Gutiérrez, “TOC-UCO: A comprehensive repository of tabular ordinal classification datasets,” *Neurocomputing*, vol. 684, art. no. 133528, 2026. https://doi.org/10.1016/j.neucom.2026.133528

[15] I. Alpaydin and F. Alimoglu, “Pen-Based Recognition of Handwritten Digits,” *UCI Machine Learning Repository*. https://doi.org/10.24432/C5MG6K

[16] K. Hamidieh, “Superconductivty Data,” *UCI Machine Learning Repository*, 2018. https://doi.org/10.24432/C53P47

---

## Catatan Verifikasi

- Referensi [1]–[14] mencakup artikel jurnal dan paper konferensi. Referensi [15]–[16] adalah sitasi dataset UCI, bukan artikel jurnal.
- Research gap dibatasi pada kebutuhan evaluasi komparatif, bukan klaim bahwa belum ada penelitian sebelumnya yang membandingkan operator serupa.
- Deskripsi adaptasi mengikuti rancangan eksperimen yang diberikan, belum diverifikasi terhadap seluruh implementasi kode. Narasi operator, ARC, dan pseudo-label perlu dicocokkan kembali dengan kode final.
