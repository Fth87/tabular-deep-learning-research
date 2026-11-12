# Template IEEE lengkap dengan Bab II

Gunakan **`template_lengkap.tex`** sebagai berkas utama dan `template_lengkap.pdf` sebagai hasil kompilasinya. Ini adalah salinan template konferensi IEEE lengkap: judul, enam contoh penulis, abstrak, kata kunci, pendahuluan, acknowledgment, referensi contoh, dan teks panduan template tetap ada. Section II template diganti dengan Bab II penelitian dari `bab_ii_isi.tex`.

Referensi Bab II ditambahkan ke daftar pustaka melalui `referensi_bab_ii_items.tex`; 11 referensi contoh template tetap dipertahankan. Nomor sitasi contoh otomatis bergeser karena referensi Bab II muncul lebih dahulu. Beberapa paket LaTeX ditambahkan untuk tabel dan URL. Berkas `IEEEtran.cls` dan `template_asli/` berasal dari template awal yang disalin; Markdown sumber tidak diubah.

Kompilasi dari folder ini:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error template_lengkap.tex
```

`bab_ii.tex` dan `bab_ii.pdf` adalah versi lama yang hanya menampilkan cuplikan Bab II. Keduanya berbagi `bab_ii_isi.tex` dan `referensi_bab_ii_items.tex` dengan template lengkap.

Judul, nama penulis, afiliasi, abstrak, pendahuluan, acknowledgment, contoh referensi dan teks petunjuk merah adalah **teks bawaan template** yang sengaja dipertahankan sesuai permintaan. Ganti atau hapus semua teks contoh sebelum mengirim paper; Bab II saja belum menjadikan naskah lengkap. Ikuti juga ketentuan bahasa dan batas halaman konferensi tujuan.

Penulis referensi dataset Pen Digits pada salinan LaTeX dikoreksi dari I. Alpaydin menjadi E. Alpaydin dan tahun 1996 ditambahkan sesuai [metadata UCI](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits). Referensi lain ditranskripsi dari sumber, belum seluruh metadata dan klaim diverifikasi. Catatan sumber ada di `catatan_verifikasi.md`.

Revisi terbaru merapikan empat tabel menjadi satu kolom dan merampingkan uraian Bab II. Rincian perubahan dan keterbatasannya ada di `CATATAN_REVISI.md`; versi sebelum revisi ada di `arsip_sebelum_revisi/`.
