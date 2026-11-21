# Autoresearch augmentasi RankUp

Folder ini mengadaptasi siklus [karpathy/autoresearch](https://github.com/karpathy/autoresearch): agen membuat hipotesis, mengubah `candidate.py`, menguji kandidat pada protokol tetap, mencatat skor, kemudian memutuskan keep/discard. Sistem ini **bukan** clone upstream yang melakukan optimasi LLM nanochat, dan bukan agen yang membuat hipotesis sendiri setelah sesi Codex selesai. Berkas `program.md` menjelaskan batas dan aturan.

`candidate.py` adalah kandidat yang sedang diuji; `snapshots/round*.py` menyimpan versi putaran terdahulu. `runner.py` memakai loop RankUp yang sama dengan baseline, `modal_ar.py` mengirim job ke Modal, dan `score.py` memeriksa sembilan kondisi serta checksum input sebelum mengeluarkan keputusan. Hasil ada di `../hasil/autoresearch/` dan `../hasil/autoresearch-round*/`. Tidak ada checkpoint yang tersimpan pada pilot ini.

Perintah dari folder `4-augmentasi-input`, setelah kandidat dipilih dan diuji:

```bash
uv run modal run --profile lagi usulan/autoresearch/modal_ar.py --action check
uv run modal run --detach --profile lagi usulan/autoresearch/modal_ar.py --action run --run-id autoresearch-round-N --parallel 2
uv run modal volume get experiment-ta-results autoresearch-round-N usulan/hasil --profile lagi --force
uv run python usulan/autoresearch/score.py autoresearch-round-N
```

Gunakan run ID baru untuk tiap perubahan agar hasil berbeda tidak bercampur. Perintah kedua mengantrekan 9 job; tunggu semua job selesai sebelum mengambil hasil. `--detach` mempertahankan job Modal, bukan proses agen yang akan menemukan kandidat baru sendiri. Jangan jalankan banyak putaran tanpa mengendalikan biaya. Validation lama telah dipakai berulang dan test historis sudah terekspos; keputusan keep/discard satu seed hanya screening dan butuh split/holdout serta seed tambahan sebelum klaim penelitian.
