#!/usr/bin/env bash
# Menyusun ulang gambar (PDF vektor + PNG 300 dpi) dari file .tex.
# Semua:  ./bangun.sh        Satu gambar:  ./bangun.sh cosine_skema
set -euo pipefail
cd "$(dirname "$0")"
for tex in */${1:-*}.tex; do
  dir=$(dirname "$tex"); name=$(basename "$tex" .tex)
  (cd "$dir" && pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" >/dev/null \
    && pdftoppm -png -r 300 -singlefile "$name.pdf" "$name" && rm -f "$name.aux" "$name.log")
  echo "ok  $dir/$name"
done
