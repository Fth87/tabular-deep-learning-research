#!/usr/bin/env bash
# Membuat kaggle_data.zip (hanya split data: label_01/, label_10/, label_30/) untuk Kaggle Dataset. Kode ada di notebook.
set -euo pipefail
cd "$(dirname "$0")/../5-augmentasi-latent/data"
rm -f ../../7-tabpfn-latent/kaggle_data.zip
zip -qr ../../7-tabpfn-latent/kaggle_data.zip label_01 label_10 label_30 -x '*__pycache__*'
echo "ok  $(du -h ../../7-tabpfn-latent/kaggle_data.zip | cut -f1)  7-tabpfn-latent/kaggle_data.zip"
