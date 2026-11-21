#!/usr/bin/env bash
# Mengunduh kode resmi SSR-RCUS ke paper_repo/ pada commit yang dipakai eksperimen. Kode ini tidak diubah sama sekali;
# semua adaptasi tabular ada di tabular_rcus.py. paper_repo/ tidak ikut git (kode dan lisensi MIT milik penulis aslinya).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=e79e53f7555d188435869d24e700ec7fe45cca33
[ -d paper_repo/.git ] || git clone -q https://github.com/changchunli/SSR-RCUS.git paper_repo
git -C paper_repo checkout -q "$COMMIT"
git -C paper_repo diff --quiet || { echo "paper_repo berubah dari commit resmi; batalkan perubahannya dulu" >&2; exit 1; }
echo "ok  SSR-RCUS @ ${COMMIT:0:7}"
