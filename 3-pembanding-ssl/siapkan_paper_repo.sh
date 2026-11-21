#!/usr/bin/env bash
# Mengunduh kode paper pihak ketiga ke experiments/<metode>/paper_repo pada commit yang dipakai di eksperimen,
# lalu menerapkan perubahan lokal dari paper_repo_patch/. Folder paper_repo tidak ikut git (kode dan lisensi milik penulis aslinya).
set -euo pipefail
cd "$(dirname "$0")"

siapkan() {  # nama  url  commit
  local dir="experiments/$1/paper_repo" patch="$PWD/paper_repo_patch/$1.patch"
  [ -d "$dir/.git" ] || git clone -q "$2" "$dir"
  git -C "$dir" checkout -q "$3"
  if [ -f "$patch" ]; then
    if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then echo "$1: patch sudah terpasang"
    else git -C "$dir" apply "$patch" && echo "$1: patch dipasang"; fi
  fi
  echo "ok  $1 @ ${3:0:7}"
}

siapkan vime  https://github.com/jsyoon0823/VIME.git  996c58cf4c570061b30c38ecf2a754a9af85aafd
siapkan stunt https://github.com/jaehyun513/STUNT.git  e860675d0e390dba5f12eb9fd7bdcdd8d379f012
siapkan sslae https://github.com/sintija-s/sslae.git  ddbef98d27dd13e699f425352695175b48c4ad5e
siapkan gftab https://github.com/Yoontae6719/Geodesic-Flow-Kernels-for-Semi-Supervised-Learning-on-Mixed-Variable-Tabular-Dataset.git  3f65da17ee8edd089a265b60b6bcaf79f94cd35b
