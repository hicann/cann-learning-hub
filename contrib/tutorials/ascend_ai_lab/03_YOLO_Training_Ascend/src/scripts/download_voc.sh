#!/usr/bin/env bash
set -euo pipefail

ROOT="${VOC_ROOT:-/mnt/workspace/datasets/voc}"
mkdir -p "${ROOT}"
cd "${ROOT}"

download() {
  local url="$1"
  local file="$2"
  if [ -f "${file}" ]; then
    echo "skip existing ${file}"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -c "${url}" -O "${file}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --retry 3 "${url}" -o "${file}"
  else
    echo "Neither wget nor curl was found. Please upload ${file} manually."
    exit 1
  fi
}

extract() {
  local file="$1"
  if command -v tar >/dev/null 2>&1; then
    tar -xf "${file}" -C "${ROOT}"
  else
    echo "tar was not found. Please extract ${file} under ${ROOT} manually."
    exit 1
  fi
}

download "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" "VOCtrainval_06-Nov-2007.tar"
download "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" "VOCtest_06-Nov-2007.tar"
download "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" "VOCtrainval_11-May-2012.tar"

extract "VOCtrainval_06-Nov-2007.tar"
extract "VOCtest_06-Nov-2007.tar"
extract "VOCtrainval_11-May-2012.tar"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/prepare_voc_yolo.py" --root "${ROOT}"
