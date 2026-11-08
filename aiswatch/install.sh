#!/usr/bin/env bash
# Cài môi trường: venv + thư viện. Không train gì cả.
set -euo pipefail
cd "$(dirname "$0")"
command -v java >/dev/null || { echo "Thiếu Java 17+ (vd: sudo apt install openjdk-17-jre-headless)"; exit 1; }
java -version 2>&1 | head -1
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), "cần Python >= 3.10"'
$PY -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
[ -f config.env ] || cp config.env.example config.env
echo "Xong. Sửa config.env rồi chạy ./run.sh (xử lý + chấm điểm) và ./serve.sh (2 giao diện)."
