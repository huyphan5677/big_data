#!/bin/sh
# Tính lại feature trượt (kèm hành trình) -> chấm điểm Spark -> gộp file cho app. Chạy tuần tự.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
rm -rf data/features_rolling data/predict
$PY features_rolling.py
PYTHONPATH="$PWD/.deps:$PWD/shims" PYTHONWARNINGS=ignore $PY predict_daily.py
$PY compact_predict.py
echo "$(date -Is) xong"
