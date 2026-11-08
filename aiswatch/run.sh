#!/usr/bin/env bash
# Xử lý dữ liệu từ START_DATE tới hôm qua + chấm điểm bằng mô hình đã train + xuất kết quả.
#   ./run.sh                      dùng START_DATE trong config.env
#   ./run.sh --start 2026-01-01   ghi đè; mọi tham số khác truyền thẳng cho pipeline (xem --help)
#   ./run.sh demo <bundle.tar.gz> chế độ demo: giải nén kết quả có sẵn (không cần tải/Spark)
set -euo pipefail
cd "$(dirname "$0")"
set -a; source config.env; set +a
export PYTHONPATH="$PWD:$PWD/shims"
mkdir -p "$AISWATCH_DATA/logs"
if [ "${1:-}" = demo ]; then
  tar -xzf "$2" -C "$AISWATCH_DATA"; echo "Đã giải nén kết quả demo vào $AISWATCH_DATA"; exit 0
fi
ARGS=(--start "$START_DATE" --mode "$AISWATCH_MODE")
[ -n "${SCORE_FROM:-}" ] && ARGS+=(--score-from "$SCORE_FROM")
.venv/bin/python -m aiswatch.pipeline "${ARGS[@]}" "$@" 2>&1 | tee -a "$AISWATCH_DATA/logs/pipeline.log"
