#!/usr/bin/env bash
# Chạy 2 giao diện: viz (trang giám sát, tĩnh) và app (kết quả mô hình, Streamlit).
#   ./serve.sh viz | ./serve.sh app | ./serve.sh   (cả hai, nền trước)
set -euo pipefail
cd "$(dirname "$0")"
set -a; source config.env; set +a
export PYTHONPATH="$PWD:$PWD/shims"
viz() { exec .venv/bin/python -m http.server "$PORT_VIZ" --bind "$BIND" --directory "$AISWATCH_DATA/web"; }
app() { exec .venv/bin/python -m streamlit run app/app.py --global.developmentMode false --server.address "$BIND" \
        --server.port "$PORT_APP" --server.headless true --browser.gatherUsageStats false; }
case "${1:-both}" in
  viz) viz ;; app) app ;;
  both) (viz) & (app) & wait ;;
esac
