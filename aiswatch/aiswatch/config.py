"""Cấu hình đọc từ biến môi trường (xem config.env.example). Mọi dữ liệu nằm dưới AISWATCH_DATA."""
import json, os, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

os.environ['TZ'] = 'UTC'; time.tzset()                       # PySpark + timestamp đều theo UTC

PKG = Path(__file__).resolve().parent.parent                  # thư mục gói (model/, maps/, web/, app/)
DATA = Path(os.environ.get('AISWATCH_DATA', PKG)).resolve()            # dữ liệu nằm ở DATA/data/...
MODEL = PKG/'model'
MAPS = PKG/'maps'
SPARK_CORES = int(os.environ.get('SPARK_CORES', '4'))
SPARK_MEM = os.environ.get('SPARK_MEM', '4g')
KEEP_SILVER_DAYS = int(os.environ.get('KEEP_SILVER_DAYS', '1'))   # silver chỉ cần cho tới khi trích xong quỹ đạo
WATCH_K = int(os.environ.get('WATCH_K', '30'))
WATCH_DAYS = 3
WINDOW_DAYS = 30
VOYAGE_DAYS = 180
LABEL_SOURCES = ['eu_fsf', 'eu_sanctions_map', 'eu_journal_sanctions', 'gb_fcdo_sanctions']


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        tmp = Path(f.name); f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def log(*a):
    print(f'[{now()}]', *a, flush=True)
