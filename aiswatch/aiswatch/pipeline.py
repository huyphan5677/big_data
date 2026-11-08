"""Chạy toàn bộ từ ngày X tới hôm nay (hoặc --end): xử lý dữ liệu + chấm điểm bằng mô hình đã train + xuất kết quả.

    python -m aiswatch.pipeline --start 2025-10-01            # tới hôm qua (UTC)
    python -m aiswatch.pipeline --start 2025-10-01 --end 2026-03-31

Mỗi ngày D (bỏ qua bước đã có kết quả, chạy lại an toàn):
  1. tải aisdk-D.zip → bronze (kiểm tra số dòng/MMSI/checksum) → silver (Spark, khử trùng lặp) → xoá zip, bronze
  2. tầng ngày (Spark) → quỹ đạo 10 phút (Spark) → xoá silver cũ (giữ KEEP_SILVER_DAYS ngày)
Sau đó, với mọi ngày đã có đủ 30 ngày dữ liệu trước đó: feature trượt 30 ngày + hành trình 180 ngày → chấm điểm (Spark)
→ gộp kết quả → bảng cho app (8502) + dữ liệu trang giám sát (8501).
Lưu ý: feature hành trình dùng tối đa 180 ngày lịch sử; nên chọn X sớm hơn ngày muốn xem khoảng 6 tháng để điểm ổn định nhất.
"""
import argparse, json, os, shutil, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
from .config import DATA, SPARK_CORES, SPARK_MEM, KEEP_SILVER_DAYS, WINDOW_DAYS, save_json, now, log

D = lambda p: DATA/'data'/p


def download(day):
    """aisdk-YYYY-MM-DD.zip từ kho Cục Hàng hải Đan Mạch; kiểm tra đủ byte theo Content-Length; thử tối đa 3 lần."""
    path = D('tmp')/f'{day}.zip'; path.parent.mkdir(parents=True, exist_ok=True)
    keys = [f'aisdk-{day}.zip', f'{day[:4]}/aisdk-{day}.zip']
    for attempt in range(1, 4):
        for key in keys:
            try:
                with requests.get('http://aisdata.ais.dk/'+key, stream=True, timeout=(30, 120)) as r:
                    if r.status_code == 404: continue
                    r.raise_for_status(); size = int(r.headers.get('Content-Length', 0)); got = 0
                    with path.open('wb') as f:
                        for chunk in r.iter_content(1 << 20): got += len(chunk); f.write(chunk)
                assert size == 0 or got == size, f'thiếu byte {got}/{size}'
                return path
            except Exception as exc:
                log('download', day, key, f'lần {attempt}:', type(exc).__name__, str(exc)[:120])
        time.sleep(30*attempt)
    raise RuntimeError(f'{day}: không tải được (kho có thể chưa có ngày này)')


def ingest_day(spark, day, high, ports, need_tracks=True):
    from . import bronze, silver, features_day, tracks, identity, labels
    if (D('features_day')/f'date={day}'/'_meta.json').exists() and (not need_tracks or (D('app/tracks')/f'date={day}.parquet').exists()):
        return 'đã có'
    t = time.monotonic()
    if not (D('silver')/f'date={day}'/'_verification.json').exists():
        z = download(day)
        shutil.rmtree(D('bronze')/f'date={day}', ignore_errors=True); (D('vessel_day')/f'date={day}.parquet').unlink(missing_ok=True)
        info = bronze.build(spark, z, day)
        shutil.rmtree(D('silver')/f'date={day}', ignore_errors=True)
        silver.build(spark, day, info)
        z.unlink(missing_ok=True); shutil.rmtree(D('bronze')/f'date={day}', ignore_errors=True)
    if not (D('features_day')/f'date={day}'/'_meta.json').exists():
        features_day.build(spark, day, high, ports)
    if need_tracks:
        month = day[:4]+day[5:7]
        m = identity.month_map(month, labels.load())
        tracks.extract(spark, day, set(m.loc[m.is_tanker, 'MMSI']))
    return f'{time.monotonic()-t:.0f}s'


def cleanup_silver(end):
    keep = {str((pd.Timestamp(end) - pd.Timedelta(days=i)).date()) for i in range(KEEP_SILVER_DAYS)}
    for p in D('silver').glob('date=*'):
        if p.name[5:] not in keep: shutil.rmtree(p, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', required=True, help='ngày bắt đầu xử lý dữ liệu (YYYY-MM-DD)')
    ap.add_argument('--score-from', help='chỉ chấm điểm + hiển thị từ ngày này (mặc định: start + 29 ngày); dữ liệu trước đó dùng làm lịch sử')
    ap.add_argument('--end', default=str((datetime.now(timezone.utc) - timedelta(days=1)).date()), help='mặc định: hôm qua (UTC)')
    ap.add_argument('--mode', default=os.environ.get('AISWATCH_MODE', 'live'), choices=['live', 'demo'],
                    help='demo: hiện thêm "đáp án" (ngày bị cấm sau này) trên giao diện; live: chỉ tình trạng tới hiện tại')
    ap.add_argument('--no-label-download', action='store_true', help='dùng bản nhãn đã có, không tải lại')
    a = ap.parse_args()
    os.environ['PYSPARK_PYTHON'] = sys.executable
    from . import labels, rolling, predict, publish
    from .silver import session
    from .features_day import maps
    days = [str(x.date()) for x in pd.date_range(a.start, a.end)]
    score_from = a.score_from or str((pd.Timestamp(a.start) + pd.Timedelta(days=WINDOW_DAYS-1)).date())
    status = dict(started=now(), start=a.start, end=a.end, mode=a.mode)
    log('nhãn EU/UK…'); labels.refresh(fetch=not a.no_label_download); L = labels.load()
    spark = session(DATA, app_name='aiswatch', cores=SPARK_CORES, driver_memory=SPARK_MEM)
    spark.conf.set('spark.sql.shuffle.partitions', '64')
    try:
        high, ports, _ = maps(spark)
        for day in days:
            log('xử lý', day, ingest_day(spark, day, high, ports, need_tracks=day >= score_from))
            spark._jvm.java.lang.System.gc()
        cleanup_silver(a.end)
        have = [d for d in days if (D('features_day')/f'date={d}'/'_meta.json').exists()]
        first = min(have)
        score_days = [d for d in have if d >= score_from and all((D('features_day')/f'date={x.date()}'/'_meta.json').exists()
                                             for x in pd.date_range(pd.Timestamp(d) - pd.Timedelta(days=WINDOW_DAYS-1), d))]
        log('chấm điểm', len(score_days), 'ngày', score_days[:1], '…', score_days[-1:])
        V, N = rolling.load_daily(first, a.end, L) if score_days else (None, None)
        for day in score_days:
            if (D('predict')/f'date={day}').exists() and day != score_days[-1]: continue
            rolling.build(day, L, V, N, first)
            r = predict.score_day(spark, day)
            log('điểm', day, 'tàu', len(r), 'danh sách theo dõi', int((r.rank_watch <= 30).sum()))
    finally:
        spark.stop()
    if not any(D('predict').glob('date=*')):
        log('chưa có ngày nào đủ 30 ngày dữ liệu để chấm; chạy tiếp các ngày sau'); return
    P = predict.compact()
    Vt = publish.app_tables(P, L, sorted(P.date.unique()))
    publish.web(P, Vt, a.mode)
    status.update(finished=now(), scored_days=len(P.date.unique()), last_day=str(P.date.max()), data_start=first)
    save_json(DATA/'data/status.json', status); log('xong', status)


if __name__ == '__main__':
    main()
