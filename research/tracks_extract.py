"""Trích quỹ đạo tàu dầu (điểm cuối mỗi ô 10 phút) từ silver cho cửa sổ phát lại của trang giám sát.

Chỉ lấy MMSI của các tàu dầu đã được chấm điểm. Spark đọc silver từng ngày, lọc cờ lỗi như features_day.
Đầu ra: data/app/tracks/date=D.parquet (MMSI, bin, t, lat, lon, sog, cog, draught, dest).
"""
import argparse, json, os, sys, time
import pandas as pd
from train_model import ROOT
from features_day import FLAGS

BIN = 600
OUT = ROOT/'data/app/tracks'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--start', default='2025-10-20'); ap.add_argument('--end', default='2025-11-03'); a = ap.parse_args()
    os.environ['PYSPARK_PYTHON'] = sys.executable
    from silver import session
    from pyspark.sql import functions as F
    spark = session(app_name='AIS-tracks', cores=2, driver_memory='3g'); spark.conf.set('spark.sql.shuffle.partitions', '16')
    V = pd.read_parquet(ROOT/'data/app/vessels.parquet')
    scored = set(pd.read_parquet(ROOT/'data/predict_all.parquet', columns=['vessel_id']).vessel_id)
    mm = sorted({m for ms, vid in zip(V.mmsis, V.vessel_id) if vid in scored for m in ms.split(',')})
    keep = spark.createDataFrame([(m,) for m in mm], 'MMSI string')
    OUT.mkdir(parents=True, exist_ok=True)
    for D in pd.date_range(a.start, a.end):
        day = str(D.date()); target = OUT/f'date={day}.parquet'
        if target.exists(): continue
        t0 = time.monotonic()
        raw = spark.read.parquet(str(ROOT/'data/silver'/f'date={day}'))
        df = (raw.filter(~F.col(FLAGS[0]) & ~F.col(FLAGS[1]) & ~F.col(FLAGS[2])).join(F.broadcast(keep), 'MMSI')
              .withColumn('t', F.col('ts').cast('long')).withColumn('bin', F.floor(F.col('t')/BIN))
              .groupBy('MMSI', 'bin').agg(F.max(F.struct('t', 'Latitude', 'Longitude', 'SOG', 'COG', 'Draught', 'Destination')).alias('p'))
              .select('MMSI', 'bin', 'p.t', F.col('p.Latitude').alias('lat'), F.col('p.Longitude').alias('lon'), F.col('p.SOG').alias('sog'),
                      F.col('p.COG').alias('cog'), F.col('p.Draught').alias('draught'), F.col('p.Destination').alias('dest')))
        stage = OUT/f'.stage-{day}'
        df.coalesce(1).write.mode('overwrite').parquet(str(stage))
        x = pd.read_parquet(stage); x.to_parquet(target, index=False)
        import shutil; shutil.rmtree(stage)
        print(json.dumps(dict(date=day, rows=len(x), mmsi=int(x.MMSI.nunique()), s=round(time.monotonic()-t0, 1))), flush=True)
    spark.stop()


if __name__ == '__main__':
    main()
