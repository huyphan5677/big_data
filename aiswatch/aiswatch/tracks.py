"""Quỹ đạo 10 phút (điểm cuối mỗi ô) của các tàu dầu trong một ngày, từ silver — cho bản đồ phát lại và luật theo phút."""
import shutil
import pandas as pd
from .config import DATA
from .features_day import FLAGS

BIN = 600
OUT = DATA/'data/app/tracks'


def extract(spark, day, mmsis):
    from pyspark.sql import functions as F
    OUT.mkdir(parents=True, exist_ok=True)
    keep = spark.createDataFrame([(m,) for m in sorted(mmsis)], 'MMSI string')
    raw = spark.read.parquet(str(DATA/'data/silver'/f'date={day}'))
    df = (raw.filter(~F.col(FLAGS[0]) & ~F.col(FLAGS[1]) & ~F.col(FLAGS[2])).join(F.broadcast(keep), 'MMSI')
          .withColumn('t', F.col('ts').cast('long')).withColumn('bin', F.floor(F.col('t')/BIN))
          .groupBy('MMSI', 'bin').agg(F.max(F.struct('t', 'Latitude', 'Longitude', 'SOG', 'COG', 'Draught', 'Destination')).alias('p'))
          .select('MMSI', 'bin', 'p.t', F.col('p.Latitude').alias('lat'), F.col('p.Longitude').alias('lon'), F.col('p.SOG').alias('sog'),
                  F.col('p.COG').alias('cog'), F.col('p.Draught').alias('draught'), F.col('p.Destination').alias('dest')))
    stage = OUT/f'.stage-{day}'
    df.coalesce(1).write.mode('overwrite').parquet(str(stage))
    x = pd.read_parquet(stage); x.to_parquet(OUT/f'date={day}.parquet', index=False); shutil.rmtree(stage)
    return len(x)
