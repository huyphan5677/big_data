"""Reusable PySpark daily Silver: exact typed-field dedup, flags, reconciliation."""
import json
import shutil
import time
from pathlib import Path
from common import ROOT,save_json,now,configure_utc
from ais_schema import COLUMNS


def build(spark,date,bronze_info,root=ROOT):
    from pyspark.sql import functions as F
    root=Path(root);start=time.monotonic()
    source=root/'data/bronze'/('date='+date)
    stage=root/'data/silver'/('.staging-'+date);target=root/'data/silver'/('date='+date)
    shutil.rmtree(stage,ignore_errors=True)
    raw=spark.read.parquet(str(source))
    assert set(raw.columns)==set(COLUMNS+['ts'])
    grouped=raw.groupBy(*COLUMNS).agg(F.count('*').alias('n_copies'),F.min('ts').alias('ts'))
    lat,lon=F.col('Latitude'),F.col('Longitude')
    good=lat.between(-90,90)&lon.between(-180,180)&~F.isnan(lat)&~F.isnan(lon)&~((lat==0)&(lon==0))
    m=F.col('MMSI')
    good_mmsi=m.rlike('^[0-9]{9}$')&F.substring(m,1,3).cast('int').between(201,775)
    grouped=grouped.withColumns({'flag_bad_pos':~F.coalesce(good,F.lit(False)),
      'flag_not_class_a':F.coalesce(F.col('Type of mobile')!='Class A',F.lit(True)),
      'flag_bad_mmsi':~F.coalesce(good_mmsi,F.lit(False))})
    (grouped.repartition(64,'MMSI').sortWithinPartitions('MMSI','ts')
       .write.mode('overwrite').option('compression','zstd').parquet(str(stage)))
    result=spark.read.parquet(str(stage))
    flags=['flag_bad_pos','flag_not_class_a','flag_bad_mmsi']
    agg=result.agg(F.sum('n_copies').alias('rows_bronze'),F.count('*').alias('rows_dedup'),
      (F.countDistinct('MMSI')+F.max(F.col('MMSI').isNull().cast('int'))).alias('n_mmsi'),
      F.sum(F.col('MMSI').cast('long')*F.col('n_copies')).alias('mmsi_sum'),
      F.sum(F.col('ts').isNull().cast('long')).alias('missing_ts'),
      *[F.sum(F.when(F.col(f),F.col('n_copies')).otherwise(0)).alias('n_'+f) for f in flags]).first().asDict()
    assert agg['rows_bronze']==bronze_info['rows_bronze'],'Silver n_copies mismatch'
    assert agg['n_mmsi']==bronze_info['n_mmsi'],'Silver distinct MMSI mismatch'
    assert int(agg['mmsi_sum'] or 0)==bronze_info['mmsi_sum'],'Silver MMSI checksum mismatch'
    assert agg['missing_ts']==0,'Silver missing timestamps'
    info=dict(date=date,format_version=3,**agg,bytes=sum(p.stat().st_size for p in stage.glob('part-*.parquet')),
      silver_s=round(time.monotonic()-start,3),flag_count_unit='original rows weighted by n_copies',
      physical_layout='64 MMSI hash partitions, sorted MMSI/ts within files',verified_at=now())
    save_json(stage/'_verification.json',info)
    assert not target.exists(),'Verify existing Silver before replacing'
    stage.rename(target)
    return info


def session(root=ROOT,app_name='AIS-bronze-silver',cores=4,driver_memory='4g'):
    import os
    configure_utc()
    os.environ.setdefault('SPARK_LOCAL_IP','127.0.0.1')
    os.environ['SPARK_LOCAL_DIRS']=str(Path(root)/'data/spark_tmp')
    from pyspark.sql import SparkSession
    spark=(SparkSession.builder.master(f'local[{cores}]').appName(app_name)
      .config('spark.driver.memory',driver_memory).config('spark.driver.maxResultSize','128m')
      .config('spark.sql.shuffle.partitions','256').config('spark.memory.fraction','.6')
      .config('spark.sql.files.maxPartitionBytes',str(32*1024**2))
      .config('spark.sql.session.timeZone','UTC').config('spark.ui.enabled','false')
      .config('spark.ui.showConsoleProgress','false')
      .config('spark.sql.ui.retainedExecutions','10').config('spark.ui.retainedJobs','10')
      .config('spark.ui.retainedStages','20').config('spark.sql.adaptive.enabled','true').getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    from spark_status import start
    start(spark,root)
    return spark
