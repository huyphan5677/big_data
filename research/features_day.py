"""Tầng NGÀY của đặc trưng: silver 1 ngày -> các bảng nhỏ theo MMSI, chỉ chứa đại lượng CỘNG ĐƯỢC.

Tầng tháng / cửa sổ lxm (và _vs_) cộng các bảng này; tỉ lệ chỉ chia ở tầng cuối.
- Chỉ dùng dòng sạch (bỏ flag_bad_pos / flag_bad_mmsi / flag_not_class_a).
- Bản đồ cảng và vùng phủ sóng cố định, lấy từ 04/2025 (data/gate2_maps/*=202504).
- Khoảng trống vắt qua nửa đêm: tầng tháng nối từ last_* ngày trước và first_* ngày sau.
- Gặp gỡ: chỉ lưu MMSI đối tác; nhãn "đã bị cấm" nối ở tầng snapshot (tránh rò rỉ).
Đầu ra: data/features_day/date=D/{vessel,values,encounters,hourly}.parquet + _meta.json.
"""
import argparse
import json
import shutil
import time
from common import ROOT,now,save_json,configure_utc

OUT=ROOT/'data/features_day'
STOP=ROOT/'logs/stop_features'
FLAGS=['flag_bad_pos','flag_bad_mmsi','flag_not_class_a']
# Cổng eo biển: (tên, trục cắt, giá trị, khoảng toạ độ còn lại, chiều "ra khỏi Baltic").
GATES=[('greatbelt','Latitude',55.34,(10.85,11.25),'up'),
       ('oresund','Latitude',55.98,(12.55,12.80),'up'),  # nam tuyến phà Helsingør–Helsingborg
       ('skagen','Longitude',10.65,(57.75,58.60),'down')]
RU_DEST='PRIMORSK|USTLUGA|VYSOTSK|STPETERSBURG|SAINTPETERSBURG|RULED|RUULU|RUVYG|RUPRI'
VAGUE_DEST='^(FORORDERS|ORDERS|TBA|TBN|NA|NONE|UNKNOWN)$'
VALUE_FIELDS=['Ship type','Cargo type','Name','Callsign','IMO','Destination','Length','Width','dims']


def distance_nm(F,lat1,lon1,lat2,lon2):
    a=(F.sin(F.radians(lat2-lat1)/2)**2+F.cos(F.radians(lat1))*F.cos(F.radians(lat2))*F.sin(F.radians(lon2-lon1)/2)**2)
    return 3440.065*2*F.asin(F.sqrt(F.least(F.lit(1.),F.greatest(F.lit(0.),a))))


def maps(spark):
    """Ô phủ sóng cao và ô cảng, cố định từ 04/2025 (cùng ngưỡng với gate2_features cũ)."""
    import pandas as pd
    cov=pd.read_parquet(ROOT/'data/gate2_maps/coverage=202504')
    q=float(cov.density.quantile(.75))
    high=cov[(cov.density>=q)&(cov.ships>=20)&(cov.hours>=30*24*.8)][['cx','cy']].astype('int64')
    ports=pd.read_parquet(ROOT/'data/gate2_maps/ports=202504')[['px','py']].astype('int64')
    df=lambda d,flag:spark.createDataFrame([tuple(int(x) for x in r)+(True,) for r in d.itertuples(index=False)],
                                            f'{d.columns[0]} long, {d.columns[1]} long, {flag} boolean').cache()
    return (df(high,'high'),df(ports,'port'),
            dict(high_cells=len(high),port_cells=len(ports),high_density_p75=q,source='gate2_maps 202504'))


def build(spark,date,high,ports):
    from pyspark.sql import functions as F, Window
    start=time.monotonic()
    raw=spark.read.parquet(str(ROOT/'data/silver'/('date='+date)))
    clean=(raw.filter(~F.col(FLAGS[0])&~F.col(FLAGS[1])&~F.col(FLAGS[2]))
      .withColumn('t',F.col('ts').cast('long'))
      .withColumn('sog',F.when(F.col('SOG').between(0,102.2),F.col('SOG')))
      .withColumn('dest',F.regexp_replace(F.upper(F.coalesce('Destination',F.lit(''))),'[^A-Z0-9]',''))
      .withColumn('cx',F.floor(F.col('Latitude')/.05)).withColumn('cy',F.floor(F.col('Longitude')/.1))
      .join(F.broadcast(high),['cx','cy'],'left').withColumn('high',F.coalesce('high',F.lit(False))))

    # --- 1 lượt window theo MMSI: khoảng trống, cú nhảy, quãng đường, qua eo biển, đếm theo điểm ---
    w=Window.partitionBy('MMSI').orderBy('t','Latitude','Longitude','sog')
    tr=clean
    for c in ['t','Latitude','Longitude','high']:tr=tr.withColumn('p_'+c,F.lag(c).over(w))
    dist=distance_nm(F,F.col('p_Latitude'),F.col('p_Longitude'),F.col('Latitude'),F.col('Longitude'))
    tr=(tr.withColumn('dt',F.col('t')-F.col('p_t')).withColumn('dist',dist)
      .withColumn('jump',(F.col('dist')>0)&((F.col('dt')==0)|(F.col('dist')/F.col('dt')*3600>50)))
      .withColumn('seg',F.col('dt')>0)
      .withColumn('cov',F.col('high')&F.col('p_high')))
    ok=F.col('seg')&~F.col('jump')&(F.col('dt')<=6*3600)
    implied=F.col('dist')/F.col('dt')*3600
    n=lambda cond:F.sum(F.when(cond,1).otherwise(0))
    agg=[F.sum('n_copies').alias('n_msgs'),F.count('*').alias('n_rows'),
         F.min('t').alias('first_t'),F.max('t').alias('last_t'),
         F.min_by('Latitude','t').alias('first_lat'),F.min_by('Longitude','t').alias('first_lon'),F.min_by('high','t').alias('first_high'),
         F.max_by('Latitude','t').alias('last_lat'),F.max_by('Longitude','t').alias('last_lon'),F.max_by('high','t').alias('last_high'),
         n(F.col('seg')).alias('n_seg'),n(F.col('jump')).alias('n_jump50'),
         F.sum(F.when(ok,F.col('dist')).otherwise(0.)).alias('distance_nm'),
         F.max(F.when(F.col('seg'),F.col('dt'))).alias('max_gap_s'),
         F.sum('sog').alias('sog_sum'),F.count('sog').alias('sog_n'),
         n(F.col('Heading').isNull()).alias('n_heading_null'),n(F.col('ROT').isNull()).alias('n_rot_null'),
         F.min(F.when(F.col('Draught').between(.1,25.5),F.col('Draught'))).alias('draught_min'),
         F.max(F.when(F.col('Draught').between(.1,25.5),F.col('Draught'))).alias('draught_max'),
         F.min_by(F.when(F.col('Draught').between(.1,25.5),F.col('Draught')),F.when(F.col('Draught').between(.1,25.5),F.col('t'))).alias('draught_first'),
         F.max_by(F.when(F.col('Draught').between(.1,25.5),F.col('Draught')),F.when(F.col('Draught').between(.1,25.5),F.col('t'))).alias('draught_last'),
         n(F.col('dest').rlike(RU_DEST)).alias('n_dest_ru'),n(F.col('dest').rlike(VAGUE_DEST)).alias('n_dest_vague'),n(F.col('dest')=='').alias('n_dest_empty'),
         n(F.col('Navigational status').isin('At anchor','Moored')&(F.col('sog')>3)).alias('n_anchor_moving'),
         n(ok&(F.col('sog')>3)).alias('n_seg_moving'),
         n(ok&(F.col('sog')>3)&(F.col('dist')==0)).alias('n_frozen_moving'),
         n(ok&F.col('dt').between(10,600)&F.col('sog').isNotNull()).alias('n_seg_speedcheck'),
         n(ok&F.col('dt').between(10,600)&(F.abs(implied-F.col('sog'))>5)).alias('n_sog_mismatch')]
    for h in [2,6]:
        for covered in [False,True]:
            ev=F.col('seg')&(F.col('dt')>h*3600)&(F.col('cov') if covered else F.lit(True))
            name=('cov_' if covered else '')+f'gap{h}h'
            agg+=[n(ev).alias('n_'+name),F.sum(F.when(ev,F.col('dt')).otherwise(0)).alias(name+'_s')]
    for gate,axis,value,(lo,hi),out in GATES:
        other='Longitude' if axis=='Latitude' else 'Latitude'
        a0,a1=F.col('p_'+axis),F.col(axis)
        cross=ok&((a0-value)*(a1-value)<0)
        at=F.col('p_'+other)+(F.col(other)-F.col('p_'+other))*(F.lit(value)-a0)/(a1-a0)
        cross=cross&at.between(lo,hi)
        outbound=(a1>a0) if out=='up' else (a1<a0)
        agg+=[n(cross&outbound).alias(f'n_{gate}_out'),n(cross&~outbound).alias(f'n_{gate}_in')]
    vessel=tr.groupBy('MMSI').agg(*agg)

    # --- Ô 5 phút (điểm cuối mỗi ô): lảng vảng + gặp gỡ ---
    bins=(clean.withColumn('bin',F.floor(F.col('t')/300))
      .groupBy('MMSI','bin').agg(F.max(F.struct('t','Latitude','Longitude','sog')).alias('p')).select('MMSI','bin','p.*')
      .withColumn('px',F.floor(F.col('Latitude')/.01)).withColumn('py',F.floor(F.col('Longitude')/.02))
      .join(F.broadcast(ports),['px','py'],'left').withColumn('port',F.coalesce('port',F.lit(False))).cache())
    loiter=bins.groupBy('MMSI').agg(F.count('*').alias('n_bins'),n(~F.col('port')).alias('n_bins_offport'),
      n(~F.col('port')&(F.col('sog')<1)).alias('n_bins_loiter'),n(F.col('port')).alias('n_bins_port'))
    lat=F.radians(F.col('Latitude'));lon=F.radians(F.col('Longitude'));R=3440.065*1852
    slow=(bins.filter((F.col('sog')<3)&~F.col('port'))
      .withColumn('gx',F.floor(R*F.cos(lat)*F.cos(lon)/500)).withColumn('gy',F.floor(R*F.cos(lat)*F.sin(lon)/500))
      .withColumn('gz',F.floor(R*F.sin(lat)/500)).select('MMSI','bin','Latitude','Longitude','gx','gy','gz'))
    left=slow
    for ax in 'xyz':left=left.withColumn('n'+ax,F.explode(F.sequence(F.col('g'+ax)-1,F.col('g'+ax)+1)))
    a,b=left.alias('a'),slow.alias('b')
    pair=(a.join(b,(F.col('a.bin')==F.col('b.bin'))&(F.col('a.nx')==F.col('b.gx'))&(F.col('a.ny')==F.col('b.gy'))
                 &(F.col('a.nz')==F.col('b.gz'))&(F.col('a.MMSI')<F.col('b.MMSI')))
      .filter(distance_nm(F,F.col('a.Latitude'),F.col('a.Longitude'),F.col('b.Latitude'),F.col('b.Longitude'))*1852<500)
      .select(F.col('a.MMSI').alias('m1'),F.col('b.MMSI').alias('m2'),F.col('a.bin').alias('bin'),
              F.col('a.Latitude').alias('lat'),F.col('a.Longitude').alias('lon')).dropDuplicates(['m1','m2','bin']))
    wp=Window.partitionBy('m1','m2').orderBy('bin')
    pair=pair.withColumn('session',F.sum(F.when(F.col('bin')-F.lag('bin').over(wp)==1,0).otherwise(1)).over(wp))
    encounters=(pair.groupBy('m1','m2','session').agg(F.min('bin').alias('first_bin'),F.max('bin').alias('last_bin'),
                  F.count('*').alias('n_bins'),F.avg('lat').alias('lat'),F.avg('lon').alias('lon')).drop('session'))

    # --- Giá trị khai báo (đếm theo MMSI × trường × giá trị) và độ phủ hệ thống theo giờ ---
    dims=F.concat_ws('|',*[F.col(c).cast('int').cast('string') for c in 'ABCD'])
    kv=F.explode(F.array(*[F.struct(F.lit(f).alias('field'),
         (dims if f=='dims' else F.upper(F.trim(F.col(f).cast('string')))).alias('value')) for f in VALUE_FIELDS]))
    values=(clean.select('MMSI',kv.alias('kv')).select('MMSI','kv.*')
      .filter(F.col('value').isNotNull()&(F.col('value')!='')).groupBy('MMSI','field','value').count())
    hourly=clean.groupBy(F.floor(F.col('t')/3600).alias('hour')).agg(F.sum('n_copies').alias('n_msgs'),F.countDistinct('MMSI').alias('n_ships'))

    import pandas as pd
    stage=OUT/('.staging-'+date);target=OUT/('date='+date)
    shutil.rmtree(stage,ignore_errors=True)
    for name,df in [('vessel',vessel.join(loiter,'MMSI','left')),('encounters',encounters),('values',values),('hourly',hourly)]:
        df.coalesce(1).write.option('compression','zstd').parquet(str(stage/name))
    bins.unpersist()
    v,e,val,hr=[pd.read_parquet(stage/x) for x in ['vessel','encounters','values','hourly']]

    # --- Đối chiếu sau khi ghi; hai nhánh độc lập (window theo MMSI, group theo giờ) phải khớp tổng tin ---
    day0=int(pd.Timestamp(date,tz='UTC').timestamp())
    assert v.MMSI.is_unique and v.MMSI.notna().all(),'vessel key'
    assert int(v.n_msgs.sum())==int(hr.n_msgs.sum()),'n_msgs mismatch'
    assert set(val.MMSI)<=set(v.MMSI) and set(e.m1)|set(e.m2)<=set(v.MMSI),'MMSI ngoài bảng vessel'
    assert v.first_t.min()>=day0 and v.last_t.max()<day0+86400,'data outside day'
    assert (v.n_bins>0).all(),'vessel without 5-min bin'
    assert hr.hour.between(day0//3600,day0//3600+23).all()
    meta=dict(date=date,n_vessels=len(v),n_msgs=int(hr.n_msgs.sum()),n_encounters=len(e),n_values=len(val),
              feature_s=round(time.monotonic()-start,3),at=now())
    save_json(stage/'_meta.json',meta)
    shutil.rmtree(target,ignore_errors=True);stage.rename(target)
    return meta


def main():
    import pandas as pd
    p=argparse.ArgumentParser();p.add_argument('--dates',nargs='*');p.add_argument('--cores',type=int,default=4)
    p.add_argument('--driver-memory',default='4g');args=p.parse_args()
    configure_utc()
    import os,sys
    os.environ['PYSPARK_PYTHON']=sys.executable  # worker phải cùng bản Python với driver
    from silver import session
    from pipeline_resources import Watch
    dates=args.dates or sorted(d.name[5:] for d in (ROOT/'data/silver').glob('date=*'))
    spark=session(ROOT,app_name='AIS-features-day',cores=args.cores,driver_memory=args.driver_memory)
    spark.conf.set('spark.sql.shuffle.partitions','64')  # 1 ngày ~11 triệu dòng; 256 là dư
    watch=Watch()
    high,ports,map_info=maps(spark)
    save_json(OUT/'_maps.json',dict(map_info,gates=GATES,at=now()))
    log=ROOT/'results/features_day_stats.csv'
    try:
        for date in dates:
            if STOP.exists():print('stop file -> dừng',flush=True);break
            if (OUT/('date='+date)/'_meta.json').exists() and not args.dates:continue
            watch.reset(date);watch.stage='features_day'
            meta=build(spark,date,high,ports)
            meta.update(peak_gib=round(watch.peak/1024**3,3),anon_peak_gib=round(watch.anon_peak/1024**3,3),
                        host_min_gib=round(watch.host_min/1024**3,3))
            pd.DataFrame([meta]).to_csv(log,mode='a',header=not log.exists(),index=False)
            print(json.dumps(meta),flush=True)
            spark._jvm.java.lang.System.gc()
    finally:
        spark.stop();watch.close()


if __name__=='__main__':
    from common import pipeline_lock
    with pipeline_lock('features'):main()
