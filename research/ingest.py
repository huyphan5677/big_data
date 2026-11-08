"""One day at a time: download -> Bronze -> Spark Silver -> reconcile -> delete zip."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('ARROW_NUM_THREADS','1')
import argparse
import csv
import fcntl
import json
import shutil
import time
import requests
from common import ROOT,now
from bronze import verify_bronze,discard_cache

GIB=1024**3
STATS=ROOT/'results/day_stats.csv'
FIELDS=['date','key','bytes_expected','bytes_got','rows','n_mmsi','download_s','parse_s','status','attempt',
 'csv_bytes','crc32','rows_missing_mmsi','rows_valid_pos','summary_bytes','free_disk_gib','finished_at','error',
 'download_source','rows_dedup','n_flag_bad_pos','n_flag_not_class_a','n_flag_bad_mmsi','ais_bytes','ais_version',
 'format_version','bronze_s','silver_s','bronze_bytes','silver_bytes','rows_bronze','peak_memory_gib','peak_anon_gib',
 'mmsi_sum','numeric_cast_failures','wall_s','spark_app_id','spark_master','host_min_available_gib','bronze_workers','prefetch_days','bronze_peak_rss_gib']


def planned_dates():
    import pandas as pd
    return [str(x.date()) for start,end in [('2025-04-01','2025-09-30'),('2026-01-01','2026-03-31'),
          ('2025-10-01','2025-12-31')] for x in pd.date_range(start,end)]


def append_stats(row):
    fresh=not STATS.exists()
    with STATS.open('a') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        if fresh:writer.writeheader()
        writer.writerow({k:v for k,v in row.items() if k in FIELDS})
        stream.flush();os.fsync(stream.fileno())
    if row.get('status')=='ok' and row.get('format_version')==3:
        silver_stats=ROOT/'results/silver_stats.csv'
        cols=['date','rows_bronze','rows_dedup','n_flag_bad_pos','n_flag_not_class_a','n_flag_bad_mmsi','silver_bytes','silver_s','mmsi_sum']
        current={}
        for record in csv.DictReader(STATS.open()):
            if record.get('status')=='ok' and record.get('format_version')=='3':
                current[record['date']]=record
        temporary=silver_stats.with_suffix('.tmp')
        with temporary.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=cols);writer.writeheader()
            writer.writerows({c:r[c] for c in cols} for _,r in sorted(current.items()))
        temporary.replace(silver_stats)


def migrate_stats():
    if not STATS.exists():return []
    rows=list(csv.DictReader(STATS.open()))
    if rows and list(rows[0])!=FIELDS:
        temporary=STATS.with_suffix('.tmp')
        with temporary.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
        temporary.replace(STATS)
    return rows


def verify_receipts(date,row):
    """Resume verified days without rescanning all row-level columns.

    The original day completed full independent reconciliation before status=ok.
    Recheck receipts, Parquet metadata and file sizes now; final audit rescans rows.
    """
    import pyarrow.parquet as pq
    b=ROOT/'data/bronze'/('date='+date);s=ROOT/'data/silver'/('date='+date)
    bi=json.loads((b/'_verification.json').read_text());si=json.loads((s/'_verification.json').read_text())
    assert bi['format_version']==si['format_version']==3
    assert bi['rows']==si['rows_bronze']==int(row['rows'])
    assert si['rows_dedup']==int(row['rows_dedup'])
    assert bi['n_mmsi']==si['n_mmsi']==int(row['n_mmsi'])
    assert bi['mmsi_sum']==si['mmsi_sum']==int(row['mmsi_sum'])
    for folder,rows,size in [(b,bi['rows'],bi['bronze_bytes']),(s,si['rows_dedup'],si['bytes'])]:
        files=list(folder.glob('part-*.parquet'))
        assert files and sum(pq.ParquetFile(p).metadata.num_rows for p in files)==rows
        assert sum(p.stat().st_size for p in files)==size
    summary=ROOT/'data/vessel_day'/f'date={date}.parquet'
    assert pq.ParquetFile(summary).metadata.num_rows==bi['n_mmsi']
    return bi,si


def verify_outputs(date):
    import pyarrow.parquet as pq
    import polars as pl
    b=ROOT/'data/bronze'/('date='+date);s=ROOT/'data/silver'/('date='+date)
    bi=json.loads((b/'_verification.json').read_text());si=json.loads((s/'_verification.json').read_text())
    assert bi['format_version']==si['format_version']==3
    verify_bronze(b,bi['rows'],bi['n_mmsi'],bi['mmsi_sum'])
    total=unique=checksum=0;identifiers=set()
    flags={f:0 for f in ['flag_bad_pos','flag_not_class_a','flag_bad_mmsi']}
    for path in sorted(s.glob('part-*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['MMSI','ts','n_copies']+list(flags)):
            frame=pl.from_arrow(batch);copies=frame['n_copies']
            assert copies.min()>=1
            total+=int(copies.sum());unique+=frame.height;identifiers.update(frame['MMSI'].unique().to_list())
            checksum+=int((frame['MMSI'].cast(pl.Int64,strict=False)*copies).sum() or 0)
            assert frame['ts'].null_count()==0
            assert str(frame['ts'].min().date())==date and str(frame['ts'].max().date())==date
            for f in flags:flags[f]+=int(frame.filter(pl.col(f))['n_copies'].sum())
        discard_cache(path)
    assert total==bi['rows']==si['rows_bronze'] and unique==si['rows_dedup']
    assert len(identifiers)==bi['n_mmsi']==si['n_mmsi'] and checksum==bi['mmsi_sum']==si['mmsi_sum']
    assert all(flags[f]==si['n_'+f] for f in flags)
    summary=pl.read_parquet(ROOT/'data/vessel_day'/f'date={date}.parquet',columns=['MMSI','n_msgs'])
    assert summary.height==bi['n_mmsi'] and summary['n_msgs'].sum()==total
    return bi,si


def download(item):
    date,key,expected=item['date'],item['key'],int(item['size'])
    path=ROOT/'data/tmp'/(date+'.zip');path.parent.mkdir(parents=True,exist_ok=True)
    ledger=ROOT/'logs'/f'{date}.downloads.jsonl'
    history=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    marker=ROOT/'logs'/f'{date}.range_incomplete.json'
    if path.exists() and path.stat().st_size==expected and not marker.exists():
        return path,dict(bytes_got=expected,download_s=0,download_source='recovered_size_verified',attempt=max(1,len(history)))
    if path.exists():
        with (ROOT/'logs'/f'{date}.jsonl').open('a') as stream:
            known=json.loads(marker.read_text()).get('bytes_verified',0) if marker.exists() else path.stat().st_size
            stream.write(json.dumps(dict(at=now(),attempt=len(history),error='recovered_interrupted_download',bytes_got=known,bytes_are_lower_bound=marker.exists()))+'\n')
    path.unlink(missing_ok=True)
    for attempt in range(len(history)+1,4):
        free=shutil.disk_usage(ROOT).free
        if free<60*GIB+expected+8*GIB:raise RuntimeError('disk_stopped: giữ 60 GiB và dự phòng xử lý')
        zips=set((ROOT/'data/tmp').glob('*.zip'))
        if zips-{path}:raise RuntimeError('zip_slots_full: xử lý zip tồn trước khi tải ngày khác')
        path.touch(exist_ok=True)
        start=time.monotonic();got=0
        with ledger.open('a') as stream:
            stream.write(json.dumps(dict(at=now(),attempt=attempt,bytes_expected=expected,format_version=3))+'\n')
            stream.flush();os.fsync(stream.fileno())
        try:
            with requests.get('http://aisdata.ais.dk/'+key,stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                if 'Content-Length' in response.headers:assert int(response.headers['Content-Length'])==expected
                with path.open('wb') as stream:
                    for chunk in response.iter_content(1024**2):got+=len(chunk);stream.write(chunk)
            assert got==path.stat().st_size==expected,'Archive byte count mismatch'
            marker.unlink(missing_ok=True)
            return path,dict(bytes_got=got,download_s=round(time.monotonic()-start,3),download_source='network',attempt=attempt)
        except Exception as exc:
            got=max(got,getattr(exc,'bytes_got',0))
            with (ROOT/'logs'/f'{date}.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(at=now(),attempt=attempt,error=type(exc).__name__,bytes_got=got))+'\n')
            path.unlink(missing_ok=True)
    raise RuntimeError('failed: đã hết 2 lượt tải lại')


def run(limit=None,until=None,redo_silver=None,workers=1,prefetch=1,spark_cores=4,driver_memory='4g'):
    from ingest_pipeline import run as pipeline_run
    return pipeline_run(limit,until,redo_silver,workers,prefetch,spark_cores,driver_memory)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--until')
    parser.add_argument('--redo-silver');parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--prefetch',type=int,default=1);parser.add_argument('--spark-cores',type=int,default=4)
    parser.add_argument('--driver-memory',default='4g');args=parser.parse_args()
    lock=(ROOT/'logs/ingest.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    run(args.limit,args.until,args.redo_silver,args.workers,args.prefetch,args.spark_cores,args.driver_memory)
