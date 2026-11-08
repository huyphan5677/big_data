"""Stream one zip into ordered typed Bronze and a reconciled vessel summary."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import csv
import gc
import json
import shutil
import resource
import time
import zipfile
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq
import polars as pl
from common import ROOT,save_json,now
from ais_schema import COLUMNS,NUMERIC
COLS=['# Timestamp','Type of mobile','MMSI','Latitude','Longitude','IMO','Name','Callsign','Ship type']
HIST={'Type of mobile':'mobile_counts','IMO':'imo_counts','Name':'name_counts',
      'Callsign':'callsign_counts','Ship type':'ship_type_counts'}


def imo_expr():
    value=pl.col('IMO').str.replace(r'^IMO\s*','').str.strip_chars()
    digits=[value.str.slice(i,1).cast(pl.Int32,strict=False) for i in range(7)]
    valid=value.str.contains(r'^[1-9][0-9]{6}$')
    checksum=sum(digits[i]*(7-i) for i in range(6))%10==digits[6]
    return pl.when(valid&checksum).then(value).otherwise(None).alias('IMO')


def discard_cache(path):
    if isinstance(path,(str,os.PathLike)) and hasattr(os,'posix_fadvise'):
        with Path(path).open('rb') as stream:
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)


def verify_bronze(directory,expected_rows,expected_mmsi,expected_sum):
    total=checksum=0; identifiers=set()
    for path in sorted(Path(directory).glob('part-*.parquet')):
        file=pq.ParquetFile(path)
        assert file.schema_arrow.names==COLUMNS+['ts']
        assert file.schema_arrow.field('ts').type.tz=='UTC'
        assert all(file.schema_arrow.field(c).type==pa.float64() for c in NUMERIC), 'Bronze numeric schema'
        assert file.metadata.row_group(0).column(0).compression=='ZSTD', 'Bronze compression'
        for batch in file.iter_batches(batch_size=65536,columns=['MMSI']):
            s=pl.from_arrow(batch)['MMSI']
            total+=len(s);identifiers.update(s.unique().to_list())
            checksum+=int(s.cast(pl.Int64,strict=False).sum() or 0)
        discard_cache(path)
    assert total==expected_rows, 'Bronze row count'
    assert len(identifiers)==expected_mmsi, 'Bronze distinct MMSI'
    assert checksum==expected_sum, 'Bronze MMSI checksum'
    return dict(rows=total,n_mmsi=len(identifiers),mmsi_sum=checksum)


def build(path,date,root=ROOT):
    root=Path(root);start=time.monotonic()
    target=root/'data/bronze'/('date='+date)
    stage=root/'data/bronze'/('.staging-'+date)
    shutil.rmtree(stage,ignore_errors=True);stage.mkdir(parents=True)
    parts=[];hist={c:[] for c in HIST}
    rows=missing=valid_rows=mmsi_sum=0
    failures={c:0 for c in NUMERIC}
    with zipfile.ZipFile(path) as z:
        entries=[e for e in z.infolist() if e.filename.lower().endswith('.csv') and not e.is_dir()]
        assert len(entries)==1,'Expected one CSV'
        entry=entries[0]
        with z.open(entry) as stream:
            reader=pc.open_csv(stream,read_options=pc.ReadOptions(block_size=8*1024**2,use_threads=False),
              convert_options=pc.ConvertOptions(column_types={c:pa.string() for c in COLUMNS},
                  strings_can_be_null=True,null_values=['']))
            assert reader.schema.names==COLUMNS,'Wrong header/order'
            writer=None
            try:
                for batch in reader:
                    raw=pl.from_arrow(batch)
                    typed=raw.with_columns(*[pl.col(c).cast(pl.Float64,strict=False) for c in NUMERIC],
                        pl.col('# Timestamp').str.strptime(pl.Datetime,'%d/%m/%Y %H:%M:%S',strict=True)
                          .dt.replace_time_zone('UTC').alias('ts'))
                    assert typed['ts'].null_count()==0
                    assert str(typed['ts'].min().date())==date and str(typed['ts'].max().date())==date
                    for c in NUMERIC:
                        failures[c]+=int((raw[c].is_not_null() & typed[c].is_null()).sum())
                    table=typed.to_arrow()
                    if writer is None:
                        writer=pq.ParquetWriter(stage/'part-000.parquet',table.schema,compression='zstd')
                    writer.write_table(table)
                    rows+=raw.height;missing+=raw['MMSI'].null_count()
                    mmsi_sum+=int(raw['MMSI'].cast(pl.Int64,strict=False).sum() or 0)
                    # Preserve the frozen legacy summary semantics, separately from raw Bronze.
                    frame=raw.select(COLS).with_columns(*[pl.col(c).replace(pc.ConvertOptions().null_values,None) for c in COLS])
                    frame=frame.with_columns(pl.col('Latitude').cast(pl.Float64,strict=False),
                        pl.col('Longitude').cast(pl.Float64,strict=False),imo_expr(),
                        typed['ts'].dt.replace_time_zone(None).alias('time'))
                    frame=frame.with_columns((pl.col('Latitude').is_between(-90,90)&
                        pl.col('Longitude').is_between(-180,180)&(pl.col('Longitude')!=0)).fill_null(False).alias('valid'))
                    valid_rows+=int(frame['valid'].sum())
                    parts.append(frame.group_by('MMSI').agg(pl.len().alias('n_msgs'),
                      pl.col('valid').sum().cast(pl.UInt64).alias('n_valid_pos'),
                      pl.col('time').min().alias('first_ts'),pl.col('time').max().alias('last_ts'),
                      *[pl.col(c).filter(pl.col('valid')).min().alias('min_'+c.lower()) for c in ['Latitude','Longitude']],
                      *[pl.col(c).filter(pl.col('valid')).max().alias('max_'+c.lower()) for c in ['Latitude','Longitude']]))
                    for c in HIST:
                        hist[c].append(frame.filter(pl.col(c).is_not_null()&(pl.col(c)!=''))
                                         .group_by('MMSI',c).len(name='count'))
            finally:
                if writer:writer.close()
    assert rows>0
    daily=pl.concat(parts).group_by('MMSI').agg(pl.col('n_msgs').sum().cast(pl.UInt64),
      pl.col('n_valid_pos').sum(),pl.col('first_ts').min(),pl.col('last_ts').max(),
      *[pl.col(c).min() for c in ['min_latitude','min_longitude']],
      *[pl.col(c).max() for c in ['max_latitude','max_longitude']])
    assert daily['n_msgs'].sum()==rows and daily['n_valid_pos'].sum()==valid_rows
    records={r['MMSI']:r for r in daily.iter_rows(named=True)}
    for c,name in HIST.items():
        counts=pl.concat(hist[c]).group_by('MMSI',c).agg(pl.col('count').sum())
        values={}
        for m,v,n in counts.iter_rows():values.setdefault(m,{})[v]=n
        for m,r in records.items():
            h=values.get(m,{})
            r[name]=json.dumps(h,sort_keys=True,separators=(',',':'))
            if c!='IMO':
                r[{'Type of mobile':'type_of_mobile','Name':'name','Callsign':'callsign','Ship type':'ship_type'}[c]]=min(h,key=lambda x:(-h[x],x)) if h else None
    out=pl.DataFrame(list(records.values()),infer_schema_length=None).sort('MMSI')
    # Numeric checksum is independently reconciled against the MMSI-level summary.
    summary_sum=sum(int(r['MMSI'])*int(r['n_msgs']) for r in records.values()
                    if r['MMSI'] is not None and str(r['MMSI']).lstrip('-').isdigit())
    assert summary_sum==mmsi_sum
    verify_bronze(stage,rows,len(records),mmsi_sum)
    summary=root/'data/vessel_day'/f'date={date}.parquet';summary.parent.mkdir(parents=True,exist_ok=True)
    if summary.exists():
        old=pl.read_parquet(summary).sort('MMSI')
        assert old.equals(out),'Legacy summary mismatch'
    else:
        temp=summary.with_suffix('.tmp.parquet');out.write_parquet(temp,compression='zstd');temp.replace(summary)
    info=dict(date=date,format_version=3,rows=rows,rows_bronze=rows,n_mmsi=out.height,mmsi_sum=mmsi_sum,
      rows_missing_mmsi=missing,rows_valid_pos=valid_rows,csv_bytes=entry.file_size,crc32=f'{entry.CRC:08x}',
      summary_bytes=summary.stat().st_size,bronze_bytes=sum(p.stat().st_size for p in stage.glob('part-*')),
      bronze_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
      numeric_cast_failures=json.dumps(failures,sort_keys=True),bronze_s=round(time.monotonic()-start,3),
      timezone='UTC assumed; source timestamps contain no offset',timestamp_format='%d/%m/%Y %H:%M:%S',
      source_row_order_preserved=True,nonempty_string_null_tokens_preserved=True,verified_at=now())
    save_json(stage/'_verification.json',info)
    assert not target.exists(),'Verify existing Bronze before replacing'
    stage.rename(target);discard_cache(path)
    return info


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--date',required=True);p.add_argument('--zip',required=True);p.add_argument('--result',required=True)
    args=p.parse_args()
    try:save_json(args.result,build(args.zip,args.date))
    except Exception as exc:
        save_json(args.result,dict(status='error',error_type=type(exc).__name__,error=str(exc)[:240]))
        raise
