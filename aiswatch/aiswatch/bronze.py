"""Bronze bằng Spark: zip của một ngày -> Parquet có kiểu dữ liệu + bảng tóm tắt theo MMSI (data/vessel_day/date=D.parquet).

Spark không đọc được CSV nằm trong zip, nên CSV được giải nén tạm ra data/tmp rồi xoá ngay sau khi ghi xong.
Giữ nguyên ngữ nghĩa của bản lúc xây dựng (đã đối chiếu):
- bronze: 26 cột gốc dạng chuỗi, 13 cột số ép sang double (không ép được -> null), thêm `ts` (UTC);
- vessel_day: số bản tin, số vị trí hợp lệ, khung thời gian, khung toạ độ, và bảng đếm giá trị (JSON) của
  Type of mobile / IMO (chỉ IMO hợp lệ checksum) / Name / Callsign / Ship type, cùng giá trị chiếm đa số.
Đối chiếu: số dòng, số MMSI phân biệt, tổng MMSI (checksum) của bronze phải khớp bảng tóm tắt.
"""
import shutil, time, zipfile
import pandas as pd
from .config import DATA as ROOT, save_json, now
from .ais_schema import COLUMNS, NUMERIC

COLS = ['# Timestamp', 'Type of mobile', 'MMSI', 'Latitude', 'Longitude', 'IMO', 'Name', 'Callsign', 'Ship type']
HIST = {'Type of mobile': ('mobile_counts', 'type_of_mobile'), 'IMO': ('imo_counts', None), 'Name': ('name_counts', 'name'),
        'Callsign': ('callsign_counts', 'callsign'), 'Ship type': ('ship_type_counts', 'ship_type')}
# Chuỗi được coi là rỗng trong bảng tóm tắt (mặc định của pyarrow lúc xây dựng).
NULL_TOKENS = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN',
               'n/a', 'nan', 'null']


def extract(zip_path, date):
    with zipfile.ZipFile(zip_path) as z:
        entries = [e for e in z.infolist() if e.filename.lower().endswith('.csv') and not e.is_dir()]
        assert len(entries) == 1, 'zip phải chứa đúng 1 CSV'
        csv = ROOT/'data/tmp'/f'{date}.csv'
        with z.open(entries[0]) as src, csv.open('wb') as dst:
            shutil.copyfileobj(src, dst, 16 << 20)
        with csv.open() as f:
            header = next(iter(f)).rstrip('\r\n').split(',')
        assert header == COLUMNS, 'sai header hoặc thứ tự cột'
        return csv, entries[0]


def build(spark, zip_path, date):
    from pyspark.sql import functions as F, Window
    from pyspark.sql.types import StructType, StructField, StringType
    start = time.monotonic()
    target = ROOT/'data/bronze'/f'date={date}'; stage = ROOT/'data/bronze'/f'.staging-{date}'
    shutil.rmtree(stage, ignore_errors=True)
    csv, entry = extract(zip_path, date)
    try:
        schema = StructType([StructField(c, StringType()) for c in COLUMNS])
        raw = spark.read.option('escape', '"').csv(str(csv), header=True, schema=schema, mode='FAILFAST')   # "" trong trường có ngoặc kép = 1 dấu "
        typed = raw.select(*[F.col(f'`{c}`').cast('double').alias(c) if c in NUMERIC else F.col(f'`{c}`') for c in COLUMNS],
                           F.to_timestamp(F.col('`# Timestamp`'), 'dd/MM/yyyy HH:mm:ss').alias('ts'))
        typed.write.option('compression', 'zstd').parquet(str(stage))
        b = spark.read.parquet(str(stage))
        chk = b.agg(F.count('*').alias('rows'), F.sum(F.col('ts').isNull().cast('long')).alias('null_ts'),
                    F.min(F.to_date('ts')).alias('d0'), F.max(F.to_date('ts')).alias('d1'),
                    F.sum(F.col('MMSI').isNull().cast('long')).alias('missing'),
                    (F.countDistinct('MMSI') + F.max(F.col('MMSI').isNull().cast('int'))).alias('n_mmsi'),
                    F.sum(F.col('MMSI').cast('long')).alias('mmsi_sum')).first().asDict()
        assert chk['null_ts'] == 0, 'có thời điểm không đọc được'
        assert str(chk['d0']) == date and str(chk['d1']) == date, 'dữ liệu nằm ngoài ngày'
        fails = raw.agg(*[F.sum((F.col(f'`{c}`').isNotNull() & F.col(f'`{c}`').cast('double').isNull()).cast('long')).alias(c) for c in NUMERIC]).first().asDict()

        # ---- bảng tóm tắt theo MMSI (ngữ nghĩa giống bản lúc xây dựng) ----
        fr = raw.select(*[F.when(F.col(f'`{c}`').isin(NULL_TOKENS), None).otherwise(F.col(f'`{c}`')).alias(c) for c in COLS])
        imo = F.trim(F.regexp_replace('IMO', r'^IMO\s*', ''))
        dig = [F.substring(imo, i+1, 1).cast('int') for i in range(7)]
        ok_imo = imo.rlike(r'^[1-9][0-9]{6}$') & ((sum(dig[i]*(7-i) for i in range(6)) % 10) == dig[6])
        fr = (fr.withColumn('IMO', F.when(ok_imo, imo)).withColumn('lat', F.col('Latitude').cast('double')).withColumn('lon', F.col('Longitude').cast('double'))
                .withColumn('time', F.to_timestamp('`# Timestamp`', 'dd/MM/yyyy HH:mm:ss'))
                .withColumn('valid', F.coalesce(F.col('lat').between(-90, 90) & F.col('lon').between(-180, 180) & (F.col('lon') != 0), F.lit(False))))
        fr = fr.cache()
        v = F.col('valid')
        base = fr.groupBy('MMSI').agg(F.count('*').alias('n_msgs'), F.sum(v.cast('long')).alias('n_valid_pos'),
                                      F.min('time').alias('first_ts'), F.max('time').alias('last_ts'),
                                      F.min(F.when(v, F.col('lat'))).alias('min_latitude'), F.min(F.when(v, F.col('lon'))).alias('min_longitude'),
                                      F.max(F.when(v, F.col('lat'))).alias('max_latitude'), F.max(F.when(v, F.col('lon'))).alias('max_longitude'))
        out = base
        for c, (cnt_name, mode_name) in HIST.items():
            h = fr.filter(F.col(f'`{c}`').isNotNull() & (F.col(f'`{c}`') != '')).groupBy('MMSI', F.col(f'`{c}`').alias('val')).count()
            j = h.groupBy('MMSI').agg(F.to_json(F.map_from_entries(F.array_sort(F.collect_list(F.struct('val', 'count'))))).alias(cnt_name))
            if mode_name:   # giá trị chiếm đa số; hoà thì lấy giá trị nhỏ nhất
                top = h.withColumn('r', F.row_number().over(Window.partitionBy('MMSI').orderBy(F.desc('count'), 'val'))) \
                       .filter('r = 1').select('MMSI', F.col('val').alias(mode_name))
                j = j.join(top, 'MMSI', 'left')
            out = out.join(j, out.MMSI.eqNullSafe(j.MMSI), 'left').drop(j.MMSI)
        tmp = ROOT/'data/vessel_day'/f'.stage-{date}'
        shutil.rmtree(tmp, ignore_errors=True)
        out.coalesce(1).write.parquet(str(tmp))
        s = pd.read_parquet(tmp)
        for cnt_name, _ in HIST.values(): s[cnt_name] = s[cnt_name].fillna('{}')
        for c in ('first_ts', 'last_ts'): s[c] = s[c].dt.tz_localize(None) if s[c].dt.tz is not None else s[c]
        s = s.sort_values('MMSI', na_position='last').reset_index(drop=True)
        fr.unpersist()
        # ---- đối chiếu bronze với bảng tóm tắt ----
        assert int(s.n_msgs.sum()) == chk['rows'], 'tổng bản tin lệch'
        assert len(s) == chk['n_mmsi'], 'số MMSI lệch'
        ms = pd.to_numeric(s.MMSI, errors='coerce')
        assert int((ms*s.n_msgs).sum()) == int(chk['mmsi_sum'] or 0), 'checksum MMSI lệch'
        summary = ROOT/'data/vessel_day'/f'date={date}.parquet'
        s.to_parquet(summary.with_suffix('.tmp.parquet'), index=False); summary.with_suffix('.tmp.parquet').replace(summary)
        shutil.rmtree(tmp, ignore_errors=True)
        info = dict(date=date, rows=chk['rows'], rows_bronze=chk['rows'], n_mmsi=chk['n_mmsi'], mmsi_sum=int(chk['mmsi_sum'] or 0),
                    rows_missing_mmsi=chk['missing'], rows_valid_pos=int(s.n_valid_pos.sum()), csv_bytes=entry.file_size,
                    numeric_cast_failures=fails, bronze_s=round(time.monotonic()-start, 1), engine='spark', verified_at=now())
        save_json(stage/'_verification.json', info)
        shutil.rmtree(target, ignore_errors=True); stage.rename(target)
        return info
    finally:
        csv.unlink(missing_ok=True)
