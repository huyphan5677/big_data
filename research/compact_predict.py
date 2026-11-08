"""Gộp data/predict/date=*/ (182 thư mục nhỏ do Spark ghi) thành 1 file cho phân tích/viz đọc nhanh.
Chạy lại sau mỗi lần predict_daily thêm ngày. Đầu ra: data/predict_all.parquet (sắp theo date, rank_all)."""
import glob, time
import pandas as pd
from train_model import ROOT
t = time.time()
parts = [pd.read_parquet(p).assign(date=p.split('date=')[1]) for p in sorted(glob.glob(str(ROOT/'data/predict/date=*')))]
P = pd.concat(parts, ignore_index=True).sort_values(['date', 'rank_all'])
P['date'] = P.date.astype('category')
assert not P.duplicated(['date', 'vessel_id']).any()
tmp = ROOT/'data/.predict_all.parquet'
P.to_parquet(tmp, index=False, compression='zstd', row_group_size=50_000); tmp.rename(ROOT/'data/predict_all.parquet')
print(len(P), P.date.nunique(), f'{time.time()-t:.1f}s')
