"""Chuẩn bị dữ liệu nhỏ cho app Streamlit (tính trước để app chỉ đọc nhanh).

Đầu ra trong data/app/:
- vessels.parquet: vessel_id, tên, MMSI, trạng thái cấm (first_listed, label_id), tàu dầu hay không.
- daily_pos.parquet: vị trí đầu/cuối mỗi ngày, quãng đường, tỉ lệ ở cảng theo ngày cho mọi tàu đã được chấm.
- crossings.parquet: ngày có lượt qua gate (ra/vào Baltic) + mớn nước, cho các tàu đã được chấm.
- encounters.parquet: phiên gặp >= 30 phút (10/2025-03/2026), kèm vessel_id và trạng thái cấm của hai bên tại ngày gặp.
"""
import json
from pathlib import Path
import pandas as pd
from count_presence import month_vessels
from train_model import ROOT

DAY = ROOT/'data/features_day'; OUT = ROOT/'data/app'; OUT.mkdir(parents=True, exist_ok=True)
DAYS = [str(d.date()) for d in pd.date_range('2025-10-01', '2026-03-31')]
labels = pd.read_parquet(ROOT/'data/labels/vessels_first_listed.parquet')
listed = dict(zip(labels.vessel_id, labels.first_listed.str[:10]))

# MMSI -> danh tính tàu theo từng tháng (giống cách ghép lúc train)
rows = []
for m in pd.period_range('2025-10', '2026-03', freq='M'):
    ds = [str(d.date()) for d in pd.date_range(m.start_time, m.end_time)]
    vs, _ = month_vessels([ROOT/'data/vessel_day'/f'date={d}.parquet' for d in ds], labels, ais_identity_only=True)
    for v in vs:
        for x in v['mmsis']:
            rows.append(dict(MMSI=x, vessel_id=v['vessel_id'], label_id=v['label_id'] or '', is_tanker=bool(v['is_tanker']),
                             name=v['name'], month=str(m)))
M = pd.DataFrame(rows).sort_values('month').drop_duplicates('MMSI', keep='last')
M['first_listed'] = M.label_id.map(listed).fillna('')
V = (M.groupby('vessel_id').agg(name=('name', 'last'), mmsis=('MMSI', lambda s: ','.join(sorted(s))), label_id=('label_id', 'max'),
                               first_listed=('first_listed', 'max'), is_tanker=('is_tanker', 'max')).reset_index())
V.to_parquet(OUT/'vessels.parquet', index=False)
print('vessels', len(V), 'tankers', int(V.is_tanker.sum()), 'ever listed', int((V.first_listed != '').sum()), flush=True)

scored = set(pd.read_parquet(ROOT/'data/predict_all.parquet', columns=['vessel_id']).vessel_id)
mm = M[M.vessel_id.isin(scored)][['MMSI', 'vessel_id']]
P = []
for d in DAYS:
    x = pd.read_parquet(DAY/f'date={d}'/'vessel', columns=['MMSI', 'first_t', 'last_t', 'first_lat', 'first_lon', 'last_lat', 'last_lon',
                                                           'distance_nm', 'n_bins', 'n_bins_port', 'n_msgs'])
    P.append(x.merge(mm, on='MMSI').assign(date=d))
P = pd.concat(P, ignore_index=True)
P['port_frac'] = P.n_bins_port/P.n_bins.where(P.n_bins > 0)
P.to_parquet(OUT/'daily_pos.parquet', index=False, compression='zstd')
print('daily_pos', len(P), flush=True)

# lượt qua gate (ra/vào Baltic) kèm mớn nước khai báo trong ngày — cho phần kể chuyện "con thoi"
G = []
gc = [f'n_{g}_{x}' for g in ('greatbelt', 'oresund', 'skagen') for x in ('out', 'in')]
for d in DAYS:
    x = pd.read_parquet(DAY/f'date={d}'/'vessel', columns=['MMSI', 'draught_max'] + gc)
    x['out'] = x[[c for c in gc if c.endswith('out')]].sum(axis=1); x['in'] = x[[c for c in gc if c.endswith('in')]].sum(axis=1)
    x = x[(x.out > 0) | (x['in'] > 0)]
    G.append(x[['MMSI', 'draught_max', 'out', 'in']].merge(mm, on='MMSI').assign(date=d))
G = pd.concat(G, ignore_index=True)
G.to_parquet(OUT/'crossings.parquet', index=False)
print('crossings', len(G), flush=True)

E = []
for d in DAYS:
    e = pd.read_parquet(DAY/f'date={d}'/'encounters', columns=['m1', 'm2', 'first_bin', 'last_bin', 'lat', 'lon'])
    e = e[(e.last_bin-e.first_bin)*5 >= 30]
    E.append(e.assign(date=d, hours=(e.last_bin-e.first_bin)/12).drop(columns=['first_bin', 'last_bin']))
E = pd.concat(E, ignore_index=True)
info = M.set_index('MMSI')
for k in ('1', '2'):
    m = E[f'm{k}']
    E[f'v{k}'] = m.map(info.vessel_id).fillna('MMSI'+m)
    E[f'tanker{k}'] = m.map(info.is_tanker).fillna(False).astype(bool)
    fl = m.map(info.first_listed).fillna('')
    E[f'status{k}'] = [('already' if f <= d else 'future') if f else 'none' for f, d in zip(fl, E.date)]
E.to_parquet(OUT/'encounters.parquet', index=False, compression='zstd')
print('encounters', len(E), 'with already-sanctioned', int(((E.status1 == 'already') | (E.status2 == 'already')).sum()),
      'with future-sanctioned', int(((E.status1 == 'future') | (E.status2 == 'future')).sum()), flush=True)
