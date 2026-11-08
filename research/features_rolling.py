"""Feature cửa sổ trượt 30 ngày cho từng ngày chấm D (dùng lại đúng hàm window() của lúc train).

Danh sách tàu được chấm tại D:
- là tàu dầu có mặt trong 30 ngày (D-29..D), có ít nhất 10 bản tin;
- danh tính tàu (MMSI -> vessel_id) lấy từ cohort các snapshot tháng;
- chưa bị EU/UK cấm tại D.
Nhãn backtest: y = bị cấm lần đầu trong (D, D + 6 tháng]. Chỉ dùng để đánh giá, không đưa vào feature.
Kèm feature hành trình qua gate (voyage_features.compute, nhìn lại 180 ngày).
Đầu ra: data/features_rolling/date=D/features.parquet. Ngày nào đã có file thì bỏ qua.
"""
import argparse, json
import numpy as np, pandas as pd
from features_window import window, DAY
import voyage_features as vy
from train_model import ROOT, load

OUT = ROOT/'data/features_rolling'


def cohort_mapping():
    d = load()
    d = d[d.is_tanker].sort_values('snapshot')
    m = d.assign(MMSI=d.mmsis.map(json.loads)).explode('MMSI')
    m = m.drop_duplicates('MMSI', keep='last')                      # danh tính mới nhất của MMSI
    info = d.drop_duplicates('vessel_id', keep='last').set_index('vessel_id')[['label_id', 'first_listed']]
    return m[['MMSI', 'vessel_id']], info


def build(day, cmap, info, labels_by_id, V, N):
    D = pd.Timestamp(day)
    dates = [str(x.date()) for x in pd.date_range(D - pd.Timedelta(days=29), D)]
    r, _ = window(None, 1, cmap, labels_by_id, dates=dates, cutoff=D, partners=False)
    assert r is not None, f'{day}: thiếu ngày trong tầng ngày'
    out, _ = r
    v = pd.concat([pd.read_parquet(DAY/f'date={x}'/'vessel', columns=['MMSI', 'n_msgs', 'last_t', 'last_lat', 'last_lon']).assign(date=x)
                   for x in dates]).merge(cmap, on='MMSI')
    g = v.sort_values('last_t').groupby('vessel_id').agg(n_msgs=('n_msgs', 'sum'), last_seen=('date', 'max'), last_t=('last_t', 'last'),
                                                         last_lat=('last_lat', 'last'), last_lon=('last_lon', 'last'))
    out = out.add_suffix('_l1m').join(g, how='inner').join(info, how='left')
    fl = pd.to_datetime(out.first_listed.replace('', None).str[:10], errors='coerce')
    out = out[(out.n_msgs >= 10) & ~(fl <= D)].copy()
    fl = fl.loc[out.index]
    out['y'] = ((fl > D) & (fl <= D + pd.DateOffset(months=6))).astype(int)
    out['days_since_seen'] = (D - pd.to_datetime(out.last_seen)).dt.days
    out['date'] = day
    out = out.reset_index().rename(columns={'index': 'vessel_id'})
    vf, _ = vy.compute(D, out.vessel_id, V, N)                     # feature hành trình, nhìn lại 180 ngày tới D
    out = out.merge(vf, on='vessel_id', how='left', validate='1:1')
    assert out.vessel_id.is_unique
    p = OUT/f'date={day}'; p.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p/'.features.parquet', index=False); (p/'.features.parquet').rename(p/'features.parquet')
    return len(out), int(out.y.sum())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--start', default='2025-10-01'); ap.add_argument('--end', default='2026-03-31')
    a = ap.parse_args()
    cmap, info = cohort_mapping()
    labels = pd.read_parquet(ROOT/'data/labels/vessels_first_listed.parquet')
    labels_by_id = dict(zip(labels.vessel_id, labels.first_listed))
    V, N = vy.load_daily(end=a.end)
    for D in pd.date_range(a.start, a.end):
        day = str(D.date())
        if (OUT/f'date={day}'/'features.parquet').exists():
            continue
        n, pos = build(day, cmap, info, labels_by_id, V, N)
        print(json.dumps(dict(date=day, n=n, pos=pos)), flush=True)
