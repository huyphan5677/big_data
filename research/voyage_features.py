"""Feature theo hành trình qua gate (nhìn lại tối đa 180 ngày, chỉ dữ liệu <= cutoff), cho mỗi snapshot tháng.

Gate: Great Belt, Øresund, Skagen. "out" = hướng ra Biển Bắc, "in" = hướng vào Baltic (định nghĩa trong features_day.py).
Ý tưởng: tàu xuất dầu Baltic đi ra có hàng (mớn sâu), đi vào không hàng, quay vòng đều đặn; hay đổi cờ/tên.
Danh tính tàu theo từng tháng lấy từ cohort snapshot của tháng đó (MMSI -> vessel_id, khoá theo IMO).
Đầu ra: data/features_voyage/snapshot=S.parquet (vessel_id + v_*).
"""
import json
import numpy as np, pandas as pd
from train_model import ROOT
from features_window import PLACEHOLDERS

DAY = ROOT/'data/features_day'; OUT = ROOT/'data/features_voyage'; OUT.mkdir(parents=True, exist_ok=True)
SNAPS = list(pd.period_range('2025-04', '2026-03', freq='M').strftime('%Y%m'))
LOOK = 180
COLS = ['MMSI', 'n_rows', 'n_dest_ru', 'draught_max'] + [f'n_{g}_{d}' for g in ('greatbelt', 'oresund', 'skagen') for d in ('out', 'in')]


def log(*a): print(*a, flush=True)


def load_daily(snaps=SNAPS, start='2025-04-01', end='2026-03-31'):
    """Bảng tàu-ngày (gate, mớn nước, điểm đến Nga) và tên/callsign, đã gắn vessel_id theo cohort của tháng."""
    mp = []
    for s in snaps:
        f = pd.read_parquet(ROOT/f'data/features/snapshot={s}/features.parquet', columns=['vessel_id', 'mmsis'])
        mp.append(f.assign(MMSI=f.mmsis.map(json.loads)).explode('MMSI')[['MMSI', 'vessel_id']].assign(month=s))
    mp = pd.concat(mp).drop_duplicates(['MMSI', 'month'])
    V, N = [], []
    for d in [str(x.date()) for x in pd.date_range(start, end)]:
        V.append(pd.read_parquet(DAY/f'date={d}'/'vessel', columns=COLS).assign(date=d, month=d[:4]+d[5:7]))
        n = pd.read_parquet(DAY/f'date={d}'/'values', filters=[('field', 'in', ['Name', 'Callsign'])], columns=['MMSI', 'field', 'value'])
        N.append(n[~n.value.isin(PLACEHOLDERS)].assign(date=d, month=d[:4]+d[5:7]))
    V = pd.concat(V, ignore_index=True).merge(mp, on=['MMSI', 'month'])
    N = pd.concat(N, ignore_index=True).merge(mp, on=['MMSI', 'month'])
    V['out'] = V[[c for c in COLS if c.endswith('_out')]].sum(axis=1) > 0
    V['in'] = V[[c for c in COLS if c.endswith('_in')]].sum(axis=1) > 0
    V['t'] = pd.to_datetime(V.date); N['t'] = pd.to_datetime(N.date)
    log('vessel-days', len(V), 'names/callsigns', len(N))
    return V, N


def med_gap(ts):
    ts = np.sort(ts.unique())
    return np.median(np.diff(ts).astype('timedelta64[D]').astype(float)) if len(ts) >= 2 else np.nan


def stay(g):
    """Trung vị số ngày từ lần đi vào Baltic tới lần đi ra kế tiếp (thời gian ở trong Baltic, gồm bốc hàng)."""
    ins, outs = np.sort(g.loc[g['in'], 't'].unique()), np.sort(g.loc[g.out, 't'].unique())
    r = [(outs[outs > i][0]-i).astype('timedelta64[D]').astype(float) for i in ins if (outs > i).any()]
    return np.median(r) if r else np.nan


def compute(C, cohort, V, N):
    """Feature hành trình tại cutoff C (Timestamp) cho các vessel_id trong cohort, chỉ dùng dữ liệu (C-LOOK, C]."""
    C = pd.Timestamp(C)
    start = max(pd.Timestamp('2025-04-01'), C - pd.Timedelta(days=LOOK-1))
    hist = (C - start).days + 1
    w = V[(V.t >= start) & (V.t <= C) & V.vessel_id.isin(cohort)]
    day = w.groupby(['vessel_id', 'date']).agg(out=('out', 'max'), **{'in': ('in', 'max')}, dmax=('draught_max', 'max'),
                                                  t=('t', 'first'), rows=('n_rows', 'sum'), ru=('n_dest_ru', 'sum')).reset_index()
    g = day.groupby('vessel_id')
    f = pd.DataFrame(index=g.size().index)
    f['v_present_per30'] = g.size()/hist*30
    f['v_out_per30'] = g.out.sum()/hist*30
    f['v_in_per30'] = g['in'].sum()/hist*30
    dout = day[day.out].groupby('vessel_id').dmax.median(); din = day[day['in']].groupby('vessel_id').dmax.median()
    f['v_draught_out_minus_in'] = dout - din
    f['v_draught_out_rel'] = dout / g.dmax.max()
    f['v_cycle_days'] = day[day.out].groupby('vessel_id').t.apply(med_gap)
    f['v_baltic_stay_days'] = day.groupby('vessel_id')[['t', 'in', 'out']].apply(stay)
    f['v_dest_ru_share'] = g.ru.sum()/g.rows.sum().where(g.rows.sum() > 0)
    nn = N[(N.t >= start) & (N.t <= C) & N.vessel_id.isin(cohort)]
    f['v_n_names'] = nn[nn.field == 'Name'].groupby('vessel_id').value.nunique()
    f['v_n_callsigns'] = nn[nn.field == 'Callsign'].groupby('vessel_id').value.nunique()
    mm = w.groupby('vessel_id').MMSI.agg(lambda x: set(x))
    f['v_n_mmsis'] = mm.map(len); f['v_n_mids'] = mm.map(lambda x: len({m[:3] for m in x}))   # đổi cờ = MID khác
    return f.reindex(pd.Index(cohort, name='vessel_id')).reset_index(), hist


if __name__ == '__main__':
    V, N = load_daily()
    for s in SNAPS:
        cohort = pd.read_parquet(ROOT/f'data/features/snapshot={s}/features.parquet', columns=['vessel_id']).vessel_id
        f, hist = compute(pd.Period(s, 'M').end_time.normalize(), cohort, V, N)
        f.to_parquet(OUT/f'snapshot={s}.parquet', index=False)
        log(s, len(f), 'hist_days', hist, {c: round(float(f[c].notna().mean()), 2) for c in ['v_draught_out_minus_in', 'v_cycle_days', 'v_baltic_stay_days']})
