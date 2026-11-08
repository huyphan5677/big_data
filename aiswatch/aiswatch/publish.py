"""Xuất kết quả cho 2 giao diện.

- data/app/*.parquet   : bảng nhỏ cho app kết quả (cổng 8502): tàu, vị trí theo ngày, lượt qua gate, phiên gặp nhau.
- web/data (DATA/web)  : dữ liệu tĩnh cho trang giám sát (cổng 8501): meta, lịch sử điểm, mỗi ngày một file
                         (danh sách theo dõi + lý do, quỹ đạo 10 phút, sự kiện luật theo phút: qua gate, mất tín hiệu ≥2 giờ).
"""
import json, shutil
import numpy as np, pandas as pd
from .config import DATA, PKG, WATCH_K, log
from .predict import FE, META
from . import identity

APP = DATA/'data/app'
WEB = DATA/'web'
DAY = DATA/'data/features_day'
txt = lambda x: x if isinstance(x, str) else ''
GATES = [dict(id='greatbelt', name='Great Belt', axis='lat', v=55.34, lo=10.85, hi=11.25, out_dir=+1),
         dict(id='oresund', name='Øresund', axis='lat', v=55.98, lo=12.55, hi=12.80, out_dir=+1),
         dict(id='skagen', name='Skagen', axis='lon', v=10.65, lo=57.75, hi=58.60, out_dir=-1)]
BOX = (54.0, 59.0, 7.0, 15.5)  # lat_min, lat_max, lon_min, lon_max: vùng có sóng AIS Đan Mạch (xấp xỉ)

PHRASE = {
    'distance_nm_l1m': lambda v: f'chạy xa {v:.0f} hải lý/ngày',
    'port_l1m': lambda v: 'không ghé cảng' if v < 0.01 else (f'ít ghé cảng ({v:.0%})' if v < 0.1 else None),
    'max_gap_hours_l1m': lambda v: f'AIS liên tục (mất tối đa {v:.1f} giờ)' if v < 3 else f'có lúc mất tín hiệu {v:.0f} giờ',
    'mean_sog_l1m': lambda v: f'tốc độ TB {v:.1f} kn',
    'dest_vague_l1m': lambda v: f'điểm đến mơ hồ {v:.0%}' if v > 0.05 else None,
    'v_draught_out_rel': lambda v: f'đi ra đầy hàng ({v:.0%} mớn tối đa)' if v >= 0.95 else None,
    'v_draught_out_minus_in': lambda v: f'ra sâu hơn vào {v:.1f} m' if v > 1 else None,
    'v_baltic_stay_days': lambda v: f'ở trong Baltic {v:.0f} ngày/lượt',
    'v_in_per30': lambda v: f'vào Baltic {v:.1f} lượt/30 ngày',
    'v_dest_ru_share': lambda v: f'khai cảng Nga ({v:.0%})' if v > 0 else None}


def reasons(r, n=3):
    out = []
    for v, f in sorted(((r[f'shap_{f}'], f) for f in FE), reverse=True):
        if v <= 0 or len(out) == n: break
        t = None if pd.isna(r[f]) else PHRASE[f](r[f])
        if t: out.append(t)
    return out



def gate_events(tr):
    """Phát hiện lượt qua gate trên quỹ đạo 10 phút (cặp điểm liên tiếp ≤ 1 giờ, cắt đường gate)."""
    ev = []
    for vid, g in tr.groupby('vessel_id'):
        g = g.sort_values('t'); a, b = g.iloc[:-1], g.iloc[1:]
        ok = (b.t.values - a.t.values) <= 3600
        for gt in GATES:
            k = 'lat' if gt['axis'] == 'lat' else 'lon'; o = 'lon' if k == 'lat' else 'lat'
            x0, x1 = a[k].values - gt['v'], b[k].values - gt['v']
            cross = ok & (x0*x1 < 0)
            for i in np.where(cross)[0]:
                f = x0[i]/(x0[i]-x1[i]); oc = a[o].values[i] + f*(b[o].values[i]-a[o].values[i])
                if gt['lo'] <= oc <= gt['hi']:
                    direction = 'out' if np.sign(x1[i]-x0[i]) == gt['out_dir'] else 'in'
                    dr = b.draught.values[i]
                    ev.append(dict(t=int(a.t.values[i] + f*(b.t.values[i]-a.t.values[i])), id=vid, type='gate', gate=gt['name'], dir=direction,
                                   draught=None if pd.isna(dr) else float(dr)))
        # mất tín hiệu ≥ 2 giờ khi cả hai đầu nằm trong vùng phủ sóng
        dt = b.t.values - a.t.values
        inside = lambda x: (x.lat.values >= BOX[0]) & (x.lat.values <= BOX[1]) & (x.lon.values >= BOX[2]) & (x.lon.values <= BOX[3])
        for i in np.where((dt >= 7200) & inside(a) & inside(b))[0]:
            ev.append(dict(t=int(b.t.values[i]), id=vid, type='gap', hours=round(dt[i]/3600, 1),
                           lat=float(b.lat.values[i]), lon=float(b.lon.values[i])))
    return ev




def app_tables(P, labels, dates):
    """Bảng cho app kết quả, dựng lại từ đầu mỗi lần (nhỏ)."""
    APP.mkdir(parents=True, exist_ok=True)
    listed = dict(zip(labels.vessel_id, labels.first_listed.str[:10]))
    months = sorted({d[:4]+d[5:7] for d in dates})
    M = pd.concat([identity.month_map(m, labels).assign(month=m) for m in months]).sort_values('month').drop_duplicates('MMSI', keep='last')
    M['first_listed'] = M.label_id.map(listed).fillna('')
    V = (M.groupby('vessel_id').agg(name=('name', 'last'), mmsis=('MMSI', lambda s: ','.join(sorted(s))), label_id=('label_id', 'max'),
                                   first_listed=('first_listed', 'max'), is_tanker=('is_tanker', 'max')).reset_index())
    V.to_parquet(APP/'vessels.parquet', index=False)
    mm = M[M.vessel_id.isin(set(P.vessel_id))][['MMSI', 'vessel_id']]
    pos, cr, enc = [], [], []
    gc = [f'n_{g}_{x}' for g in ('greatbelt', 'oresund', 'skagen') for x in ('out', 'in')]
    info = M.set_index('MMSI')
    for d in dates:
        x = pd.read_parquet(DAY/f'date={d}'/'vessel')
        pos.append(x[['MMSI', 'first_t', 'last_t', 'first_lat', 'first_lon', 'last_lat', 'last_lon', 'distance_nm', 'n_bins', 'n_bins_port', 'n_msgs']]
                   .merge(mm, on='MMSI').assign(date=d))
        x['out'] = x[[c for c in gc if c.endswith('out')]].sum(axis=1); x['in'] = x[[c for c in gc if c.endswith('in')]].sum(axis=1)
        x = x[(x.out > 0) | (x['in'] > 0)]
        cr.append(x[['MMSI', 'draught_max', 'out', 'in']].merge(mm, on='MMSI').assign(date=d))
        e = pd.read_parquet(DAY/f'date={d}'/'encounters', columns=['m1', 'm2', 'first_bin', 'last_bin', 'lat', 'lon'])
        e = e[(e.last_bin-e.first_bin)*5 >= 30]
        enc.append(e.assign(date=d, hours=(e.last_bin-e.first_bin)/12).drop(columns=['first_bin', 'last_bin']))
    pos = pd.concat(pos, ignore_index=True); pos['port_frac'] = pos.n_bins_port/pos.n_bins.where(pos.n_bins > 0)
    pos.to_parquet(APP/'daily_pos.parquet', index=False)
    pd.concat(cr, ignore_index=True).to_parquet(APP/'crossings.parquet', index=False)
    E = pd.concat(enc, ignore_index=True)
    for k in ('1', '2'):
        m = E[f'm{k}']
        E[f'v{k}'] = m.map(info.vessel_id).fillna('MMSI'+m)
        E[f'tanker{k}'] = m.map(info.is_tanker).fillna(False).astype(bool)
        fl = m.map(info.first_listed).fillna('')
        E[f'status{k}'] = [('already' if f <= d else 'future') if f else 'none' for f, d in zip(fl, E.date)]
    E.to_parquet(APP/'encounters.parquet', index=False)
    return V


def web(P, V, mode):
    """Dữ liệu tĩnh cho trang giám sát; chỉ ghi lại file ngày còn thiếu (trừ ngày cuối, luôn ghi lại)."""
    out = WEB/'data'; (out/'day').mkdir(parents=True, exist_ok=True)
    if not (WEB/'index.html').exists() or (PKG/'web/index.html').stat().st_mtime > (WEB/'index.html').stat().st_mtime:
        shutil.copy(PKG/'web/index.html', WEB/'index.html')
    V = V.set_index('vessel_id'); P = P.copy(); P['date'] = P.date.astype(str)
    dates = sorted(P.date.unique()); scored = sorted(P.vessel_id.unique())
    tracked = [d for d in dates if (APP/f'tracks/date={d}.parquet').exists()]
    meta = dict(mode=mode, dates=tracked, watch_k=WATCH_K, default_day=tracked[-1] if tracked else None,
                gates=[dict(id=g['id'], name=g['name'], path=[[g['lo'], g['v']], [g['hi'], g['v']]] if g['axis'] == 'lat' else [[g['v'], g['lo']], [g['v'], g['hi']]])
                       for g in GATES],
                vessels={v: dict(n=txt(V.name.get(v)) or v, l=txt(V.first_listed.get(v))) for v in scored},
                model=dict(algorithm=META['algorithm'], train=META['train_snapshots']))
    (out/'meta.json').write_text(json.dumps(meta, ensure_ascii=False, allow_nan=False))
    di = {d: i for i, d in enumerate(tracked)}
    G = pd.read_parquet(APP/'crossings.parquet'); hist = {}
    for vid, g in P[P.date.isin(di)].groupby('vessel_id'):
        hist[vid] = dict(s=[[di[d], round(float(s), 4), int(r)] for d, s, r in zip(g.date, g.score, g.rank_all)])
    for vid, g in G.groupby('vessel_id'):
        if vid in hist:
            hist[vid]['x'] = [[d, 'o' if o > 0 else 'i', None if pd.isna(dr) else float(dr)] for d, o, dr in zip(g.date, g.out, g.draught_max)]
    (out/'history.json').write_text(json.dumps(hist, separators=(',', ':'), allow_nan=False))
    mm = {m: vid for vid in scored for m in txt(V.mmsis.get(vid)).split(',') if m}
    prev = None
    for day in tracked:
        f = out/'day'/f'{day}.json'
        tr = pd.read_parquet(APP/f'tracks/date={day}.parquet'); tr['vessel_id'] = tr.MMSI.map(mm)
        tr = tr.dropna(subset=['vessel_id', 'lat', 'lon']).sort_values(['vessel_id', 't']).drop_duplicates(['vessel_id', 'bin'], keep='last')
        if f.exists() and day != tracked[-1]:
            prev = tr[tr.t >= tr.t.max() - 3600]; continue
        both = pd.concat([prev, tr]) if prev is not None else tr
        t0 = int(pd.Timestamp(day, tz='UTC').timestamp())
        ev = [e for e in gate_events(both) if e['t'] >= t0]
        d = P[P.date == day]
        watch = [dict(id=r.vessel_id, rank=int(r.rank_watch), score=round(float(r.score), 4), y=int(r.y), why=reasons(r))
                 for _, r in d[d.rank_watch <= 40].sort_values('rank_watch').iterrows()]
        rank = {v: int(r) for v, r in zip(d.vessel_id, d.rank_watch) if not pd.isna(r)}
        for e in ev: e['rank'] = rank.get(e['id'])
        tracks = [dict(id=vid, p=np.round(g[['lon', 'lat']].values, 4).ravel().tolist(), t=((g.t.values - t0)//60).tolist(),
                       dr=None if g.draught.isna().all() else float(g.draught.dropna().iloc[-1])) for vid, g in tr.groupby('vessel_id')]
        res = dict(date=day, t0=t0, watch=watch, n_scored=len(d), n_present=int((d.days_since_seen < 3).sum()),
                   score={v: round(float(s), 4) for v, s in zip(d.vessel_id, d.score)}, rank=rank, tracks=tracks, events=sorted(ev, key=lambda e: e['t']))
        f.write_text(json.dumps(res, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
        prev = tr[tr.t >= tr.t.max() - 3600]
    log('web: ngày có dữ liệu', len(tracked))
