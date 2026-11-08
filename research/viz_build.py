"""Dựng dữ liệu tĩnh cho trang giám sát viz/index.html (phát lại 10/2025–03/2026).

viz/data/meta.json      : ngày, gate, danh mục tàu (tên, IMO, ngày bị cấm), nhãn dấu hiệu
viz/data/history.json   : điểm/hạng theo ngày + lượt qua gate (mớn nước) của mọi tàu đã chấm (cho panel chi tiết)
viz/data/day/D.json     : danh sách theo dõi ngày D (lý do bằng lời), điểm mọi tàu, quỹ đạo 10 phút, sự kiện luật
                          (qua gate, mất tín hiệu ≥2 giờ) — mô phỏng lớp cảnh báo theo phút trên dữ liệu phát lại.
"""
import json
import numpy as np, pandas as pd
from train_model import ROOT

OUT = ROOT/'viz/data'; (OUT/'day').mkdir(parents=True, exist_ok=True)
GATES = [dict(id='greatbelt', name='Great Belt', axis='lat', v=55.34, lo=10.85, hi=11.25, out_dir=+1),
         dict(id='oresund', name='Øresund', axis='lat', v=55.98, lo=12.55, hi=12.80, out_dir=+1),
         dict(id='skagen', name='Skagen', axis='lon', v=10.65, lo=57.75, hi=58.60, out_dir=-1)]
BOX = (54.0, 59.0, 7.0, 15.5)  # lat_min, lat_max, lon_min, lon_max: vùng có sóng AIS Đan Mạch (xấp xỉ)
WATCH_K = 30
meta_model = json.load(open(ROOT/'models/final/meta.json')); FE = meta_model['features']
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


txt = lambda x: x if isinstance(x, str) else ''   # NaN -> '' (JSON không có NaN)
P = pd.read_parquet(ROOT/'data/predict_all.parquet'); P['date'] = P.date.astype(str)
V = pd.read_parquet(ROOT/'data/app/vessels.parquet').set_index('vessel_id')
scored = sorted(P.vessel_id.unique())
DATES = sorted(P.date.unique())
mm = {m: vid for vid in scored for m in V.mmsis.get(vid, '').split(',') if m}
meta = dict(dates=DATES, watch_k=WATCH_K, gates=[dict(id=g['id'], name=g['name'],
            path=[[g['lo'], g['v']], [g['hi'], g['v']]] if g['axis'] == 'lat' else [[g['v'], g['lo']], [g['v'], g['hi']]]) for g in GATES],
            vessels={v: dict(n=txt(V.name.get(v)) or v, l=txt(V.first_listed.get(v))) for v in scored},
            model=dict(algorithm=meta_model['algorithm'], train=meta_model['train_snapshots']))
(OUT/'meta.json').write_text(json.dumps(meta, ensure_ascii=False, allow_nan=False))

# lịch sử điểm + lượt qua gate
di = {d: i for i, d in enumerate(DATES)}
G = pd.read_parquet(ROOT/'data/app/crossings.parquet')
hist = {}
for vid, g in P.groupby('vessel_id'):
    hist[vid] = dict(s=[[di[d], round(float(s), 4), int(r)] for d, s, r in zip(g.date, g.score, g.rank_all)])
for vid, g in G.groupby('vessel_id'):
    if vid in hist:
        hist[vid]['x'] = [[d, 'o' if o > 0 else 'i', None if pd.isna(dr) else float(dr)] for d, o, dr in zip(g.date, g.out, g.draught_max)]
(OUT/'history.json').write_text(json.dumps(hist, separators=(',', ':'), allow_nan=False))


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


prev = None
for day in DATES:
    f = ROOT/f'data/app/tracks/date={day}.parquet'
    if not f.exists():
        prev = None; continue
    tr = pd.read_parquet(f); tr['vessel_id'] = tr.MMSI.map(mm); tr = tr.dropna(subset=['vessel_id', 'lat', 'lon'])
    tr = tr.sort_values(['vessel_id', 't']).drop_duplicates(['vessel_id', 'bin'], keep='last')
    both = pd.concat([prev, tr]) if prev is not None else tr                       # nối đêm hôm trước để bắt sự kiện qua 0 giờ
    t0 = int(pd.Timestamp(day, tz='UTC').timestamp())
    ev = [e for e in gate_events(both) if e['t'] >= t0]
    d = P[P.date == day]
    w = d[d.rank_watch <= 40].sort_values('rank_watch')
    watch = [dict(id=r.vessel_id, rank=int(r.rank_watch), score=round(float(r.score), 4), y=int(r.y), why=reasons(r))
             for _, r in w.iterrows()]
    rank = {v: int(r) for v, r in zip(d.vessel_id, d.rank_watch) if not pd.isna(r)}
    for e in ev: e['rank'] = rank.get(e['id'])
    tracks = [dict(id=vid, p=np.round(g[['lon', 'lat']].values, 4).ravel().tolist(), t=((g.t.values - t0)//60).tolist(),
                   dr=None if g.draught.isna().all() else float(g.draught.dropna().iloc[-1]))
              for vid, g in tr.groupby('vessel_id')]
    out = dict(date=day, t0=t0, watch=watch, n_scored=len(d), n_present=int((d.days_since_seen < 3).sum()),
               score={v: round(float(s), 4) for v, s in zip(d.vessel_id, d.score)}, rank=rank, tracks=tracks, events=sorted(ev, key=lambda e: e['t']))
    (OUT/'day'/f'{day}.json').write_text(json.dumps(out, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
    prev = tr[tr.t >= tr.t.max() - 3600]
    print(day, 'tracks', len(tracks), 'events', len(ev), 'watch-gate', sum(e['type'] == 'gate' and (e['rank'] or 99) <= WATCH_K for e in ev), flush=True)
