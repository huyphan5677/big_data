"""Feature gặp gỡ + đồ thị cho mỗi snapshot (pandas + scipy; bản thử nghiệm, sau port sang GraphFrames).

Đồ thị: nút = MMSI (Class A), cạnh = phiên gặp >=30 phút (tốc độ <3 kn, ngoài cảng, cách <500 m; từ features_day),
trọng số = số giờ gặp. Hạt giống = MMSI của tàu ĐÃ bị EU/UK cấm tại cutoff (không dùng nhãn tương lai).
Hai cửa sổ: l1m (tháng snapshot) và cum (từ 2025-04-01 tới cutoff).
Đầu ra: data/features_graph/snapshot=S.parquet (vessel_id + g_*).
"""
import json
from pathlib import Path

import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from count_presence import month_vessels
from train_model import ROOT

DAY = ROOT/'data/features_day'
OUT = ROOT/'data/features_graph'; OUT.mkdir(parents=True, exist_ok=True)
SNAPS = list(pd.period_range('2025-04', '2026-03', freq='M').strftime('%Y%m'))
labels = pd.read_parquet(ROOT/'data/labels/vessels_first_listed.parquet')
listed = dict(zip(labels.vessel_id, labels.first_listed.str[:10]))


def log(*a): print(*a, flush=True)


def days_of(snaps):
    return [str(d.date()) for s in snaps for d in pd.date_range(pd.Period(s, 'M').start_time, pd.Period(s, 'M').end_time)]


# 1. Gặp gỡ >=30 phút của cả năm (khoảng 2 nghìn phiên/ngày)
E = []
for d in days_of(SNAPS):
    e = pd.read_parquet(DAY/f'date={d}'/'encounters', columns=['m1', 'm2', 'first_bin', 'last_bin', 'lat', 'lon'])
    e = e[(e.last_bin-e.first_bin)*5 >= 30]
    E.append(e.assign(date=d, hours=(e.last_bin-e.first_bin)/12)[['m1', 'm2', 'date', 'hours', 'lat', 'lon']])
E = pd.concat(E, ignore_index=True)
log('encounters>=30min', len(E))

# 2. MMSI -> (label_id, tanker) theo từng tháng (cùng cách khớp danh tính với cohort)
mmsi_label, mmsi_tanker = {}, {}
for s in SNAPS:
    vs, _ = month_vessels([ROOT/'data/vessel_day'/f'date={d}.parquet' for d in days_of([s])], labels, ais_identity_only=True)
    for v in vs:
        for m in v['mmsis']:
            if v['label_id']: mmsi_label[m] = v['label_id']
            mmsi_tanker[m] = mmsi_tanker.get(m, False) or bool(v['is_tanker'])
log('mmsi mapped', len(mmsi_tanker), 'labelled', len(mmsi_label))


def graph_feats(e, cutoff, cohort):
    """e: phiên gặp trong cửa sổ; cohort: DataFrame MMSI, vessel_id. Trả feature theo vessel_id."""
    nodes = pd.Index(pd.unique(pd.concat([e.m1, e.m2, cohort.MMSI])))
    i, j = nodes.get_indexer(e.m1), nodes.get_indexer(e.m2)
    n = len(nodes)
    W = sp.coo_matrix((e.hours.values, (i, j)), shape=(n, n)).tocsr()
    W = W + W.T                                                  # vô hướng, tổng giờ gặp
    A = (W > 0).astype(np.float32)
    seed = np.array([listed.get(mmsi_label.get(m), '9999') <= cutoff for m in nodes])
    tank = np.array([mmsi_tanker.get(m, False) for m in nodes])
    deg = np.asarray(A.sum(1)).ravel()
    seed_nb = A @ seed.astype(float)
    seed_hours = W @ seed.astype(float)
    tank_nb = A @ tank.astype(float)
    two = (A @ A) > 0                                            # đi được trong <=2 bước
    two.setdiag(False)
    seed_2hop = np.asarray(two @ seed.astype(float)).ravel() - seed_nb
    # khoảng cách (số bước) tới hạt giống gần nhất, chặn ở 4
    dist = np.where(seed, 0.0, 5.0); front = seed.copy()      # BFS nhiều nguồn, chặn ở 4 bước
    for k in range(1, 5):
        front = (np.asarray(A @ front.astype(float)).ravel() > 0) & (dist == 5)
        dist[front] = k
    ncomp, comp = connected_components(A, directed=False)
    comp_size = np.bincount(comp)[comp]
    comp_seed = np.bincount(comp, weights=seed.astype(float))[comp] > 0
    # personalized PageRank từ hạt giống, trọng số = giờ gặp; báo theo phân vị trong snapshot
    ppr = np.zeros(n)
    if seed.any():
        dinv = 1/np.maximum(np.asarray(W.sum(1)).ravel(), 1e-12)
        P = sp.diags(dinv) @ W
        r = seed/seed.sum(); x = r.copy()
        for _ in range(40):
            x = 0.15*r + 0.85*(P.T @ x)
        ppr = x
    f = pd.DataFrame(dict(MMSI=nodes, deg=deg, seed_nb=seed_nb, seed_hours=seed_hours, seed_2hop=seed_2hop,
                          tanker_nb=tank_nb, dist_seed=dist, comp_size=comp_size, comp_seed=comp_seed.astype(float),
                          ppr=ppr, is_seed=seed))
    f = f[~f.is_seed]                                           # chính tàu hạt giống không phải đối tượng chấm
    f = f.merge(cohort, on='MMSI')
    g = f.groupby('vessel_id').agg(deg=('deg', 'sum'), seed_nb=('seed_nb', 'sum'), seed_hours=('seed_hours', 'sum'),
        seed_2hop=('seed_2hop', 'sum'), tanker_nb=('tanker_nb', 'sum'), dist_seed=('dist_seed', 'min'),
        comp_size=('comp_size', 'max'), comp_seed=('comp_seed', 'max'), ppr=('ppr', 'sum'))
    g['ppr_pct'] = g.ppr.rank(pct=True).where(g.ppr > 0, 0.0)
    return g.drop(columns='ppr')


def session_feats(e, cohort):
    both = pd.concat([e.rename(columns={'m1': 'MMSI', 'm2': 'other'}), e.rename(columns={'m2': 'MMSI', 'm1': 'other'})])
    both = both.merge(cohort, on='MMSI')
    both['tanker'] = both.other.map(lambda m: mmsi_tanker.get(m, False))
    both['long'] = both.hours >= 2
    both['tt_long'] = both.long & both.tanker                   # tàu dầu gặp tàu dầu >=2 giờ (kiểu chuyển tải STS)
    return both.groupby('vessel_id').agg(n_long=('long', 'sum'), n_tt_long=('tt_long', 'sum'),
                                          max_hours=('hours', 'max'), n_sessions=('hours', 'size'))


for s in SNAPS:
    feats = pd.read_parquet(ROOT/f'data/features/snapshot={s}/features.parquet', columns=['vessel_id', 'mmsis'])
    cohort = feats.assign(MMSI=feats.mmsis.map(json.loads)).explode('MMSI')[['MMSI', 'vessel_id']]
    cutoff = str(pd.Period(s, 'M').end_time.date())
    month = set(days_of([s]))
    parts = []
    for tag, e in [('l1m', E[E.date.isin(month)]), ('cum', E[E.date <= cutoff])]:
        g = graph_feats(e, cutoff, cohort).add_prefix(f'g_').add_suffix(f'_{tag}')
        parts.append(g)
        if tag == 'l1m':
            parts.append(session_feats(e, cohort).add_prefix('enc_').add_suffix('_l1m'))
    out = feats[['vessel_id']].set_index('vessel_id').join(parts)
    # tàu không có phiên gặp nào: đếm = 0, khoảng cách = 5 (không tới được), phân vị = 0
    for c in out.columns:
        out[c] = out[c].fillna(5 if c.startswith('g_dist_seed') else 0)
    out.reset_index().to_parquet(OUT/f'snapshot={s}.parquet', index=False)
    log(s, len(out), {c: round(float((out[c] > 0).mean()), 3) for c in out.columns if 'seed' in c or 'tt' in c})
