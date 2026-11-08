"""Danh tính tàu: gom MMSI thành tàu theo IMO chiếm đa số trong tháng (giống hệt lúc train, ais_identity_only=True).

month_map(month) đọc data/vessel_day của các ngày đã có trong tháng, cache ở data/identity/month=YYYYMM.parquet
(tháng chưa đủ ngày được tính lại mỗi lần chạy).
"""
import json
from collections import Counter, defaultdict
import pandas as pd
from .config import DATA
from .labels import normalize_imo, valid_mmsi

VDAY = DATA/'data/vessel_day'; CACHE = DATA/'data/identity'


def mode(counts):
    return min(counts, key=lambda k: (-counts[k], k)) if counts else None


def is_tanker(value):
    text = str(value).strip()
    return 'tanker' in text.lower() or (text.isdigit() and 80 <= int(text) <= 89)


def match_identity(imo_counts, mmsi, imo_index, mmsi_index):
    valid = {i:n for i,n in imo_counts.items() if normalize_imo(i)}
    chosen = mode(valid)
    tied = chosen and sum(n == valid[chosen] for n in valid.values()) > 1
    if tied:
        return None, 'ambiguous_imo', None
    if chosen:
        # A different valid IMO blocks a potentially recycled label MMSI.
        label = imo_index.get(chosen)
        return label, 'imo' if label else 'valid_imo_unmatched', chosen
    candidates = mmsi_index.get(mmsi, set())
    if len(candidates) == 1:
        return next(iter(candidates)), 'mmsi_only', None
    return None, 'ambiguous_mmsi' if candidates else 'unmatched', None


def label_indexes(labels):
    imos, mmsis = {}, defaultdict(set)
    for r in labels.to_dict('records'):
        for i in json.loads(r['imos']):
            assert i not in imos or imos[i] == r['vessel_id']
            imos[i] = r['vessel_id']
        for m in json.loads(r['mmsis']):
            mmsis[m].add(r['vessel_id'])
    return imos, mmsis


def month_vessels(paths, labels, *, ais_identity_only=False):
    imos, mmsis = label_indexes(labels)
    by_mmsi = {}
    hists = ['mobile_counts','imo_counts','name_counts','callsign_counts','ship_type_counts']
    for path in paths:
        date = path.stem.removeprefix('date=')
        for r in pd.read_parquet(path).to_dict('records'):
            m = r['MMSI']
            if not valid_mmsi(m):
                continue
            record = by_mmsi.setdefault(m, dict(n_valid_pos=0, n_msgs=0, days=set(),
                **{h:Counter() for h in hists}))
            for h in hists:
                record[h].update(json.loads(r[h]))
            record['n_valid_pos'] += r['n_valid_pos']
            record['n_msgs'] += r['n_msgs']
            if r['n_valid_pos']:
                record['days'].add(date)
    vessels = {}
    quality = Counter()
    for m, r in by_mmsi.items():
        if mode(r['mobile_counts']) != 'Class A':
            continue
        label, method, dominant = match_identity(r['imo_counts'], m, imos, mmsis)
        quality[method] += 1
        if len(r['imo_counts']) > 1:
            quality['multiple_valid_imo'] += 1
        # Gate 2 must not merge AIS records because of future sanction identity.
        key = 'IMO'+dominant if dominant else ('MMSI'+m if ais_identity_only else label or 'MMSI'+m)
        vessel = vessels.setdefault(key, dict(vessel_id=key, label_id=label, imo=dominant,
            mmsis=set(), methods=set(), days=set(), n_valid_pos=0, n_msgs=0,
            **{h:Counter() for h in hists}))
        assert vessel['label_id'] == label, 'Identity grouping conflict'
        for h in hists:
            vessel[h].update(r[h])
        vessel['mmsis'].add(m)
        vessel['methods'].add(method)
        vessel['days'].update(r['days'])
        vessel['n_valid_pos'] += r['n_valid_pos']
        vessel['n_msgs'] += r['n_msgs']
    for r in vessels.values():
        r['ship_type'] = mode(r['ship_type_counts'])
        r['name'] = mode(r['name_counts'])
        r['is_tanker'] = is_tanker(r['ship_type'])
        r['match_method'] = 'imo' if 'imo' in r['methods'] else 'mmsi_only' if 'mmsi_only' in r['methods'] else mode(Counter(r['methods']))
    return list(vessels.values()), dict(quality)


def month_map(month, labels):
    """month 'YYYYMM' -> DataFrame MMSI, vessel_id, label_id, is_tanker, name, n_valid_pos (mọi tàu Class A)."""
    p = pd.Period(month, 'M')
    days = [str(d.date()) for d in pd.date_range(p.start_time, p.end_time)]
    have = [d for d in days if (VDAY/f'date={d}.parquet').exists()]
    f = CACHE/f'month={month}.parquet'
    if f.exists():
        old = pd.read_parquet(f)
        if old.attrs.get('n_days') == len(have) or ('n_days' in old.columns and int(old.n_days.iloc[0]) == len(have)):
            return old
    vs, _ = month_vessels([VDAY/f'date={d}.parquet' for d in have], labels, ais_identity_only=True)
    rows = [dict(MMSI=m, vessel_id=v['vessel_id'], label_id=v['label_id'] or '', is_tanker=bool(v['is_tanker']), name=v['name'] or '',
                 n_valid_pos=int(v['n_valid_pos'])) for v in vs for m in sorted(v['mmsis'])]
    out = pd.DataFrame(rows, columns=['MMSI', 'vessel_id', 'label_id', 'is_tanker', 'name', 'n_valid_pos']).assign(n_days=len(have))
    CACHE.mkdir(parents=True, exist_ok=True); out.to_parquet(f, index=False)
    return out
