"""Nhãn trừng phạt EU/UK (OpenSanctions): tải bản mới nhất và dựng bảng tàu + ngày bị liệt kê lần đầu.

Ở môi trường chạy, nhãn chỉ dùng để (1) loại tàu ĐÃ bị cấm tại ngày chấm (đúng định nghĩa lúc train),
(2) hiển thị tình trạng cấm. Ngày pháp lý: listingDate, nếu không có thì startDate; không dùng first_seen.
"""
import json
import re
from collections import defaultdict
import pandas as pd
import requests
from .config import DATA, LABEL_SOURCES, log

LABELS = DATA/'data/labels'


def download(name):
    d = LABELS/name; d.mkdir(parents=True, exist_ok=True)
    index = requests.get(f'https://data.opensanctions.org/datasets/latest/{name}/index.json', timeout=90).json()
    res = next(r for r in index['resources'] if r['name'] == 'entities.ftm.json')
    tmp = d/'entities.ftm.json.part'
    with requests.get(res['url'], stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(1 << 20): f.write(chunk)
    assert tmp.stat().st_size == res['size'], f'{name}: sai kích thước'
    tmp.replace(d/'entities.ftm.json'); (d/'index.json').write_text(json.dumps(index))
    log('labels', name, 'last_change', index.get('last_change'))


def refresh(fetch=True):
    """Tải lại 4 nguồn (nếu fetch) rồi dựng data/labels/vessels_first_listed.parquet. Lỗi mạng -> dùng bản cũ."""
    if fetch:
        for name in LABEL_SOURCES:
            try: download(name)
            except Exception as exc:
                if not (LABELS/name/'entities.ftm.json').exists(): raise
                log('labels', name, 'tải lỗi, dùng bản cũ:', type(exc).__name__)
    frame, _ = build(LABEL_SOURCES)
    frame[frame.first_listed.notna()].to_parquet(LABELS/'vessels_first_listed.parquet', index=False)
    return LABELS/'vessels_first_listed.parquet'


def load():
    return pd.read_parquet(LABELS/'vessels_first_listed.parquet')


def normalize_imo(value):
    value = re.sub(r'^IMO\s*', '', str(value).strip(), flags=re.I)
    if not re.fullmatch(r'[1-9][0-9]{6}', value):
        return None
    return value if sum(int(value[i]) * (7-i) for i in range(6)) % 10 == int(value[-1]) else None


def valid_mmsi(value):
    return bool(re.fullmatch(r'[0-9]{9}', str(value)) and 201 <= int(str(value)[:3]) <= 775)


def build(sources):
    vessels = defaultdict(list)
    sanctions = defaultdict(list)
    diagnostics = {}
    for source in sources:
        entities = [json.loads(x) for x in (LABELS/source/'entities.ftm.json').read_text().splitlines()]
        ids = {e['id'] for e in entities if e['schema'] == 'Vessel'}
        diagnostics[source] = {'vessels':len(ids)}
        for entity in entities:
            if entity['schema'] == 'Vessel':
                vessels[entity['id']].append((source, entity))
            elif entity['schema'] == 'Sanction':
                p = entity['properties']
                # listingDate is the original designation; startDate is used only if absent.
                dates = p.get('listingDate') or p.get('startDate', [])
                dates = [d for d in dates if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d)]
                for id in p.get('entity', []):
                    if id in ids:
                        sanctions[id].extend(dict(source=source, sanction_id=entity['id'], date=d,
                            date_property='listingDate' if p.get('listingDate') else 'startDate') for d in dates)
    parent = {id:id for id in vessels}
    def root(id):
        while parent[id] != id:
            parent[id] = parent[parent[id]]
            id = parent[id]
        return id
    by_imo = {}
    for id, entries in vessels.items():
        for _, e in entries:
            for raw in e['properties'].get('imoNumber', []):
                imo = normalize_imo(raw)
                if imo:
                    if imo in by_imo:
                        parent[root(id)] = root(by_imo[imo])
                    by_imo[imo] = id
    groups = defaultdict(list)
    for id in vessels:
        groups[root(id)].append(id)
    records = []
    for ids in groups.values():
        entries = [entry for id in ids for entry in vessels[id]]
        properties = [e['properties'] for _, e in entries]
        imos = sorted({i for p in properties for v in p.get('imoNumber', []) if (i:=normalize_imo(v))})
        evidence = [s for id in ids for s in sanctions[id]]
        records.append(dict(vessel_id='IMO'+imos[0] if len(imos)==1 else min(ids),
            imo=imos[0] if len(imos)==1 else None, imos=json.dumps(imos),
            mmsis=json.dumps(sorted({m for p in properties for m in p.get('mmsi', []) if valid_mmsi(m)})),
            name=min({n for p in properties for n in p.get('name', [])}, default=''),
            names=json.dumps(sorted({n for p in properties for n in p.get('name', [])}), ensure_ascii=False),
            label_types=json.dumps(sorted({n for p in properties for n in p.get('type', [])})),
            first_listed=min((e['date'] for e in evidence), default=None),
            sources=json.dumps(sorted({s for s, _ in entries})), entity_ids=json.dumps(sorted(ids)),
            date_evidence=json.dumps(evidence, sort_keys=True), n_entity_ids=len(ids), n_imos=len(imos)))
    frame = pd.DataFrame(records).sort_values('vessel_id')
    diagnostics['union'] = dict(raw_ids=len(vessels), deduplicated=len(frame),
        dated=int(frame.first_listed.notna().sum()), undated=int(frame.first_listed.isna().sum()),
        duplicate_ids_collapsed=len(vessels)-len(frame), multi_imo=int((frame.n_imos>1).sum()))
    return frame, diagnostics
