"""Normalize identities and legal listing dates; never use crawler first_seen."""
import calendar
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
import pandas as pd
from common import ROOT, checkpoint, save_json

SURVEY = {'2024-10':22, '2024-11':25, '2024-12':43, '2025-02':66, '2025-05':214,
          '2025-06':17, '2025-07':49, '2025-09':61, '2025-10':75, '2025-12':16,
          '2026-02':3, '2026-04':32, '2026-06':4, '2026-07':37, '2026-08':6, '2026-10':6}
SOURCES = ['eu_fsf', 'eu_sanctions_map', 'eu_journal_sanctions', 'gb_fcdo_sanctions']


def normalize_imo(value):
    value = re.sub(r'^IMO\s*', '', str(value).strip(), flags=re.I)
    if not re.fullmatch(r'[1-9][0-9]{6}', value):
        return None
    return value if sum(int(value[i]) * (7-i) for i in range(6)) % 10 == int(value[-1]) else None


def valid_mmsi(value):
    return bool(re.fullmatch(r'[0-9]{9}', str(value)) and 201 <= int(str(value)[:3]) <= 775)


def cutoff_window(snapshot):
    y, m = int(str(snapshot)[:4]), int(str(snapshot)[4:6])
    cutoff = date(y, m, calendar.monthrange(y, m)[1])
    total = y*12 + m-1 + 6
    ey, em = divmod(total, 12)
    em += 1
    end = date(ey, em, min(cutoff.day, calendar.monthrange(ey, em)[1]))
    return cutoff, end


def label_status(first_listed, snapshot):
    cutoff, end = cutoff_window(snapshot)
    listed = date.fromisoformat(str(first_listed)[:10])
    return 'already' if listed <= cutoff else 'positive' if listed <= end else 'later'


def build(sources):
    vessels = defaultdict(list)
    sanctions = defaultdict(list)
    diagnostics = {}
    for source in sources:
        entities = [json.loads(x) for x in (ROOT/f'data/labels/{source}/entities.ftm.json').read_text().splitlines()]
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


def main():
    frame, diagnostics = build(SOURCES)
    dated = frame[frame.first_listed.notna()]
    dated.to_parquet(ROOT/'data/labels/vessels_first_listed.parquet', index=False)
    frame[frame.first_listed.isna()].to_csv(ROOT/'results/labels_undated.csv', index=False)
    frame[frame.n_entity_ids>1].to_csv(ROOT/'results/labels_merged.csv', index=False)
    survey_frame, reference_diag = build(['eu_sanctions_map', 'gb_fcdo_sanctions'])
    final_counts = Counter(dated.first_listed.str[:7])
    reference_counts = Counter(survey_frame.first_listed.dropna().str[:7])
    rows=[]
    for period in pd.period_range('2016-01','2026-10',freq='M'):
        month=str(period)
        survey = SURVEY.get(month,0) if month >= '2024-10' else None
        rows.append(dict(month=month, n_first_listed_eu_uk=final_counts[month],
            n_map_uk_reproduction=reference_counts[month], survey_n=survey,
            delta_survey=final_counts[month]-survey if survey is not None else None))
    pd.DataFrame(rows).to_csv(ROOT/'results/label_monthly.csv', index=False)
    # Independently inspect Journal coverage; keep comparison visible even when reference differs.
    base, base_diag = build(['eu_fsf', 'eu_sanctions_map', 'gb_fcdo_sanctions'])
    frame[~frame.vessel_id.isin(base.vessel_id)].to_csv(ROOT/'results/labels_journal_additional.csv',index=False)
    save_json(ROOT/'results/label_diagnostics.json', dict(selected_sources=SOURCES, selected=diagnostics,
        survey_reproduction=reference_diag, without_journal=base_diag,
        date_rule='listingDate, otherwise startDate; never first_seen',
        identity_rule='canonical entity ID, union by checksum-valid IMO'))
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(pd.DataFrame(rows).query("month >= '2024-10'").to_string(index=False))
    checkpoint('Bảng nhãn', f'{len(dated)} tàu có ngày pháp lý; {len(frame)-len(dated)} tàu thiếu ngày. '
               f'{diagnostics["union"]["duplicate_ids_collapsed"]} ID được gộp theo IMO.',
               'Kiểm thử biên nhãn, khớp định danh và xử lý thử một ngày AIS.')


if __name__ == '__main__':
    main()
