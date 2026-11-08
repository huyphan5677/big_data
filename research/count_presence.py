"""Full-month counts from reconciled vessel/day summaries (PLAN_V2)."""
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from common import ROOT, checkpoint, save_json, completed_day_rows
from labels import label_status, normalize_imo, valid_mmsi

SAMPLE_DAYS = {3, 8, 13, 18, 23, 28}


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


def counts(vessels, labels_by_id, snapshot, threshold):
    eligible = [r for r in vessels if r['n_valid_pos'] >= threshold]
    positives, already = [], []
    for r in eligible:
        if r['label_id']:
            status = label_status(labels_by_id[r['label_id']]['first_listed'], snapshot)
            if status == 'positive':
                positives.append(r)
            if status == 'already':
                already.append(r)
    clean = [r for r in positives if labels_by_id[r['label_id']]['first_listed'] >= '2026-07-01']
    result = dict(n_vessels=len(eligible), n_tankers=sum(r['is_tanker'] for r in eligible),
        n_already_sanctioned=len(already), n_pos=len(positives), n_pos_tanker=sum(r['is_tanker'] for r in positives),
        n_pos_clean=len(clean), n_pos_clean_tanker=sum(r['is_tanker'] for r in clean),
        n_pos_imo_match=sum(r['match_method']=='imo' for r in positives),
        n_pos_mmsi_only=sum(r['match_method']=='mmsi_only' for r in positives))
    assert len({r['label_id'] for r in positives}) == len(positives), 'Repeated positive vessel'
    return result, positives


def main():
    labels = pd.read_parquet(ROOT/'data/labels/vessels_first_listed.parquet')
    labels_by_id = labels.set_index('vessel_id').to_dict('index')
    stats = pd.read_csv(ROOT/'results/day_stats.csv')
    done = set(completed_day_rows(stats).date)
    rows, recall, positives, coverage, quality = [], [], [], [], []
    for period in pd.period_range('2025-04','2026-03',freq='M'):
        snapshot = str(period).replace('-','')
        dates = pd.date_range(period.start_time, period.end_time)
        test = snapshot >= '202601'
        requested = {str(d.date()) for d in dates}
        available = sorted(requested & done)
        coverage.append(dict(snapshot=snapshot, expected_days=len(requested), days_ok=len(available),
                             missing_dates=';'.join(sorted(requested-done))))
        if not available:
            continue
        paths = [ROOT/'data/vessel_day'/f'date={d}.parquet' for d in available]
        vessels, diag = month_vessels(paths, labels)
        quality.append(dict(snapshot=snapshot, **diag))
        count, pos = counts(vessels, labels_by_id, snapshot, 10)
        sensitivity, _ = counts(vessels, labels_by_id, snapshot, 1)
        theoretical = sum(label_status(d, snapshot)=='positive' for d in labels.first_listed)
        role = 'test' if test else 'train_pa1' if snapshot <= '202506' else 'train_pa2'
        method = 'full' if len(available)==len(requested) else 'partial'
        row = dict(snapshot=snapshot, role=role, method=method, n_days=len(available), is_estimate=method!='full', role_v2='train_v2' if snapshot<='202506' else 'test_v2' if snapshot<='202509' else 'ref', **count,
            n_pos_theoretical=theoretical, pos_presence_fraction=count['n_pos']/theoretical if theoretical else None,
            **{k+'_ge1':v for k,v in sensitivity.items()})
        if not test:
            for key in ['n_pos_clean','n_pos_clean_tanker','n_pos_clean_ge1','n_pos_clean_tanker_ge1']:
                row[key] = None
        rows.append(row)
        if test or snapshot in ['202507','202508','202509']:
            for r in pos:
                label = labels_by_id[r['label_id']]
                positives.append(dict(snapshot=snapshot, vessel_id=r['label_id'], imo=label['imo'],
                    name=r['name'] or label['name'], label_name=label['name'], ship_type=r['ship_type'],
                    is_tanker=r['is_tanker'], first_listed=label['first_listed'], match_method=r['match_method'],
                    n_days_present=len(r['days']), n_valid_pos=r['n_valid_pos'],
                    mmsis=json.dumps(sorted(r['mmsis'])), is_clean=label['first_listed'] >= '2026-07-01'))
        if test:
            sample_paths = [p for p in paths if int(p.stem[-2:]) in SAMPLE_DAYS]
            sample, _ = month_vessels(sample_paths, labels)
            sample_count, _ = counts(sample, labels_by_id, snapshot, 10)
            rr = dict(snapshot=snapshot, n_days_full=len(available), n_days_sample=len(sample_paths), full_complete=method=='full')
            for metric in ['n_vessels','n_tankers','n_pos','n_pos_tanker','n_pos_clean','n_pos_clean_tanker']:
                rr.update({metric+'_full':count[metric], metric+'_sample6':sample_count[metric],
                    metric+'_recall':sample_count[metric]/count[metric] if count[metric] else None})
            recall.append(rr)
    pd.DataFrame(coverage).to_csv(ROOT/'results/coverage.csv',index=False)
    legacy=ROOT/'results/presence_counts_legacy.csv'
    current=ROOT/'results/presence_counts.csv'
    if current.exists() and not legacy.exists():shutil.copyfile(current,legacy)
    pd.DataFrame(rows).to_csv(current,index=False)
    recall_path=ROOT/'results/sample_recall.csv'
    backup=ROOT/'results/sample_recall_legacy.csv'
    if recall_path.exists() and not backup.exists():shutil.copyfile(recall_path,backup)
    if recall:
        old=pd.read_csv(recall_path).to_dict('records') if recall_path.exists() and recall_path.stat().st_size>1 else []
        merged={str(int(r['snapshot'])):r for r in old}
        for r in recall:
            previous=merged.get(str(r['snapshot']))
            if r['full_complete'] or not previous or not previous['full_complete']:merged[str(r['snapshot'])]=r
        pd.DataFrame([r for _,r in sorted(merged.items())]).to_csv(recall_path,index=False)
    positive_columns=['snapshot','vessel_id','imo','name','label_name','ship_type','is_tanker',
        'first_listed','match_method','n_days_present','n_valid_pos','mmsis','is_clean']
    pd.DataFrame(positives,columns=positive_columns).to_csv(ROOT/'results/positives_test.csv',index=False)
    pd.DataFrame(quality).fillna(0).to_csv(ROOT/'results/matching_quality.csv',index=False)
    # PLAN_V2 counts every day; no full-month extrapolation is used.
    checkpoint('Đếm snapshot', f'{len(done)}/365 ngày; {len(rows)} snapshot; {len(positives)} dòng tàu dương test.',
               'Chỉ kết luận đủ tháng khi coverage không còn ngày thiếu.')


if __name__ == '__main__':
    from common import pipeline_lock
    with pipeline_lock('presence'):main()
