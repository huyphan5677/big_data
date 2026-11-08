"""Số liệu + hình cho các câu hỏi nghiên cứu của báo cáo (đọc kết quả đã có, không train).

RQ1 hình mẫu hành vi tại điểm nghẽn; RQ3 cảnh báo sớm (từ chấm điểm hằng ngày); RQ4 tàu điểm cao chưa bị EU/UK cấm;
phụ: tàu ĐÃ bị cấm vẫn đi qua Đan Mạch. Đầu ra: report/final/insights.json, report/final/fig/1x_*.png.
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from count_presence import month_vessels
from train_model import ROOT, load

OUT = ROOT/'report/final'; FIG = OUT/'fig'
plt.rcParams.update({'figure.dpi': 130, 'savefig.bbox': 'tight', 'font.size': 9})
R = {}

# ---------- RQ1: hình mẫu (tàu dầu, mọi snapshot) ----------
d = load(); d = d[d.is_tanker]
v = pd.concat([pd.read_parquet(p).assign(snapshot=p.stem.split('=')[1]) for p in sorted((ROOT/'data/features_voyage').glob('*.parquet'))])
d = d.merge(v, on=['snapshot', 'vessel_id'], how='left')
d['grp'] = np.where(d.y == 1, 'Sẽ bị cấm (≤6 tháng)', 'Tàu dầu khác')
cols = [('v_draught_out_minus_in', 'Mớn nước khi ra − khi vào (m)'), ('v_baltic_stay_days', 'Số ngày ở trong Baltic mỗi lượt'),
        ('port_l1m', 'Tỉ lệ thời gian ở cảng'), ('distance_nm_l1m', 'Quãng đường mỗi ngày (hải lý)'),
        ('loiter_l1m', 'Tỉ lệ thời gian đứng yên ngoài cảng'), ('encounter_l1m', 'Số lần gặp tàu khác mỗi ngày')]
R['rq1_medians'] = {c: {g: round(float(x[c].median()), 3) for g, x in d.groupby('grp')} for c, _ in cols}
R['rq1_share_nonzero'] = {c: {g: round(float((x[c].fillna(0) > 0).mean()), 3) for g, x in d.groupby('grp')} for c in ['loiter_l1m', 'encounter_l1m']}
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
for a, (c, t) in zip(ax.ravel()[:5], cols[:5]):
    data = [d.loc[d.grp == g, c].dropna() for g in ['Tàu dầu khác', 'Sẽ bị cấm (≤6 tháng)']]
    hi = np.nanpercentile(pd.concat(data), 98)
    a.boxplot([x.clip(upper=hi) for x in data], showfliers=False, widths=.6)
    a.set_xticks([1, 2], ['Tàu dầu khác', 'Sẽ bị cấm'], fontsize=8); a.set_title(t, fontsize=9)
a = ax.ravel()[5]; sh = pd.DataFrame(R['rq1_share_nonzero']).T
a.bar(np.arange(2)-.2, sh['Tàu dầu khác'], .4, label='Tàu dầu khác', color='#999'); a.bar(np.arange(2)+.2, sh['Sẽ bị cấm (≤6 tháng)'], .4, label='Sẽ bị cấm', color='#E45756')
a.set_xticks([0, 1], ['Có đứng yên ngoài cảng', 'Có gặp tàu khác ≥30 phút'], fontsize=8); a.set_title('Tỉ lệ tàu có hành vi dừng trong tháng', fontsize=9); a.legend(fontsize=7)
fig.suptitle('Hình mẫu hành vi: tàu sắp bị cấm so với tàu dầu khác (mọi snapshot 04/2025–03/2026)')
fig.savefig(FIG/'13_fingerprint.png'); plt.close(fig)

E = pd.read_parquet(ROOT/'data/app/encounters.parquet')
R['rq1_encounters_oct_mar'] = dict(sessions=len(E), with_already=int(((E.status1 == 'already') | (E.status2 == 'already')).sum()),
                                  with_future=int(((E.status1 == 'future') | (E.status2 == 'future')).sum()))

# ---------- Phụ: tàu đã bị cấm vẫn đi qua ----------
labels = pd.read_parquet(ROOT/'data/labels/vessels_first_listed.parquet')
listed = dict(zip(labels.vessel_id, labels.first_listed.str[:10]))
rows = []
for m in pd.period_range('2025-04', '2026-03', freq='M'):
    ds = [str(x.date()) for x in pd.date_range(m.start_time, m.end_time)]
    vs, _ = month_vessels([ROOT/'data/vessel_day'/f'date={x}.parquet' for x in ds], labels, ais_identity_only=True)
    cut = str(m.end_time.date())
    already = [x for x in vs if x['label_id'] and listed.get(x['label_id'], '9999') <= cut and x['n_valid_pos'] >= 10]
    rows.append(dict(month=str(m), already_present=len(already), already_tankers=sum(x['is_tanker'] for x in already),
                     tankers_total=sum(x['is_tanker'] and x['n_valid_pos'] >= 10 for x in vs)))
S = pd.DataFrame(rows); R['sanctioned_still_transiting'] = S.to_dict('records')
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(S.month, S.already_tankers, color='#E45756'); ax.set_ylabel('số tàu dầu'); ax.tick_params(axis='x', rotation=60)
for x, y, t in zip(S.month, S.already_tankers, S.tankers_total): ax.text(x, y, f'{y/t:.0%}', ha='center', va='bottom', fontsize=7)
ax.set_title('Tàu dầu ĐÃ bị EU/UK cấm vẫn có mặt ở vùng biển Đan Mạch mỗi tháng (% trên tổng tàu dầu)')
fig.savefig(FIG/'16_sanctioned_present.png'); plt.close(fig)

# ---------- RQ3: cảnh báo sớm (chấm hằng ngày) ----------
P = pd.read_parquet(ROOT/'data/predict_all.parquet'); P['date'] = P.date.astype(str)
V = pd.read_parquet(ROOT/'data/app/vessels.parquet').set_index('vessel_id')
pos = P[P.y == 1]; rows = []
for vid, g in pos.groupby('vessel_id'):
    fl = pd.Timestamp(g.first_listed.iloc[0]); w = g[g.rank_watch <= 30]
    rows.append(dict(vessel_id=vid, name=V.name.get(vid, ''), listed=str(fl.date()), present_days=int((g.days_since_seen == 0).sum()),
                     first_top30=w.date.min() if len(w) else None, lead_days=(fl - pd.Timestamp(w.date.min())).days if len(w) else None,
                     best_rank=None if g.rank_watch.isna().all() else int(g.rank_watch.min())))
L = pd.DataFrame(rows).sort_values('lead_days', ascending=False)
seen = L[L.present_days > 0]
R['rq3'] = dict(positives=len(L), positives_present=len(seen), caught_top30=int(L.lead_days.notna().sum()),
                lead_median=float(L.lead_days.median()), lead_ge30=int((L.lead_days >= 30).sum()), lead_ge90=int((L.lead_days >= 90).sum()),
                table=L.head(15).to_dict('records'))
pr = P[P.days_since_seen == 0].copy(); pr['pct'] = pr.groupby('date').score.rank(pct=True)
R['rq2_pct_when_present'] = {'pos': round(float(pr[pr.y == 1].pct.median()), 3), 'neg': round(float(pr[pr.y == 0].pct.median()), 3)}
LL = L.dropna(subset=['lead_days']).sort_values('listed')
fig, ax = plt.subplots(figsize=(8, 0.28*len(LL)+1))
for i, r in enumerate(LL.itertuples()):
    ax.plot([pd.Timestamp(r.first_top30), pd.Timestamp(r.listed)], [i, i], c='#999', lw=1)
    ax.plot(pd.Timestamp(r.first_top30), i, 'o', c='#4C78A8'); ax.plot(pd.Timestamp(r.listed), i, 'X', c='#E45756')
ax.set_yticks(range(len(LL)), [f'{n or v} ({int(l)} ngày)' for n, v, l in zip(LL.name, LL.vessel_id, LL.lead_days)], fontsize=7)
ax.set_title('Cảnh báo sớm: ● lần đầu vào top 30 theo dõi → ✖ ngày bị EU/UK liệt kê', fontsize=9)
fig.savefig(FIG/'15_early_warning.png'); plt.close(fig)

# ---------- RQ4: điểm cao nhưng chưa bị EU/UK cấm ----------
ever = P.groupby('vessel_id').first_listed.max() != ''
for k in (10, 30):
    s = P[P.rank_watch <= k].vessel_id.unique()
    R[f'rq4_top{k}_eventually_listed'] = dict(vessels=len(s), share=round(float(ever[s].mean()), 4))
R['rq4_base_eventually_listed'] = round(float(ever.mean()), 4)
cand = (P[(P.rank_watch <= 10) & (P.first_listed == '')].groupby('vessel_id').size().sort_values(ascending=False).head(15))
R['rq4_candidates'] = [dict(vessel_id=v, name=V.name.get(v, ''), days_in_top10=int(n)) for v, n in cand.items()]
late = P[(P.rank_watch <= 10) & (P.first_listed != '') & (P.y == 0)].drop_duplicates('vessel_id')
R['rq4_flagged_listed_after_6m'] = [dict(name=V.name.get(v, ''), listed=f) for v, f in zip(late.vessel_id, late.first_listed)]
(OUT/'insights.json').write_text(json.dumps(R, indent=1, default=str, ensure_ascii=False))
print(json.dumps({k: R[k] for k in ['rq1_medians', 'rq1_share_nonzero', 'rq1_encounters_oct_mar', 'rq2_pct_when_present', 'rq4_top10_eventually_listed',
                                    'rq4_top30_eventually_listed', 'rq4_base_eventually_listed']}, ensure_ascii=False))
print(json.dumps({k: v for k, v in R['rq3'].items() if k != 'table'}))
print(S.to_string(index=False))
