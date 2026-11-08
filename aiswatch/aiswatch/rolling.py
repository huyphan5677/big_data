"""Feature tại ngày chấm D: cửa sổ trượt 30 ngày (hàm window() y hệt lúc train) + hành trình qua gate 180 ngày.

Danh sách tàu được chấm tại D: tàu dầu có mặt trong 30 ngày (D-29..D), có ít nhất 10 bản tin, chưa bị EU/UK cấm tại D.
Đầu ra: data/features_rolling/date=D/features.parquet (kèm nhãn y nếu ngày bị cấm đã biết — chỉ để đối chiếu).
"""
import json
import numpy as np, pandas as pd
from .config import DATA, WINDOW_DAYS, VOYAGE_DAYS, log
from . import identity

ROOT = DATA
DAY = DATA/'data/features_day'
OUT = DATA/'data/features_rolling'
PLACEHOLDERS={'UNKNOWN','UNDEFINED','NONE','N/A','NA','0'}
LOOK = VOYAGE_DAYS
SUMS=['n_msgs','n_rows','n_seg','n_jump50','distance_nm','sog_sum','sog_n','n_heading_null','n_rot_null',
      'n_dest_ru','n_dest_vague','n_dest_empty','n_anchor_moving','n_seg_moving','n_frozen_moving',
      'n_seg_speedcheck','n_sog_mismatch','n_gap2h','gap2h_s','n_cov_gap2h','cov_gap2h_s','n_gap6h','gap6h_s',
      'n_cov_gap6h','cov_gap6h_s','n_greatbelt_out','n_greatbelt_in','n_oresund_out','n_oresund_in',
      'n_skagen_out','n_skagen_in','n_bins','n_bins_offport','n_bins_loiter','n_bins_port']
# (tên, loại, tử, mẫu) — loại 'day': tử / ngày có mặt; 'rate': tử / mẫu.
DERIVED=[('gap2h_cnt','day','n_gap2h',None),('gap6h_cnt','day','n_gap6h',None),
         ('cov_gap2h_cnt','day','n_cov_gap2h',None),('cov_gap6h_cnt','day','n_cov_gap6h',None),
         ('gap2h_frac','rate','gap2h_s','span_s'),('cov_gap6h_frac','rate','cov_gap6h_s','span_s'),
         ('distance_nm','day','distance_nm',None),('encounter','day','enc',None),('encounter_hours','day','enc_hours',None),
         ('encounter_tanker','day','enc_tanker',None),('encounter_already','day','enc_already',None),
         ('transit_out','day','transit_out',None),('transit_in','day','transit_in',None),
         ('jump50','rate','n_jump50','n_seg'),('frozen_moving','rate','n_frozen_moving','n_seg_moving'),
         ('sog_mismatch','rate','n_sog_mismatch','n_seg_speedcheck'),('heading_null','rate','n_heading_null','n_rows'),
         ('rot_null','rate','n_rot_null','n_rows'),('anchor_moving','rate','n_anchor_moving','n_rows'),
         ('dest_ru','rate','n_dest_ru','n_rows'),('dest_vague','rate','n_dest_vague','n_rows'),
         ('dest_empty','rate','n_dest_empty','n_rows'),('loiter','rate','n_bins_loiter','n_bins_offport'),
         ('port','rate','n_bins_port','n_bins'),('mean_sog','rate','sog_sum','sog_n')]


def imo_ok(s):
    s=str(s).removeprefix('IMO').strip()
    return len(s)==7 and s.isdigit() and s[0]!='0' and sum(int(s[i])*(7-i) for i in range(6))%10==int(s[6])


def load(dates,table,**kw):
    frames=[]
    for d in dates:
        f=pd.read_parquet(DAY/f'date={d}'/table,**kw);f['date']=d;frames.append(f)
    return pd.concat(frames,ignore_index=True)


def window(snapshot,n,cohort_map,labels_by_id,dates=None,cutoff=None,partners=False):
    """Mặc định: cửa sổ n tháng lịch của snapshot. Chấm hằng ngày: truyền dates (30 ngày trượt) + cutoff (ngày chấm);
    partners=False bỏ phần tra đối tác gặp gỡ (enc_tanker/enc_already = 0; mô hình hành vi không dùng hai cột này)."""
    if not all((DAY/f'date={d}'/'_meta.json').exists() for d in dates):return None,dates
    cutoff=pd.Timestamp(cutoff) if cutoff is not None else pd.Period(snapshot,freq='M').end_time.normalize()
    assert pd.Timestamp(dates[-1])<=cutoff,'window vượt cutoff'
    mm=set(cohort_map.MMSI)
    v=load(dates,'vessel')
    v=v[v.MMSI.isin(mm)].sort_values(['MMSI','first_t'])
    # Khoảng trống vắt qua ranh giới ngày (nối last_* hôm trước với first_* lần xuất hiện sau).
    g=v.first_t-v.groupby('MMSI').last_t.shift()
    cov=v.first_high&v.groupby('MMSI').last_high.shift().fillna(False).astype(bool)
    for h in [2,6]:
        ev=g>h*3600
        v[f'n_gap{h}h']+=ev.astype(int);v[f'gap{h}h_s']+=g.where(ev,0)
        v[f'n_cov_gap{h}h']+=(ev&cov).astype(int);v[f'cov_gap{h}h_s']+=g.where(ev&cov,0)
    v['max_gap_s']=np.fmax(v.max_gap_s,g)
    v=v.merge(cohort_map,on='MMSI')
    a=v.groupby('vessel_id').agg(**{c:(c,'sum') for c in SUMS},max_gap_s=('max_gap_s','max'),
        draught_min=('draught_min','min'),draught_max=('draught_max','max'),days=('date','nunique'),
        t0=('first_t','min'),t1=('last_t','max'))
    a['span_s']=(a.t1-a.t0).clip(lower=1)
    a['transit_out']=a[['n_greatbelt_out','n_oresund_out','n_skagen_out']].sum(axis=1)
    a['transit_in']=a[['n_greatbelt_in','n_oresund_in','n_skagen_in']].sum(axis=1)

    # Gặp gỡ: phiên >=30 phút; nhãn đối tác theo cutoff snapshot.
    e=load(dates,'encounters')
    e=e[(e.last_bin-e.first_bin)*5>=30]
    pinfo={}
    both=pd.concat([e.rename(columns={'m1':'self','m2':'other'}),e.rename(columns={'m2':'self','m1':'other'})])
    both=both.merge(cohort_map,left_on='self',right_on='MMSI')
    own=cohort_map.set_index('MMSI').vessel_id
    both=both[both.other.map(own)!=both.vessel_id]
    both['hours']=(both.last_bin-both.first_bin)/12
    both['tanker']=both.other.map(lambda m:pinfo.get(m,(False,False))[0])
    both['already']=both.other.map(lambda m:pinfo.get(m,(False,False))[1])
    enc=both.groupby('vessel_id').agg(enc=('other','size'),enc_hours=('hours','sum'),enc_tanker=('tanker','sum'),
        enc_already=('already','sum'),cnt_partners=('other','nunique'))
    a=a.join(enc).fillna({c:0 for c in enc.columns})

    # Giá trị khai báo: danh tính đổi, giá trị chiếm đa số, IMO dùng chung (tính trên MỌI tàu).
    val=load(dates,'values')
    val=val[~val.value.isin(PLACEHOLDERS)]  # 'UNKNOWN'/'UNDEFINED' là giá trị giữ chỗ, không phải danh tính
    imo=val[(val.field=='IMO')]
    imo=imo[imo.value.map(imo_ok)]
    share=imo.groupby('value').MMSI.nunique()
    cv=val[val.MMSI.isin(mm)].merge(cohort_map,on='MMSI')
    cnt=cv.groupby(['vessel_id','field','value'])['count'].sum().reset_index()
    for f,name in [('Name','cnt_names'),('Callsign','cnt_callsigns'),('Destination','cnt_destinations')]:
        a[name]=cnt[cnt.field==f].groupby('vessel_id').size()
    ci=cnt[(cnt.field=='IMO')&cnt.value.map(imo_ok)]
    a['cnt_imos']=ci.groupby('vessel_id').size()
    a['max_shared_imo_mmsis']=ci.assign(s=ci.value.map(share)).groupby('vessel_id').s.max()
    a['cnt_mmsis']=v.groupby('vessel_id').MMSI.nunique()
    out=pd.DataFrame(index=a.index)
    for name,kind,num,den in DERIVED:
        out[name]=a[num]/a['days'] if kind=='day' else a[num]/a[den].where(a[den]>0)
        if kind=='day' and num.endswith('_s'):out[name]/=3600  # giây -> giờ
        if name.endswith('_frac'):out[name]=out[name].clip(upper=1)  # tàu nhiều MMSI: khoảng trống cộng dồn > khoảng quan sát
    out['max_gap_hours']=a.max_gap_s/3600
    out['draught_range']=a.draught_max-a.draught_min
    out['days_present']=a['days']
    for c in ['cnt_names','cnt_callsigns','cnt_destinations','cnt_imos','max_shared_imo_mmsis','cnt_mmsis','cnt_partners']:
        out[c]=a[c]
    static=None
    if n==1:
        dom=(cnt.sort_values(['vessel_id','field','count','value'],ascending=[True,True,False,True])
                .drop_duplicates(['vessel_id','field']).pivot(index='vessel_id',columns='field',values='value'))
        static=pd.DataFrame(index=a.index)
        static['ship_type']=dom.get('Ship type')
        static['length']=pd.to_numeric(dom.get('Length'),errors='coerce')
        static['width']=pd.to_numeric(dom.get('Width'),errors='coerce')
        dims=dom.get('dims').str.split('|',expand=True).apply(pd.to_numeric,errors='coerce') if 'dims' in dom else None
        static['dims_length_gap']=(dims[0]+dims[1]-static.length).abs() if dims is not None else np.nan
        static['has_valid_imo']=a.cnt_imos.fillna(0)>0
        top=ci.sort_values('count',ascending=False).drop_duplicates('vessel_id').set_index('vessel_id').value
        static['imo_number']=pd.to_numeric(top.str.removeprefix('IMO'),errors='coerce')  # proxy tuổi tàu
        top_m=v.groupby(['vessel_id','MMSI']).n_msgs.sum().reset_index().sort_values('n_msgs',ascending=False).drop_duplicates('vessel_id')
        static['mid']=top_m.set_index('vessel_id').MMSI.str[:3]
    return (out,static),dates


COLS = ['MMSI', 'n_rows', 'n_dest_ru', 'draught_max'] + [f'n_{g}_{d}' for g in ('greatbelt', 'oresund', 'skagen') for d in ('out', 'in')]


def med_gap(ts):
    ts = np.sort(ts.unique())
    return np.median(np.diff(ts).astype('timedelta64[D]').astype(float)) if len(ts) >= 2 else np.nan


def stay(g):
    """Trung vị số ngày từ lần đi vào Baltic tới lần đi ra kế tiếp (thời gian ở trong Baltic, gồm bốc hàng)."""
    ins, outs = np.sort(g.loc[g['in'], 't'].unique()), np.sort(g.loc[g.out, 't'].unique())
    r = [(outs[outs > i][0]-i).astype('timedelta64[D]').astype(float) for i in ins if (outs > i).any()]
    return np.median(r) if r else np.nan


def compute(C, cohort, V, N, data_start):
    """Feature hành trình tại cutoff C (Timestamp) cho các vessel_id trong cohort, chỉ dùng dữ liệu (C-LOOK, C]."""
    C = pd.Timestamp(C)
    start = max(pd.Timestamp(data_start), C - pd.Timedelta(days=LOOK-1))
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



def months_for(dates):
    return sorted({d[:4]+d[5:7] for d in dates})


def load_daily(start, end, labels):
    """Bảng tàu-ngày + tên/callsign trong [start, end], gắn vessel_id theo danh tính của tháng tương ứng."""
    days = [str(x.date()) for x in pd.date_range(start, end) if (DAY/f'date={x.date()}'/'_meta.json').exists()]
    mp = pd.concat([identity.month_map(m, labels)[['MMSI', 'vessel_id']].assign(month=m) for m in months_for(days)])
    V, N = [], []
    for d in days:
        V.append(pd.read_parquet(DAY/f'date={d}'/'vessel', columns=COLS).assign(date=d, month=d[:4]+d[5:7]))
        n = pd.read_parquet(DAY/f'date={d}'/'values', filters=[('field', 'in', ['Name', 'Callsign'])], columns=['MMSI', 'field', 'value'])
        N.append(n[~n.value.isin(PLACEHOLDERS)].assign(date=d, month=d[:4]+d[5:7]))
    V = pd.concat(V, ignore_index=True).merge(mp, on=['MMSI', 'month'])
    N = pd.concat(N, ignore_index=True).merge(mp, on=['MMSI', 'month'])
    V['out'] = V[[c for c in COLS if c.endswith('_out')]].sum(axis=1) > 0
    V['in'] = V[[c for c in COLS if c.endswith('_in')]].sum(axis=1) > 0
    V['t'] = pd.to_datetime(V.date); N['t'] = pd.to_datetime(N.date)
    return V, N


def build(day, labels, V, N, data_start):
    """Feature + danh sách tàu của ngày chấm D. Cần đủ 30 ngày tầng ngày tới D."""
    D = pd.Timestamp(day)
    dates = [str(x.date()) for x in pd.date_range(D - pd.Timedelta(days=WINDOW_DAYS-1), D)]
    # Tàu dầu = từng khai là tàu dầu ở bất kỳ tháng nào từ đầu dữ liệu tới D (có tàu đổi "loại tàu" trên AIS giữa chừng).
    hist_months = months_for([str(x.date()) for x in pd.date_range(data_start, D)])
    tanker_ids = set(pd.concat([identity.month_map(m, labels).query('is_tanker').vessel_id for m in hist_months]))
    maps = [identity.month_map(m, labels).assign(month=m) for m in months_for(dates)]
    m = pd.concat(maps)
    m = m[m.vessel_id.isin(tanker_ids)].sort_values('month').drop_duplicates('MMSI', keep='last')   # danh tính tàu dầu mới nhất của MMSI
    tank = m
    cmap = tank[['MMSI', 'vessel_id']]
    listed = dict(zip(labels.vessel_id, labels.first_listed.str[:10]))
    info = tank.drop_duplicates('vessel_id', keep='last').set_index('vessel_id')[['label_id']]
    info['first_listed'] = info.label_id.map(listed).fillna('')
    labels_by_id = dict(zip(labels.vessel_id, labels.first_listed))
    r, _ = window(None, 1, cmap, labels_by_id, dates=dates, cutoff=D)
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
    out['y'] = ((fl > D) & (fl <= D + pd.DateOffset(months=6))).astype(int)   # chỉ để đối chiếu; ở hiện tại phần lớn chưa biết
    out['days_since_seen'] = (D - pd.to_datetime(out.last_seen)).dt.days
    out['date'] = day
    out = out.reset_index().rename(columns={'index': 'vessel_id'})
    vf, _ = compute(D, out.vessel_id, V, N, data_start)
    out = out.merge(vf, on='vessel_id', how='left', validate='1:1')
    assert out.vessel_id.is_unique
    p = OUT/f'date={day}'; p.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p/'.features.parquet', index=False); (p/'.features.parquet').rename(p/'features.parquet')
    return out
