"""Tầng CỬA SỔ: cộng bảng ngày (features_day) thành l1m/l3m/l6m cho mỗi snapshot, rồi tính tỉ lệ và _vs_.

Quy ước tên: {agg}_{metric}_l{x}m và {agg}_{metric}_l{x}m_vs_l{y}m.
- Cửa sổ lxm = x tháng lịch kết thúc ở tháng snapshot (≤ cutoff). Thiếu ngày nào -> cả cửa sổ để null.
- _vs_ quy về theo ngày có mặt (sum) hoặc so tỉ lệ (rate): giá trị_lx / giá trị_ly; mẫu 0 -> null.
- Quần thể + nhãn lấy từ roster(snapshot) (giống gate2_features cũ); nhãn chỉ nối SAU khi tính đặc trưng.
- Đối tác gặp gỡ "đã bị cấm" tính theo cutoff của snapshot, không theo ngày gặp.
"""
import argparse
import json
import pandas as pd
import numpy as np
from common import ROOT,now,save_json
from gate2_features import roster
from count_presence import month_vessels
from labels import label_status

DAY=ROOT/'data/features_day'
OUT=ROOT/'data/features'
WINDOWS=[1,3,6]
PAIRS=[(1,3),(1,6),(3,6)]
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
PLACEHOLDERS={'UNKNOWN','UNDEFINED','NONE','N/A','NA','0'}
VS=['gap2h_cnt','gap6h_cnt','cov_gap2h_cnt','cov_gap6h_cnt','encounter','encounter_already','transit_out',
    'jump50','frozen_moving','sog_mismatch','loiter','dest_ru']


def imo_ok(s):
    s=str(s).removeprefix('IMO').strip()
    return len(s)==7 and s.isdigit() and s[0]!='0' and sum(int(s[i])*(7-i) for i in range(6))%10==int(s[6])


def months(snapshot,n):
    end=pd.Period(snapshot,freq='M')
    return [str(p) for p in pd.period_range(end-n+1,end,freq='M')]


def days_of(ms):
    return [str(d.date()) for m in ms for d in pd.date_range(pd.Period(m).start_time,pd.Period(m).end_time)]


def load(dates,table,**kw):
    frames=[]
    for d in dates:
        f=pd.read_parquet(DAY/f'date={d}'/table,**kw);f['date']=d;frames.append(f)
    return pd.concat(frames,ignore_index=True)


def window(snapshot,n,cohort_map,labels_by_id,dates=None,cutoff=None,partners=True):
    """Mặc định: cửa sổ n tháng lịch của snapshot. Chấm hằng ngày: truyền dates (30 ngày trượt) + cutoff (ngày chấm);
    partners=False bỏ phần tra đối tác gặp gỡ (enc_tanker/enc_already = 0; mô hình hành vi không dùng hai cột này)."""
    dates=dates or days_of(months(snapshot,n))
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
    if partners:
        partners,_=month_vessels([ROOT/'data/vessel_day'/f'date={d}.parquet' for d in dates],
                                 pd.read_parquet(ROOT/'data/labels/vessels_first_listed.parquet'),ais_identity_only=True)
        for p in partners:
            listed=labels_by_id.get(p['label_id']) if p['label_id'] else None
            already=bool(listed) and pd.Timestamp(str(listed)[:10])<=cutoff
            for m in p['mmsis']:pinfo[m]=(bool(p['is_tanker']),already)
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


def build(snapshot):
    cohort,mapping,_=roster(snapshot)
    cohort_map=cohort[['vessel_id','mmsis']].assign(MMSI=cohort.mmsis.map(json.loads)).explode('MMSI')[['MMSI','vessel_id']]
    labels=pd.read_parquet(ROOT/'data/labels/vessels_first_listed.parquet')
    labels_by_id=dict(zip(labels.vessel_id,labels.first_listed))
    res=cohort.set_index('vessel_id');used={};kept={}
    for n in WINDOWS:
        r,dates=window(snapshot,n,cohort_map,labels_by_id)
        used[f'l{n}m']=dict(first=dates[0],last=dates[-1],complete=r is not None)
        if r is None:continue
        out,static=r
        if static is not None:res=res.join(static)
        kept[n]=out
        res=res.join(out.add_suffix(f'_l{n}m'))
    for x,y in PAIRS:
        if x in kept and y in kept:
            for m in VS:
                res[f'{m}_l{x}m_vs_l{y}m']=kept[x][m]/kept[y][m].where(kept[y][m]>0)
    res=res.reset_index()
    stage=OUT/f'.staging-{snapshot}';target=OUT/f'snapshot={snapshot}'
    import shutil
    shutil.rmtree(stage,ignore_errors=True);stage.mkdir(parents=True)
    assert res.vessel_id.is_unique and len(res)==len(cohort)
    res.to_parquet(stage/'features.parquet',index=False,compression='zstd')
    save_json(stage/'_provenance.json',dict(at=now(),snapshot=snapshot,windows=used,n_vessels=len(res),
        n_pos=int(res.y.sum()),n_features=len(res.columns)-len(cohort.columns),
        cutoff=str(pd.Period(snapshot,freq='M').end_time.date()),source='data/features_day',
        future_labels_used_in_features=False))
    shutil.rmtree(target,ignore_errors=True);stage.rename(target)
    return dict(snapshot=snapshot,n=len(res),pos=int(res.y.sum()),cols=len(res.columns),windows={k:v['complete'] for k,v in used.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshots',nargs='+',
        default=[str(m) for m in pd.period_range('2025-04','2026-03',freq='M')]);a=p.parse_args()
    for s in a.snapshots:print(json.dumps(build(s.replace('-',''))),flush=True)
