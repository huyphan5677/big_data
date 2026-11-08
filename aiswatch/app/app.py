"""Demo: chấm điểm rủi ro tàu dầu hằng ngày từ hành vi AIS (vùng biển Đan Mạch), phát lại 10/2025–03/2026.

Chạy: PYTHONPATH=.deps_app streamlit run app.py --server.address 127.0.0.1 --server.port 8501
Dữ liệu: data/predict_all.parquet (predict_daily.py + compact_predict.py), data/app/* (app_prep.py), models/final/meta.json, report/final/*.
"""
import json
from pathlib import Path

import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import streamlit as st

import os
PKG = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('AISWATCH_DATA', PKG)).resolve()   # kết quả do pipeline ghi
MODEL = PKG/'model'                                            # mô hình + báo cáo lúc xây dựng (tĩnh)
st.set_page_config(page_title='Giám sát tàu dầu — AIS Đan Mạch', layout='wide')
META = json.load(open(MODEL/'meta.json')); FE = META['features']
LAB = {  # nhãn tiếng Việt + cách hiển thị giá trị
    'distance_nm_l1m': ('Quãng đường mỗi ngày', '{:.0f} hải lý'), 'port_l1m': ('Thời gian ở cảng', '{:.0%}'),
    'max_gap_hours_l1m': ('Mất tín hiệu dài nhất', '{:.1f} giờ'), 'mean_sog_l1m': ('Tốc độ trung bình', '{:.1f} kn'),
    'cnt_destinations_l1m': ('Số điểm đến khai báo', '{:.0f}'), 'dest_vague_l1m': ('Điểm đến mơ hồ', '{:.0%}'),
    'draught_range_l1m': ('Biên độ mớn nước', '{:.1f} m'), 'dest_ru_l1m': ('Khai điểm đến Nga', '{:.0%}'),
    'jump50_l1m': ('Vị trí nhảy >50 hải lý', '{:.1%}'), 'frozen_moving_l1m': ('Đứng im khi đang chạy', '{:.0%}'),
    'v_draught_out_rel': ('Mớn nước khi ra / tối đa', '{:.0%}'), 'v_draught_out_minus_in': ('Mớn nước ra − vào', '{:+.1f} m'),
    'v_baltic_stay_days': ('Số ngày ở trong Baltic', '{:.0f} ngày'), 'v_in_per30': ('Lượt vào Baltic / 30 ngày', '{:.1f}'),
    'v_dest_ru_share': ('Khai điểm đến Nga (180 ngày)', '{:.0%}')}
GATES = [dict(name='Great Belt', path=[[10.85, 55.34], [11.25, 55.34]]), dict(name='Øresund', path=[[12.55, 55.98], [12.80, 55.98]]),
         dict(name='Skagen', path=[[10.65, 57.75], [10.65, 58.60]])]
WATCH = 3
MAP_VIEW = pdk.ViewState(latitude=56.3, longitude=10.8, zoom=5.3)
GATE_LAYER = pdk.Layer('PathLayer', GATES, get_path='path', get_color=[0, 150, 80], width_min_pixels=4, pickable=True)


@st.cache_data
def load_pred():
    p = pd.read_parquet(DATA/'data/predict_all.parquet')
    p['date'] = p.date.astype(str)
    return p


@st.cache_data
def load_app(name):
    return pd.read_parquet(DATA/f'data/app/{name}.parquet')


def fmt(f, v):
    return '—' if pd.isna(v) else LAB[f][1].format(v)


# Cụm từ dễ hiểu cho từng dấu hiệu (None = không đáng nêu làm lý do, ví dụ "điểm đến mơ hồ 0%").
PHRASE = {
    'distance_nm_l1m': lambda v: f'chạy xa {v:.0f} hải lý/ngày',
    'port_l1m': lambda v: 'không ghé cảng' if v < 0.01 else (f'ít ghé cảng ({v:.0%} thời gian)' if v < 0.1 else None),
    'max_gap_hours_l1m': lambda v: f'AIS liên tục (mất tối đa {v:.1f} giờ)' if v < 3 else f'có lúc mất tín hiệu {v:.0f} giờ',
    'mean_sog_l1m': lambda v: f'tốc độ trung bình {v:.1f} kn',
    'dest_vague_l1m': lambda v: f'điểm đến mơ hồ {v:.0%} thời gian' if v > 0.05 else None,
    'v_draught_out_rel': lambda v: f'đi ra đầy hàng ({v:.0%} mớn nước tối đa)' if v >= 0.95 else None,
    'v_draught_out_minus_in': lambda v: f'ra sâu hơn vào {v:.1f} m (ra có hàng, vào rỗng)' if v > 1 else None,
    'v_baltic_stay_days': lambda v: f'ở trong Baltic {v:.0f} ngày mỗi lượt',
    'v_in_per30': lambda v: f'quay vào Baltic {v:.1f} lượt/30 ngày',
    'v_dest_ru_share': lambda v: f'khai điểm đến cảng Nga ({v:.0%} thời gian)' if v > 0 else None}


def reasons(row, n=3):
    out = []
    for v, f in sorted(((row[f'shap_{f}'], f) for f in FE), reverse=True):
        if v <= 0 or len(out) == n: break
        x = row[f]
        t = None if pd.isna(x) else (PHRASE[f](x) if f in PHRASE else f'{LAB[f][0]}: {fmt(f, x)}')
        if t: out.append(t)
    return ' · '.join(out) or '—'


@st.cache_data(show_spinner='Đang dựng đồ thị…')
def layout(rng, only_t):
    import networkx as nx
    E = load_app('encounters')
    e = E[(E.date.str[:7] >= rng[0]) & (E.date.str[:7] <= rng[1])]
    if only_t: e = e[e.tanker1 | e.tanker2]
    G = nx.Graph()
    for a, b, h in e.groupby(['v1', 'v2']).hours.sum().reset_index().itertuples(index=False):
        G.add_edge(a, b, w=h)
    status = {}
    for k in '12':
        for v, s in zip(e[f'v{k}'], e[f'status{k}']):
            status[v] = 'already' if 'already' in (s, status.get(v)) else ('future' if 'future' in (s, status.get(v)) else 'none')
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    keep = set().union(*comps[:40]) | {v for v in G if status.get(v) != 'none'}
    for c in comps:
        if c & keep: keep |= c
    H = G.subgraph(keep).copy()
    return H, status, nx.spring_layout(H, k=0.08, iterations=40, seed=1), G.number_of_nodes()


P = load_pred(); V = load_app('vessels').set_index('vessel_id')
DATES = sorted(P.date.unique())
name = lambda vid: V.name.get(vid) or vid

page = st.sidebar.radio('Trang', ['0. 📖 Câu chuyện (bắt đầu ở đây)', '1. Danh sách hằng ngày', '2. Chi tiết một tàu',
                                  '3. Mạng lưới gặp nhau', '4. Cảnh báo sớm', '5. Đánh giá mô hình'])
st.sidebar.markdown('---')
answer = st.sidebar.toggle('Chế độ đáp án (hiện ngày bị cấm thật)', value=False)
st.sidebar.caption('Trang 1–5 để khám phá chi tiết. Dữ liệu phát lại 10/2025–03/2026, giai đoạn mô hình chưa từng thấy khi học.')

INS = json.load(open(MODEL/'report/insights.json'))
DOVE, HORAE, NORMAL = 'IMO9297541', 'IMO9413004', 'IMO9236315'   # ca minh hoạ của giai đoạn kiểm tra; NORMAL = OLJAREN (tàu thường)
HAS_CASE = '2025-10-24' in DATES and DOVE in set(P.vessel_id)       # dữ liệu đang chạy có chứa ca minh hoạ không
STORY_DAY = '2025-10-24' if HAS_CASE else DATES[-1]
if not HAS_CASE:                                                     # dữ liệu mới: lấy tàu hạng 1 của ngày mới nhất làm ví dụ
    _d = P[P.date == STORY_DAY]; DOVE = _d.nsmallest(1, 'rank_watch').vessel_id.iloc[0]


def chapter(n, title, lead):
    st.markdown(f'### {n}. {title}')
    st.markdown(lead)


# ------------------------------------------------------------------ 0
if page.startswith('0'):
    st.title('Mỗi sáng, nên để mắt tới tàu dầu nào đi qua eo biển Đan Mạch?')
    st.markdown('Một câu chuyện 5 phần, từ dữ liệu AIS của **365 ngày** (khoảng 18 triệu bản tin mỗi ngày) tới một danh sách mỗi sáng '
                'có thể dùng ngay. Kéo xuống từ từ.')
    st.divider()

    # 1. bối cảnh
    chapter(1, 'Một nút cổ chai, rất nhiều tàu dầu',
            'Muốn đưa dầu từ các cảng Baltic của Nga ra thế giới, tàu **bắt buộc** phải đi qua 3 cửa: **Skagen, Great Belt, Øresund** '
            '(đường xanh lá trên bản đồ). Đan Mạch không chặn được tàu, nhưng **kiểm tra** được những tàu đi qua.')
    d0 = P[(P.date == STORY_DAY) & (P.days_since_seen < WATCH)]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pydeck_chart(pdk.Deck(layers=[GATE_LAYER, pdk.Layer('ScatterplotLayer', d0.assign(tau=d0.vessel_id.map(name)),
                        get_position='[last_lon, last_lat]', get_radius=3000, get_fill_color=[90, 90, 90, 140], pickable=True)],
                        initial_view_state=MAP_VIEW, map_provider='carto', map_style='light', tooltip={'text': '{tau}'}), height=380)
        st.caption(f'Các tàu dầu có mặt ở vùng biển trong 3 ngày tính tới {STORY_DAY}: {len(d0)} tàu.')
    with c2:
        S = pd.DataFrame(INS['sanctioned_still_transiting'])
        fig = go.Figure(go.Bar(x=S.month, y=S.already_tankers, marker_color='crimson',
                               text=[f'{a/t:.0%}' for a, t in zip(S.already_tankers, S.tankers_total)], textposition='outside'))
        fig.update_layout(height=360, margin=dict(t=40, b=10), title='Tàu dầu ĐÃ bị EU/UK cấm vẫn đi qua, mỗi tháng', yaxis_title='số tàu')
        st.plotly_chart(fig, width='stretch')
    st.info('**Điều cần thấy:** mỗi tháng khoảng 900 tàu dầu đi qua, và tàu đã bị cấm vẫn đi qua ngày càng nhiều '
            '(từ 26 lên 124 tàu/tháng). Không thể kiểm tra hết, nên cần biết **ưu tiên tàu nào**.')
    st.divider()

    # 2. con thoi
    chapter(2, 'Shadow fleet không lén lút, mà chạy như con thoi',
            'Ta tưởng tàu né cấm vận sẽ tắt AIS, chuyển dầu giữa biển. Ở Đan Mạch thì **không**: cả 6 tháng chỉ có '
            f"**{INS['rq1_encounters_oct_mar']['with_already']} trên {INS['rq1_encounters_oct_mar']['sessions']:,}** lần gặp nhau có mặt tàu đã bị cấm. "
            'Dấu vết thật nằm ở **mớn nước** (độ sâu thân tàu chìm dưới nước, tức tàu chở nặng hay nhẹ) mỗi lần đi qua cửa.')
    G = load_app('crossings')
    if not HAS_CASE:
        st.image(str(MODEL/'report/fig/13_fingerprint.png'), caption='Hình mẫu đo trên dữ liệu xây dựng mô hình (04/2025–03/2026).')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[f'{name(DOVE)}: bị EU/UK cấm 24/04/2026', f'{name(NORMAL)}: tàu dầu bình thường'])
    for col, vid in [(1, DOVE), (2, NORMAL)]:
        g = G[G.vessel_id == vid]
        for kind, sym, colr, lab in [('out', 'triangle-up', 'crimson', 'đi RA Biển Bắc'), ('in', 'triangle-down', 'royalblue', 'đi VÀO Baltic')]:
            x = g[g[kind] > 0]
            fig.add_trace(go.Scatter(x=x.date, y=x.draught_max, mode='markers', marker=dict(symbol=sym, size=13, color=colr), name=lab,
                                     showlegend=col == 1), 1, col)
    fig.update_yaxes(title_text='mớn nước khai báo (m)', row=1, col=1); fig.update_layout(height=360, margin=dict(t=50, b=10))
    if HAS_CASE: st.plotly_chart(fig, width='stretch')
    md = INS['rq1_medians']; k = 'Sẽ bị cấm (≤6 tháng)'; o = 'Tàu dầu khác'
    m = st.columns(3)
    m[0].metric('Mớn nước khi ra − khi vào', f"{md['v_draught_out_minus_in'][k]:+.1f} m", f"tàu dầu khác {md['v_draught_out_minus_in'][o]:+.1f} m", delta_color='off')
    m[1].metric('Số ngày ở trong Baltic mỗi lượt', f"{md['v_baltic_stay_days'][k]:.0f} ngày", f"tàu dầu khác {md['v_baltic_stay_days'][o]:.1f} ngày", delta_color='off')
    m[2].metric('Thời gian ghé cảng', f"{md['port_l1m'][k]:.0%}", f"tàu dầu khác {md['port_l1m'][o]:.1%}", delta_color='off')
    if HAS_CASE: st.info(f'**Điều cần thấy:** {name(DOVE)} đi ra ở mớn nước khoảng 15 m (đầy dầu), đi vào ở khoảng 7,5 m (rỗng), cứ khoảng 2 tháng một vòng. '
            f'{name(NORMAL)} thì mớn nước gần như không đổi. Đây là **hình mẫu chung** của tàu sắp bị cấm (các số trung vị ở trên).')
    st.divider()

    # 3. mô hình
    chapter(3, 'Dạy máy nhận ra con thoi, rồi chấm điểm mỗi sáng',
            'Mô hình (XGBoost) học từ 6 tháng đầu (04–09/2025): những tàu nào **sau đó** bị EU/UK cấm thì đã hành xử ra sao. '
            'Nó **chỉ nhìn hành vi** (không nhìn tên, cờ hay tuổi tàu), gồm 10 dấu hiệu, mạnh nhất là:')
    st.markdown('- đi ra **đầy hàng**, đi vào **rỗng**\n- ở trong Baltic **lâu** mỗi lượt (khoảng 2 tuần, đủ để bốc dầu)\n'
                '- chạy **xa, liền một mạch**, **không ghé cảng** châu Âu\n- khai báo **điểm đến là cảng Nga**')
    st.markdown('Mỗi sáng, mọi tàu dầu vừa đi qua nhận một **điểm rủi ro**. Xếp từ cao xuống thấp được **danh sách theo dõi**.')
    h = P[P.vessel_id == DOVE].sort_values('date')
    tot = P.groupby('date').size()
    pct = 100*(1 - (h.rank_all-1)/h.date.map(tot))
    fig = go.Figure(go.Scatter(x=h.date, y=pct, mode='lines+markers', marker=dict(size=np.where(h.days_since_seen == 0, 9, 3)), line_color='crimson', name=name(DOVE)))
    fig.add_hline(y=50, line_dash='dot', line_color='gray', annotation_text='tàu dầu điển hình (giữa bảng)')
    fig.update_layout(height=300, margin=dict(t=30, b=10), yaxis_title='% tàu có điểm thấp hơn', yaxis_range=[0, 102],
                      title=f'{name(DOVE)} {"luôn nằm gần đỉnh" if HAS_CASE else "(hạng 1 ngày " + STORY_DAY + ")"} bảng xếp hạng (chấm to = ngày tàu có mặt)')
    st.plotly_chart(fig, width='stretch')
    st.divider()

    # 4. một buổi sáng
    chapter(4, f'Sáng ngày {STORY_DAY}: danh sách theo dõi nói gì?',
            f'Hôm đó có {len(d0)} tàu dầu vừa đi qua. Đây là 10 tàu đứng đầu và **vì sao** chúng bị xếp cao:')
    top = d0.nsmallest(10, 'rank_watch').copy()
    reveal = HAS_CASE and st.toggle('⏩ Tua nhanh 6 tháng: tàu nào sau đó bị EU/UK cấm?', value=False, key='reveal')
    top['Hạng'] = top.rank_watch.astype(int); top['Tàu'] = top.vessel_id.map(name); top['Vì sao điểm cao'] = top.apply(reasons, axis=1)
    cols = ['Hạng', 'Tàu', 'Vì sao điểm cao']
    if reveal:
        top['6 tháng sau'] = ['🚨 bị cấm ' + f if y else ('bị cấm muộn hơn: ' + f if f else '') for y, f in zip(top.y, top.first_listed)]
        cols.append('6 tháng sau')
    st.dataframe(top[cols], hide_index=True, width='stretch')
    if reveal:
        npos = int(d0.y.sum())
        st.success(f'Trong {len(d0)} tàu hôm đó chỉ có **{npos} tàu** sẽ bị cấm trong 6 tháng tới, và **cả {npos} đều nằm trong top 10**: '
                   f'**{name(DOVE)}** (hạng 2) và **{name(HORAE)}** (hạng 5), bị EU/UK liệt kê ngày 24/04/2026, tức **6 tháng sau**. '
                   f'Chọn ngẫu nhiên 10 tàu thì kỳ vọng chỉ được {10*npos/len(d0):.2f} tàu.')
    st.divider()

    # 5. không phải may mắn
    chapter(5, 'Không phải may mắn: kết quả kiểm tra lúc xây dựng mô hình',
            'Lặp lại "mỗi sáng một danh sách" cho 182 ngày (10/2025–03/2026), giai đoạn mô hình **chưa từng thấy** khi học:')
    r3 = INS['rq3']
    m = st.columns(4)
    m[0].metric('Tàu bị cấm trong giai đoạn và có đi qua', r3['positives_present'])
    m[1].metric('Đã lọt top 30 TRƯỚC ngày bị cấm', r3['caught_top30'])
    m[2].metric('Sớm hơn (trung vị)', f"{r3['lead_median']:.0f} ngày")
    m[3].metric('Tàu ở top 10 về sau bị cấm', f"{INS['rq4_top10_eventually_listed']['share']:.0%}", f"gấp {INS['rq4_top10_eventually_listed']['share']/INS['rq4_base_eventually_listed']:.0f} lần tàu thường", delta_color='off')
    L = pd.DataFrame(r3['table']).dropna(subset=['lead_days']).sort_values('lead_days')
    fig = go.Figure()
    for r in L.itertuples():
        fig.add_trace(go.Scatter(x=[r.first_top30, r.listed], y=[r.name]*2, mode='lines', line=dict(color='#bbb'), showlegend=False))
    fig.add_trace(go.Scatter(x=L.first_top30, y=L.name, mode='markers', marker=dict(size=11, color='royalblue'), name='lần đầu vào top 30'))
    fig.add_trace(go.Scatter(x=L.listed, y=L.name, mode='markers', marker=dict(size=12, color='crimson', symbol='x'), name='bị EU/UK liệt kê'))
    fig.update_layout(height=420, margin=dict(t=30, b=10), title='15 tàu được nhận ra sớm nhất', legend=dict(orientation='h'))
    st.plotly_chart(fig, width='stretch')
    st.info('**Điều cần thấy:** phần lớn tàu bị cấm đã nằm trong danh sách theo dõi **vài tháng trước** khi EU/UK ra lệnh, '
            'tức là đủ thời gian để lên kế hoạch kiểm tra khi tàu đi qua lần sau.')
    st.divider()
    st.markdown('### Dùng thế nào, và giới hạn')
    st.markdown('\n'.join([
        '- **Mỗi sáng:** danh sách top 30 tàu dầu đáng chú ý, kèm lý do (trang 1).',
        '- **Theo phút** (bước tiếp theo): báo khi một tàu trong danh sách **sắp vào cửa** Skagen / Great Belt / Øresund, đúng lúc có thể kiểm tra.',
        '- **Không phải kết luận vi phạm:** nhãn là lệnh cấm EU/UK, không phủ hết shadow fleet; điểm cao chỉ là **ưu tiên để con người xem xét**.',
        '- **Còn sót:** danh sách cũng chứa nhiều tàu không bị cấm. Mô hình giúp **thu hẹp** từ khoảng 270 xuống 30 tàu mỗi ngày, không thay được kiểm tra thật.',
    ]))
    st.caption('Trang 1–5 ở thanh bên để tự khám phá: chọn ngày, chọn tàu, xem mạng lưới gặp nhau và các chỉ số đánh giá.')

# ------------------------------------------------------------------ 1
elif page.startswith('1'):
    st.title('Danh sách theo dõi hằng ngày')
    st.caption('Cách đọc: chọn một ngày → bảng là các tàu dầu có điểm rủi ro cao nhất trong số tàu vừa đi qua (3 ngày gần nhất), '
               'cột cuối giải thích vì sao. Bật "Chế độ đáp án" ở thanh bên để biết tàu nào sau đó bị cấm thật.')
    c1, c2, c3 = st.columns([3, 1, 1])
    day = c1.select_slider('Ngày chấm', DATES, value='2026-01-15' if '2026-01-15' in DATES else DATES[-1])
    k = c2.number_input('Top K', 5, 100, 30, 5)
    only_watch = c3.toggle(f'Chỉ tàu thấy trong {WATCH} ngày', value=True)
    d = P[P.date == day].copy()
    pool = d[d.days_since_seen < WATCH] if only_watch else d
    top = pool.nsmallest(k, 'rank_watch' if only_watch else 'rank_all').copy()
    m = st.columns(4)
    m[0].metric('Tàu dầu được chấm', len(d)); m[1].metric(f'Đang có mặt (≤{WATCH} ngày)', int((d.days_since_seen < WATCH).sum()))
    if answer:
        m[2].metric('Trong danh sách sẽ bị cấm ≤6 tháng', int(top.y.sum()))
        m[3].metric('Tổng số tàu sẽ bị cấm (đang có mặt)', int(pool.y.sum()))
    top['Hạng'] = range(1, len(top)+1)
    top['Tàu'] = top.vessel_id.map(name)
    top['Điểm'] = top.score.round(3)
    top['Thấy lần cuối'] = top.days_since_seen.map(lambda x: 'hôm nay' if x == 0 else f'{x} ngày trước')
    top['Lý do chính (SHAP)'] = top.apply(reasons, axis=1)
    cols = ['Hạng', 'Tàu', 'vessel_id', 'Điểm', 'Thấy lần cuối', 'Lý do chính (SHAP)']
    if answer:
        top['Bị cấm'] = [f'✅ {f} (sau {(pd.Timestamp(f)-pd.Timestamp(day)).days} ngày)' if y else ('bị cấm sau hơn 6 tháng: '+f if f else '')
                         for f, y in zip(top.first_listed, top.y)]
        cols.append('Bị cấm')
    st.dataframe(top[cols], hide_index=True, width='stretch', height=min(38*len(top)+40, 700))
    st.subheader('Vị trí cuối cùng')
    bg = pool.assign(tau=pool.vessel_id.map(name), diem=pool.score.round(3))
    tp = top.assign(tau=top['Tàu'], diem=top['Điểm'], r=(80+175*top.score.rank(pct=True)).astype(int))
    layers = [GATE_LAYER, pdk.Layer('ScatterplotLayer', bg, get_position='[last_lon, last_lat]', get_radius=2500, get_fill_color=[150, 150, 150, 90], pickable=True),
              pdk.Layer('ScatterplotLayer', tp, get_position='[last_lon, last_lat]', get_radius=6000, get_fill_color='[r, 30, 30, 220]', pickable=True)]
    if answer:
        layers.append(pdk.Layer('ScatterplotLayer', pool[pool.y == 1], get_position='[last_lon, last_lat]', get_radius=11000, stroked=True,
                                filled=False, get_line_color=[255, 140, 0], line_width_min_pixels=2))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=MAP_VIEW, map_provider='carto', map_style='light',
                             tooltip={'text': '{tau}\nđiểm {diem}'}), height=520)
    st.caption('Đỏ: danh sách theo dõi (đậm = điểm cao). Xám: tàu dầu khác. Xanh lá: 3 gate. Vòng cam (chế độ đáp án): tàu sẽ bị cấm trong 6 tháng.')

# ------------------------------------------------------------------ 2
elif page.startswith('2'):
    st.title('Chi tiết một tàu')
    st.caption('Cách đọc: biểu đồ đầu là điểm rủi ro theo ngày; các ô bên dưới so hành vi của tàu (đỏ) với dải 25–75% của tàu dầu khác (xám).')
    best = P[P.rank_watch.notna()].groupby('vessel_id').rank_watch.min().sort_values()
    opts = list(best.index) + [v for v in P.vessel_id.unique() if v not in best.index]
    vid = st.selectbox('Chọn tàu (xếp theo hạng tốt nhất từng đạt)', opts,
                       format_func=lambda v: f'{name(v)} ({v}) — hạng tốt nhất {int(best[v]) if v in best else "-"}')
    h = P[P.vessel_id == vid].sort_values('date')
    fl = V.first_listed.get(vid, '')
    a, b, c = st.columns(3)
    a.metric('Số ngày được chấm', len(h)); b.metric('Hạng tốt nhất (danh sách theo dõi)', int(best[vid]) if vid in best else '-')
    c.metric('Điểm cao nhất', f'{h.score.max():.3f}')
    if answer:
        st.info(f'Bị EU/UK liệt kê lần đầu: **{fl}**' if fl else 'Không có trong danh sách EU/UK (tới 10/2026).')
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=h.date, y=h.score, name='Điểm rủi ro', mode='lines+markers', marker=dict(
        size=np.where(h.days_since_seen == 0, 8, 3))), secondary_y=False)
    fig.add_trace(go.Scatter(x=h.date, y=h.rank_all, name='Hạng (toàn bộ)', line=dict(dash='dot', color='gray')), secondary_y=True)
    fig.update_yaxes(title_text='Điểm', secondary_y=False); fig.update_yaxes(title_text='Hạng', autorange='reversed', secondary_y=True)
    if answer and fl and DATES[0] <= fl <= DATES[-1]:
        fig.add_vline(x=fl, line_color='red', line_dash='dash'); fig.add_annotation(x=fl, y=1, yref='paper', text='bị cấm', showarrow=False, font_color='red')
    fig.update_layout(height=320, margin=dict(t=30, b=10), title='Điểm rủi ro theo ngày (chấm to = ngày tàu có mặt)')
    st.plotly_chart(fig, width='stretch')

    st.subheader('Hành vi (cửa sổ 30 ngày) so với các tàu dầu khác')
    show = FE[:6]
    med = P[P.date.isin(h.date)].groupby('date')[show].quantile([.25, .5, .75]).unstack()
    fig = make_subplots(rows=2, cols=3, subplot_titles=[LAB[f][0] for f in show])
    for i, f in enumerate(show):
        r, cc = i//3+1, i % 3+1
        fig.add_trace(go.Scatter(x=med.index, y=med[(f, .75)], line=dict(width=0), showlegend=False), r, cc)
        fig.add_trace(go.Scatter(x=med.index, y=med[(f, .25)], fill='tonexty', fillcolor='rgba(150,150,150,.25)', line=dict(width=0),
                                 name='25–75% tàu dầu', showlegend=i == 0), r, cc)
        fig.add_trace(go.Scatter(x=h.date, y=h[f], line=dict(color='crimson'), name='Tàu này', showlegend=i == 0), r, cc)
    fig.update_layout(height=480, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width='stretch')

    left, right = st.columns([3, 2])
    pos = load_app('daily_pos'); pos = pos[pos.vessel_id == vid].sort_values('first_t')
    with left:
        st.subheader('Vị trí theo ngày')
        if len(pos):
            pts = pd.concat([pos[['date', 'first_t', 'first_lat', 'first_lon']].set_axis(['date', 't', 'lat', 'lon'], axis=1),
                             pos[['date', 'last_t', 'last_lat', 'last_lon']].set_axis(['date', 't', 'lat', 'lon'], axis=1)]).sort_values('t')
            path = [{'path': pts[['lon', 'lat']].values.tolist()}]
            st.pydeck_chart(pdk.Deck(layers=[
                GATE_LAYER, pdk.Layer('PathLayer', path, get_path='path', get_color=[200, 30, 30], width_min_pixels=2),
                pdk.Layer('ScatterplotLayer', pts, get_position='[lon, lat]', get_radius=3000, get_fill_color=[30, 30, 200, 180], pickable=True)],
                initial_view_state=MAP_VIEW, map_provider='carto', map_style='light', tooltip={'text': '{date}'}), height=430)
            st.caption('Điểm đầu và cuối của mỗi ngày có mặt, nối theo thời gian (quỹ đạo chi tiết sẽ bổ sung).')
    with right:
        st.subheader('Lý do điểm (SHAP)')
        dsel = st.select_slider('Ngày', list(h.date), value=h.loc[h.score.idxmax(), 'date'])
        row = h[h.date == dsel].iloc[0]
        sv = pd.Series({LAB[f][0]+f' ({fmt(f, row[f])})': row[f'shap_{f}'] for f in FE}).sort_values()
        fig = go.Figure(go.Bar(x=sv.values, y=sv.index, orientation='h', marker_color=np.where(sv.values > 0, 'crimson', 'steelblue')))
        fig.update_layout(height=380, margin=dict(t=10, b=10), xaxis_title='→ tăng rủi ro')
        st.plotly_chart(fig, width='stretch')
    E = load_app('encounters'); e = E[(E.v1 == vid) | (E.v2 == vid)]
    st.subheader(f'Các lần gặp tàu khác ≥30 phút: {len(e)}')
    if len(e):
        e = e.assign(doi_tac=np.where(e.v1 == vid, e.v2, e.v1)).assign(ten=lambda x: x.doi_tac.map(name))
        st.dataframe(e[['date', 'ten', 'doi_tac', 'hours', 'lat', 'lon']].round(2), hide_index=True, width='stretch')

# ------------------------------------------------------------------ 3
elif page.startswith('3'):
    st.title('Mạng lưới gặp nhau')
    st.caption('Cách đọc: mỗi chấm là một tàu, mỗi đường là một lần hai tàu đứng sát nhau ≥30 phút ngoài cảng. Tàu bị cấm (đỏ/cam) gần như không nằm trong mạng: chúng không dừng lại ở Đan Mạch.')
    st.caption('Phiên gặp: hai tàu cách nhau <500 m, tốc độ <3 kn, ngoài cảng, kéo dài ≥30 phút.')
    E = load_app('encounters')
    months = sorted(E.date.str[:7].unique())
    rng = st.select_slider('Khoảng tháng', months, value=(months[0], months[-1]))
    e = E[(E.date.str[:7] >= rng[0]) & (E.date.str[:7] <= rng[1])]
    only_t = st.toggle('Chỉ các phiên có ít nhất một tàu dầu', value=True)
    if only_t: e = e[e.tanker1 | e.tanker2]
    sanc = (e.status1 == 'already') | (e.status2 == 'already'); fut = (e.status1 == 'future') | (e.status2 == 'future')
    m = st.columns(4)
    m[0].metric('Số phiên gặp', f'{len(e):,}'); m[1].metric('Số tàu', f'{pd.unique(pd.concat([e.v1, e.v2])).size:,}')
    m[2].metric('Phiên có tàu ĐÃ bị cấm', int(sanc.sum())); m[3].metric('Phiên có tàu SẼ bị cấm', int(fut.sum()))
    st.warning('Tàu bị cấm gần như không gặp ai ở vùng Đan Mạch: shadow fleet đi xuyên eo biển, không dừng, không chuyển tải.')
    col = {'already': [220, 30, 30], 'future': [255, 140, 0], 'none': [70, 110, 200]}
    pts = e.assign(st=np.where(sanc, 'already', np.where(fut, 'future', 'none')))
    pts['color'] = pts.st.map(col); pts['size'] = np.where(pts.st == 'none', 1500, 9000)
    pts['a'] = pts.v1.map(name); pts['b'] = pts.v2.map(name)
    st.pydeck_chart(pdk.Deck(layers=[
        pdk.Layer('HexagonLayer', pts, get_position='[lon, lat]', radius=4000, elevation_scale=40, extruded=False, opacity=.35, coverage=.9),
        pdk.Layer('ScatterplotLayer', pts[pts.st != 'none'], get_position='[lon, lat]', get_radius='size', get_fill_color='color', pickable=True)],
        initial_view_state=MAP_VIEW, map_provider='carto', map_style='light', tooltip={'text': '{a} – {b}\n{date}, {hours} giờ'}), height=480)
    st.caption('Lớp lục giác: mật độ nơi gặp nhau (chủ yếu vùng neo Skagen, Kattegat, ngoài cảng). Chấm đỏ/cam: phiên có tàu đã/sẽ bị cấm.')

    st.subheader('Mạng lưới')
    H, status, pos, n_all = layout(rng, only_t)
    st.caption(f'Hiển thị {H.number_of_nodes():,}/{n_all:,} tàu (40 cụm lớn nhất + mọi cụm có tàu bị cấm).')
    ex, ey = [], []
    for a, b in H.edges():
        ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode='lines', line=dict(width=.4, color='#bbb'), hoverinfo='skip', showlegend=False))
    for s, lab, c, sz in [('none', 'Tàu khác', 'rgba(70,110,200,.6)', 5), ('future', 'Sẽ bị cấm', 'darkorange', 13), ('already', 'Đã bị cấm', 'red', 13)]:
        ns = [v for v in H if status.get(v) == s]
        fig.add_trace(go.Scatter(x=[pos[v][0] for v in ns], y=[pos[v][1] for v in ns], mode='markers', name=f'{lab} ({len(ns)})',
                                 marker=dict(size=sz, color=c), text=[f'{name(v)} ({v}), gặp {H.degree(v)} tàu' for v in ns], hoverinfo='text'))
    fig.update_layout(height=650, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')


# ------------------------------------------------------------------ 4
elif page.startswith('4'):
    st.title('Cảnh báo sớm: tàu lọt danh sách theo dõi bao lâu trước khi bị cấm?')
    r3 = INS['rq3']
    m = st.columns(4)
    m[0].metric('Tàu bị cấm trong giai đoạn', r3['positives']); m[1].metric('Có đi qua vùng biển', r3['positives_present'])
    m[2].metric('Từng lọt top 30 trước khi bị cấm', r3['caught_top30']); m[3].metric('Sớm hơn (trung vị)', f"{r3['lead_median']:.0f} ngày")
    st.image(str(MODEL/'report/fig/15_early_warning.png'))
    t = pd.DataFrame(r3['table']).rename(columns={'name': 'Tàu', 'listed': 'Bị liệt kê', 'first_top30': 'Lần đầu vào top 30',
                                                   'lead_days': 'Sớm hơn (ngày)', 'best_rank': 'Hạng tốt nhất', 'present_days': 'Số ngày có mặt'})
    st.dataframe(t.drop(columns='vessel_id'), hide_index=True, width='stretch')
    st.subheader('Tàu điểm cao chưa bị EU/UK cấm (ứng viên để đối chiếu OFAC/GUR)')
    c1, c2 = st.columns(2)
    c1.metric('Top 10 về sau bị EU/UK cấm', f"{INS['rq4_top10_eventually_listed']['share']:.1%}", f"tỉ lệ chung {INS['rq4_base_eventually_listed']:.1%}", delta_color='off')
    c1.dataframe(pd.DataFrame(INS['rq4_flagged_listed_after_6m']).rename(columns={'name': 'Bị cấm muộn hơn 6 tháng', 'listed': 'Ngày'}), hide_index=True)
    c2.dataframe(pd.DataFrame(INS['rq4_candidates']).rename(columns={'name': 'Tàu', 'days_in_top10': 'Số ngày ở top 10'}), hide_index=True, width='stretch')

# ------------------------------------------------------------------ 5
else:
    from sklearn.metrics import roc_auc_score
    st.title('Đánh giá mô hình (giai đoạn kiểm tra 10/2025–03/2026)')
    r = json.load(open(MODEL/'report/results.json'))
    t = pd.DataFrame(r['test']).T
    t = pd.DataFrame({'ROC-AUC': t.auc.astype(float).round(3), 'PR-AUC': t.pr_auc.astype(float).round(3),
                      'Bắt được trong top 30/tháng': t.top30.map(lambda x: f"{x['hits']}/{x['pos']}")})
    st.subheader('Snapshot theo tháng (holdout)'); st.dataframe(t, width='stretch')
    st.subheader('Chấm hằng ngày (cửa sổ trượt 30 ngày)')
    g = P.groupby('date').apply(lambda x: roc_auc_score(x.y, x.score) if 0 < x.y.sum() < len(x) else np.nan).rename('auc')
    fig = go.Figure(go.Scatter(x=g.index, y=g.values, mode='lines')); fig.update_layout(height=280, yaxis_title='ROC-AUC theo ngày', margin=dict(t=10))
    st.plotly_chart(fig, width='stretch')
    c1, c2 = st.columns(2)
    for i, f in enumerate(['05_roc_pr', '06_month_auc', '07_topk', '09_shap']):
        (c1 if i % 2 == 0 else c2).image(str(MODEL/f'report/fig/{f}.png'))
