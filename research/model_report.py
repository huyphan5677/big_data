"""Pipeline train cho báo cáo: chia dữ liệu -> chọn feature -> so thuật toán -> Optuna -> đánh giá test.

Quần thể: tàu dầu (98% dương là tàu dầu). Feature: tĩnh + l1m (l3m/l6m gần như trống trong giai đoạn train).
Train = snapshot 202504-09, test = 202510-03 (không đụng tới cho đến bước cuối).
Mọi lựa chọn (feature, thuật toán, siêu tham số) chỉ dựa trên CV trong train: StratifiedGroupKFold 5 theo tàu.
Đầu ra: report/results.json, report/fig/*.png, report/*.csv. Một job, 2 luồng.
"""
import json, re, time, warnings
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb, xgboost as xgb, optuna, shap
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_model import ROOT, ID_COLS, CAT_COLS, load, boot_ci

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os
FEATSET = os.environ.get('FEATSET', 'all')   # 'all' = tĩnh + hành vi (bản cũ); 'behaviour' = chỉ hành vi + gặp gỡ/đồ thị tháng
OUT = ROOT/('report' if FEATSET == 'all' else f'report/{FEATSET}'); FIG = OUT/'fig'; FIG.mkdir(parents=True, exist_ok=True)
SEED, NJ, NFOLD, NTRIAL = 42, 2, 5, 60
TRAIN_END = '202509'
plt.rcParams.update({'figure.dpi': 130, 'savefig.bbox': 'tight', 'font.size': 9})
R = {}
T0 = time.time()


def log(*a):
    print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)


# ---------------- 1. dữ liệu & chia ----------------
d = load()
d = d[d.is_tanker].reset_index(drop=True)
STATIC = ['mid', 'imo_number', 'width', 'length', 'dims_length_gap', 'has_valid_imo', 'n_valid_pos', 'ship_type']
if FEATSET in ('behaviour', 'final'):
    g = pd.concat([pd.read_parquet(p).assign(snapshot=p.stem.split('=')[1]) for p in sorted((ROOT/'data/features_graph').glob('*.parquet'))])
    g = g[['snapshot', 'vessel_id'] + [c for c in g.columns if c.endswith('_l1m')]]   # bỏ bản cộng dồn (PSI trôi mạnh)
    d = d.merge(g, on=['snapshot', 'vessel_id'], how='left', validate='1:1')
if FEATSET == 'final':   # thêm feature hành trình qua gate (voyage_features.py)
    vf = pd.concat([pd.read_parquet(p).assign(snapshot=p.stem.split('=')[1]) for p in sorted((ROOT/'data/features_voyage').glob('*.parquet'))])
    d = d.merge(vf, on=['snapshot', 'vessel_id'], how='left', validate='1:1')
FEATS0 = [c for c in d.columns if c not in ID_COLS + ['is_tanker'] and not re.search(r'_l[36]m', c)
          and not (FEATSET != 'all' and c in STATIC)]
tr = d[d.snapshot <= TRAIN_END].reset_index(drop=True)
te = d[d.snapshot > TRAIN_END].reset_index(drop=True)
seen_pos = set(tr.loc[tr.y == 1, 'group_id'])
R['split'] = dict(population='tanker', train_snapshots=sorted(tr.snapshot.unique()), test_snapshots=sorted(te.snapshot.unique()),
                  train_n=len(tr), train_pos=int(tr.y.sum()), train_vessels=int(tr.group_id.nunique()),
                  train_pos_vessels=int(tr.loc[tr.y == 1, 'group_id'].nunique()),
                  test_n=len(te), test_pos=int(te.y.sum()), test_pos_vessels=int(te.loc[te.y == 1, 'group_id'].nunique()),
                  test_pos_seen_in_train=int(te[te.y == 1].group_id.isin(seen_pos).sum()),
                  per_snapshot=d.groupby('snapshot').agg(n=('y', 'size'), pos=('y', 'sum')).reset_index().to_dict('records'),
                  candidate_features=len(FEATS0))
log('split', {k: v for k, v in R['split'].items() if k != 'per_snapshot'})
folds = list(StratifiedGroupKFold(NFOLD, shuffle=True, random_state=SEED).split(tr, tr.y, tr.group_id))

ps = d.groupby('snapshot').agg(n=('y', 'size'), pos=('y', 'sum'))
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
col = ['#4C78A8' if s <= TRAIN_END else '#F58518' for s in ps.index]
ax[0].bar(ps.index, ps.n, color=col); ax[0].set_title('Số tàu dầu mỗi snapshot'); ax[0].tick_params(axis='x', rotation=60)
ax[1].bar(ps.index, ps.pos, color=col); ax[1].set_title('Số tàu dương (bị cấm trong 6 tháng sau)'); ax[1].tick_params(axis='x', rotation=60)
for a in ax: a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#4C78A8'), plt.Rectangle((0, 0), 1, 1, color='#F58518')], labels=['train', 'test'])
fig.savefig(FIG/'01_split.png'); plt.close(fig)


# ---------------- mô hình ----------------
def cats(feats): return [c for c in CAT_COLS if c in feats]


def sk_pre(feats, scale):
    num = [c for c in feats if c not in CAT_COLS]
    numpipe = [SimpleImputer(strategy='median', add_indicator=True)] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([('num', make_pipeline(*numpipe), num),
                              ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20, sparse_output=False), cats(feats))])


def sk_X(X, feats):
    X = X[feats].copy()
    for c in cats(feats): X[c] = X[c].astype(str)
    return X


BASE = {
 'XGBoost': dict(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0),
 'CatBoost': dict(iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=5.0),
 'LightGBM': dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=50, subsample=0.8, colsample_bytree=0.5,
                  reg_lambda=5.0),
}


def make(name, feats, p=None, seed=SEED):
    """Trả (fit, predict); fit.model là mô hình bên dưới (dùng cho SHAP)."""
    p = {**BASE.get(name, {}), **(p or {})}
    if name in ('LogReg', 'RandomForest'):
        est = (LogisticRegression(C=p.get('C', 0.1), class_weight='balanced', max_iter=3000) if name == 'LogReg' else
               RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features='sqrt', class_weight='balanced_subsample',
                                      n_jobs=NJ, random_state=seed))
        m = make_pipeline(sk_pre(feats, name == 'LogReg'), est)
        fit, pred = (lambda X, y: m.fit(sk_X(X, feats), y)), (lambda X: m.predict_proba(sk_X(X, feats))[:, 1])
    elif name == 'XGBoost':
        m = xgb.XGBClassifier(**p, enable_categorical=True, tree_method='hist', n_jobs=NJ, random_state=seed)
        fit, pred = (lambda X, y: m.fit(X[feats], y)), (lambda X: m.predict_proba(X[feats])[:, 1])
    elif name == 'CatBoost':
        m = CatBoostClassifier(**p, thread_count=NJ, random_seed=seed, verbose=0, cat_features=cats(feats))
        fit, pred = (lambda X, y: m.fit(cbX(X, feats), y)), (lambda X: m.predict_proba(cbX(X, feats))[:, 1])
    elif name == 'LightGBM':
        m = lgb.LGBMClassifier(**p, subsample_freq=1, n_jobs=NJ, random_state=seed, verbose=-1)
        fit, pred = (lambda X, y: m.fit(X[feats], y, categorical_feature=cats(feats))), (lambda X: m.predict_proba(X[feats])[:, 1])
    else:
        raise ValueError(name)
    fit.model = m
    return fit, pred


def cbX(X, feats): return X[feats].assign(**{c: X[c].astype(str) for c in cats(feats)})


def cv(name, feats, p=None):
    oof = np.full(len(tr), np.nan); aucs, aps = [], []
    for a, b in folds:
        fit, pred = make(name, feats, p)
        fit(tr.iloc[a], tr.y.iloc[a]); s = pred(tr.iloc[b]); oof[b] = s
        aucs.append(roc_auc_score(tr.y.iloc[b], s)); aps.append(average_precision_score(tr.y.iloc[b], s))
    return dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)), pr_auc=float(np.mean(aps)), pr_sd=float(np.std(aps))), oof


# ---------------- 2. chọn feature (chỉ trên train) ----------------
fs = {}
miss = tr[FEATS0].isna().mean()
nun = tr[FEATS0].nunique()
step1 = [c for c in FEATS0 if miss[c] <= 0.95 and nun[c] > 1]
fs['step1_drop_missing_constant'] = sorted(set(FEATS0) - set(step1))
num = [c for c in step1 if c not in CAT_COLS]
uni = {c: abs(roc_auc_score(tr.y, tr[c].fillna(tr[c].median())) - 0.5) for c in num}
rho = tr[num].corr(method='spearman').abs().fillna(0)
drop = set(); pairs = []
for c in sorted(num, key=lambda c: -uni[c]):
    if c in drop: continue
    for o in num:
        if o != c and o not in drop and uni[o] <= uni[c] and rho.loc[c, o] > 0.95:
            drop.add(o); pairs.append(dict(keep=c, drop=o, rho=round(float(rho.loc[c, o]), 3)))
step2 = [c for c in step1 if c not in drop]
fs['step2_spearman_095'] = pairs
imp = pd.Series(0.0, index=step2)
for a, b in folds:
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=50, colsample_bytree=0.5,
                           subsample=0.8, subsample_freq=1, reg_lambda=5, n_jobs=NJ, random_state=SEED, verbose=-1)
    m.fit(tr.iloc[a][step2], tr.y.iloc[a], categorical_feature=cats(step2))
    imp += pd.Series(m.booster_.feature_importance('gain'), index=step2)/NFOLD
imp = imp.sort_values(ascending=False)
curve = []
for k in [5, 10, 15, 20, 25, 30, len(step2)]:
    if k > len(step2): continue
    r, _ = cv('LightGBM', list(imp.index[:k])); curve.append(dict(k=k, **r)); log('top-k', k, round(r['auc'], 4))
curve = pd.DataFrame(curve).drop_duplicates('k')
best = curve.loc[curve.auc.idxmax()]
k_sel = int(curve[curve.auc >= best.auc - best.auc_sd/np.sqrt(NFOLD)].k.min())   # quy tắc 1-SE: ít feature nhất
FEATS = list(imp.index[:k_sel])
fs.update(n_start=len(FEATS0), n_step1=len(step1), n_step2=len(step2), topk_curve=curve.round(4).to_dict('records'),
          k_selected=k_sel, selected=FEATS, importance=imp.round(1).to_dict())
R['feature_selection'] = fs
log('features', len(FEATS0), '->', len(step1), '->', len(step2), '->', k_sel)
fig, ax = plt.subplots(figsize=(5, 3))
ax.errorbar(curve.k, curve.auc, yerr=curve.auc_sd/np.sqrt(NFOLD), marker='o', capsize=3)
ax.axvline(k_sel, ls='--', c='gray'); ax.set_xlabel('Số feature (top-k theo gain)'); ax.set_ylabel('CV ROC-AUC')
ax.set_title(f'Chọn số feature (1-SE) -> k={k_sel}'); fig.savefig(FIG/'02_feature_k.png'); plt.close(fig)


# ---------------- 3. so thuật toán (CV trên train) ----------------
ALGOS = ['LogReg', 'RandomForest', 'XGBoost', 'CatBoost', 'LightGBM']
algo = {}
for a in ALGOS:
    t = time.time(); r, _ = cv(a, FEATS); r['fit_s'] = round(time.time()-t, 1); algo[a] = r; log('algo', a, r)
R['algorithms_cv'] = algo
A = pd.DataFrame(algo).T
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for i, (m, sd, t) in enumerate([('auc', 'auc_sd', 'CV ROC-AUC'), ('pr_auc', 'pr_sd', 'CV PR-AUC')]):
    ax[i].barh(A.index, A[m], xerr=A[sd], color='#4C78A8', capsize=3); ax[i].set_title(t)
    ax[i].set_xlim(max(0, A[m].min()-0.15), min(1, A[m].max()+0.1) if m == 'auc' else None)
fig.savefig(FIG/'03_algorithms.png'); plt.close(fig)


# ---------------- 4. Optuna (tune thuật toán thắng CV, CV trên train) ----------------
TUNE = max(BASE, key=lambda a: algo[a]['auc'])          # chỉ tune các họ boosting; chọn theo CV ROC-AUC


def space(trial):
    f, i = trial.suggest_float, trial.suggest_int
    if TUNE == 'CatBoost':
        return dict(iterations=i('iterations', 200, 800, step=100), learning_rate=f('learning_rate', 0.01, 0.1, log=True),
                    depth=i('depth', 3, 7), l2_leaf_reg=f('l2_leaf_reg', 1, 30, log=True),
                    random_strength=f('random_strength', 0.1, 10, log=True), bagging_temperature=f('bagging_temperature', 0, 1))
    if TUNE == 'XGBoost':
        return dict(n_estimators=i('n_estimators', 100, 800, step=50), learning_rate=f('learning_rate', 0.01, 0.1, log=True),
                    max_depth=i('max_depth', 2, 6), min_child_weight=f('min_child_weight', 1, 20, log=True),
                    subsample=f('subsample', 0.5, 1), colsample_bytree=f('colsample_bytree', 0.3, 1),
                    reg_lambda=f('reg_lambda', 1e-2, 50, log=True))
    return dict(n_estimators=i('n_estimators', 100, 800, step=50), learning_rate=f('learning_rate', 0.01, 0.1, log=True),
                num_leaves=i('num_leaves', 4, 31), min_child_samples=i('min_child_samples', 10, 150),
                colsample_bytree=f('colsample_bytree', 0.3, 1.0), subsample=f('subsample', 0.5, 1.0),
                reg_lambda=f('reg_lambda', 1e-2, 50, log=True), reg_alpha=f('reg_alpha', 1e-3, 10, log=True))


def objective(trial):
    r, _ = cv(TUNE, FEATS, space(trial))
    trial.set_user_attr('pr_auc', r['pr_auc']); trial.set_user_attr('auc_sd', r['auc_sd'])
    return r['auc']


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.enqueue_trial({**BASE[TUNE], **dict(CatBoost=dict(random_strength=1.0, bagging_temperature=1.0),
    XGBoost=dict(min_child_weight=1.0), LightGBM=dict(reg_alpha=1e-3))[TUNE]})   # trial 0 = cấu hình mặc định (tay)
study.optimize(objective, n_trials=NTRIAL)
BEST = study.best_params
tri = study.trials_dataframe()
R['optuna'] = dict(algorithm=TUNE, n_trials=NTRIAL, sampler='TPE', objective='mean CV ROC-AUC (StratifiedGroupKFold 5)',
                   default_auc=float(tri.value.iloc[0]), best_auc=float(study.best_value),
                   best_auc_sd=float(study.best_trial.user_attrs['auc_sd']), best_pr_auc=float(study.best_trial.user_attrs['pr_auc']),
                   best_params=BEST)
tri.to_csv(OUT/'optuna_trials.csv', index=False)
log('optuna', R['optuna'])
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(tri.number, tri.value, '.', alpha=.5); ax[0].plot(tri.number, tri.value.cummax(), c='C3')
ax[0].set_xlabel('Trial'); ax[0].set_ylabel('CV ROC-AUC'); ax[0].set_title('Lịch sử Optuna')
try:
    pi = optuna.importance.get_param_importances(study)
    ax[1].barh(list(pi)[::-1], list(pi.values())[::-1], color='#4C78A8'); ax[1].set_title('Độ quan trọng siêu tham số (fANOVA)')
    R['optuna']['param_importance'] = {k: round(v, 3) for k, v in pi.items()}
except Exception as e:
    ax[1].set_axis_off()
fig.savefig(FIG/'04_optuna.png'); plt.close(fig)


# ---------------- 5. đánh giá test (1 lần) ----------------
def fit_seeds(name, p, seeds=(0, 1, 2, 3, 4)):
    s = np.zeros(len(te))
    for sd in seeds:
        fit, pred = make(name, FEATS, p, seed=sd); fit(tr, tr.y); s += pred(te)/len(seeds)
    return s


def topk_month(df, k):
    g = df.groupby('snapshot')
    hits = g.apply(lambda x: int(x.nlargest(k, 'score').y.sum())); pos = g.y.sum()
    return dict(k=k, hits=int(hits.sum()), pos=int(pos.sum()), precision=float(hits.sum()/(k*len(pos))), recall=float(hits.sum()/pos.sum()))


tests = {}
final = {a: (a, None) for a in ALGOS}
FINAL = f'{TUNE}+Optuna'
final[FINAL] = (TUNE, BEST)
for label, (name, p) in final.items():
    df = te[['snapshot', 'group_id', 'vessel_id', 'y']].copy(); df['score'] = fit_seeds(name, p); df['is_tanker'] = True
    rng = np.random.default_rng(0)
    pur = df[~df.group_id.isin(seen_pos)].reset_index(drop=True)
    tests[label] = dict(auc=roc_auc_score(df.y, df.score), pr_auc=average_precision_score(df.y, df.score), **boot_ci(df, rng),
                        purged_auc=roc_auc_score(pur.y, pur.score), purged_pr_auc=average_precision_score(pur.y, pur.score),
                        top10=topk_month(df, 10), top30=topk_month(df, 30),
                        month_auc=df.groupby('snapshot').apply(lambda g: roc_auc_score(g.y, g.score)).round(3).to_dict())
    if label == FINAL: SC = df
    log('test', label, round(tests[label]['auc'], 4), round(tests[label]['pr_auc'], 4))
R['test'] = tests
SC.to_parquet(OUT/'test_scores.parquet', index=False)

# hình đánh giá
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for label, (name, p) in final.items():
    if label not in ('LogReg', 'RandomForest', TUNE, FINAL): continue
    s = SC.score.values if label == FINAL else fit_seeds(name, p, seeds=(0,))
    f, t_, _ = roc_curve(te.y, s); pr, rc, _ = precision_recall_curve(te.y, s)
    ax[0].plot(f, t_, label=f'{label} ({roc_auc_score(te.y, s):.3f})'); ax[1].plot(rc, pr, label=f'{label} ({average_precision_score(te.y, s):.3f})')
ax[0].plot([0, 1], [0, 1], 'k:', lw=.8); ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR'); ax[0].set_title('ROC trên test'); ax[0].legend(fontsize=7)
ax[1].axhline(te.y.mean(), c='k', ls=':', lw=.8); ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_title('Precision-Recall trên test'); ax[1].legend(fontsize=7)
fig.savefig(FIG/'05_roc_pr.png'); plt.close(fig)

ma = pd.DataFrame({k: v['month_auc'] for k, v in tests.items()})
fig, ax = plt.subplots(figsize=(7, 3))
for c in ma.columns: ax.plot(ma.index, ma[c], marker='o', lw=2.5 if c == FINAL else 1, label=c)
ax.set_ylabel('ROC-AUC'); ax.set_title('ROC-AUC theo tháng test'); ax.legend(fontsize=7, ncol=2); fig.savefig(FIG/'06_month_auc.png'); plt.close(fig)

g = SC.groupby('snapshot')
hk = pd.DataFrame({k: g.apply(lambda x, k=k: x.nlargest(k, 'score').y.sum()) for k in (10, 30, 50)}).assign(pos=g.y.sum())
fig, ax = plt.subplots(figsize=(7, 3)); w = 0.2; x = np.arange(len(hk))
for i, c in enumerate(['pos', 10, 30, 50]):
    ax.bar(x+(i-1.5)*w, hk[c], w, label='tổng dương' if c == 'pos' else f'bắt được trong top {c}')
ax.set_xticks(x, hk.index); ax.set_title('Số tàu dương bắt được trong top-K mỗi tháng'); ax.legend(fontsize=7)
fig.savefig(FIG/'07_topk.png'); plt.close(fig)
R['topk_by_month'] = hk.astype(int).reset_index().to_dict('records')

pt, pp = calibration_curve(SC.y, SC.score, n_bins=8, strategy='quantile')
fig, ax = plt.subplots(figsize=(4, 4)); ax.plot(pp, pt, 'o-'); ax.plot([0, pp.max()], [0, pp.max()], 'k:')
ax.set_xlabel('Xác suất dự đoán'); ax.set_ylabel('Tỉ lệ dương thực tế'); ax.set_title('Hiệu chỉnh xác suất (test)')
fig.savefig(FIG/'08_calibration.png'); plt.close(fig)

# SHAP cho mô hình cuối (seed 0)
fit, _ = make(TUNE, FEATS, BEST, seed=0); fit(tr, tr.y); m = fit.model
if TUNE == 'CatBoost':
    from catboost import Pool
    sv = m.get_feature_importance(Pool(cbX(te, FEATS), cat_features=cats(FEATS)), type='ShapValues')[:, :-1]
elif TUNE == 'LightGBM':
    sv = m.predict(te[FEATS], pred_contrib=True)[:, :-1]
else:
    sv = m.get_booster().predict(xgb.DMatrix(te[FEATS], enable_categorical=True), pred_contribs=True)[:, :-1]
Xs = te[FEATS].assign(**{c: te[c].cat.codes for c in cats(FEATS)})
plt.figure(); shap.summary_plot(sv, Xs, show=False, max_display=15, plot_size=(7, 5))
plt.title(f'SHAP trên test ({FINAL})'); plt.savefig(FIG/'09_shap.png'); plt.close('all')
R['final_model'] = FINAL
R['shap_mean_abs'] = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False).round(4).to_dict()

R['runtime_s'] = round(time.time()-T0, 1)
(OUT/'results.json').write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
log('done')
