"""Train mô hình chốt (thuật toán + tham số Optuna + feature đã chọn trong report/final/results.json) và lưu artefact.
--until: snapshot cuối dùng để train. Backtest dùng 202509 (khớp báo cáo); khi deploy dùng snapshot mới nhất có nhãn chín.
Đầu ra: models/final/{model_seed*.<ext>, meta.json}."""
import argparse, json
from datetime import datetime, timezone
from train_model import ROOT, load

# Tham số mặc định từng họ mô hình, phải giống BASE trong model_report.py (Optuna chỉ ghi đè phần đã tune).
BASE = {'XGBoost': dict(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0),
        'CatBoost': dict(iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=5.0),
        'LightGBM': dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=50, subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0)}
EXT = {'XGBoost': 'json', 'CatBoost': 'cbm', 'LightGBM': 'txt'}


def make(algo, params, seed):
    p = {**BASE[algo], **params}
    if algo == 'XGBoost':
        import xgboost as xgb
        return xgb.XGBClassifier(**p, tree_method='hist', n_jobs=2, random_state=seed)
    if algo == 'CatBoost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=2)
    import lightgbm as lgb
    return lgb.LGBMClassifier(**p, subsample_freq=1, n_jobs=2, random_state=seed, verbose=-1)


def save(m, algo, path):
    m.booster_.save_model(str(path)) if algo == 'LightGBM' else m.save_model(str(path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--until', default='202509'); a = ap.parse_args()
    res = json.load(open(ROOT/'report/final/results.json'))
    FE, ALGO, PARAMS = res['feature_selection']['selected'], res['optuna']['algorithm'], res['optuna']['best_params']
    import pandas as pd
    d = load(); d = d[d.is_tanker & (d.snapshot <= a.until)]
    for sub, pat in [('features_graph', '{}'), ('features_voyage', '{}')]:
        x = pd.concat([pd.read_parquet(p).assign(snapshot=p.stem.split('=')[1]) for p in sorted((ROOT/f'data/{sub}').glob('*.parquet'))])
        d = d.merge(x[['snapshot', 'vessel_id'] + [c for c in x.columns if c in FE]], on=['snapshot', 'vessel_id'], how='left', validate='1:1')
    out = ROOT/'models/final'; out.mkdir(parents=True, exist_ok=True)
    for seed in range(5):
        save(make(ALGO, PARAMS, seed).fit(d[FE], d.y), ALGO, out/f'model_seed{seed}.{EXT[ALGO]}')
    meta = dict(at=datetime.now(timezone.utc).isoformat(timespec='seconds'), algorithm=ALGO, features=FE, params={**BASE[ALGO], **PARAMS},
                seeds=list(range(5)), ext=EXT[ALGO], train_snapshots=sorted(d.snapshot.unique()), train_n=len(d), train_pos=int(d.y.sum()),
                population='tanker', source='report/final/results.json')
    (out/'meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ('algorithm', 'features', 'train_n', 'train_pos')}))
