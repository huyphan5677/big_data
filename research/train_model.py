"""Walk-forward cho mô hình shadow fleet trên data/features/snapshot=*.

Nhãn snapshot s = tàu bị liệt kê trong (cutoff_s, cutoff_s + 6 tháng]. Nhãn đó chỉ biết chắc khi
đã qua s+6, nên khi chấm tháng test m chỉ train trên snapshot s <= m-6 (khe 6 tháng, không rò nhãn).
Tàu đã bị cấm tại cutoff đã bị loại khỏi cohort từ roster().

Chạy: python train_model.py  -> results/model_walkforward.json, results/model_scores.parquet,
results/model_importance.csv. Một job, pandas + LightGBM 2 luồng.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
HORIZON = 6
ID_COLS = ['vessel_id', 'snapshot', 'group_id', 'label_id', 'first_listed', 'y', 'mmsis']
CAT_COLS = ['ship_type', 'mid']
PARAMS = dict(objective='binary', learning_rate=0.03, n_estimators=400, num_leaves=15,
              min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
              reg_lambda=1.0, n_jobs=2, verbose=-1)
SEEDS = [0, 1, 2, 3, 4]
BOOT = 1000


def load():
    parts = []
    for p in sorted((ROOT/'data/features').glob('snapshot=*/features.parquet')):
        parts.append(pd.read_parquet(p))
    d = pd.concat(parts, ignore_index=True)
    d['snapshot'] = d['snapshot'].astype(str)
    assert not d.duplicated(['snapshot', 'vessel_id']).any()
    d['is_tanker'] = d['is_tanker'].astype(bool)
    d['has_valid_imo'] = d['has_valid_imo'].astype(float)
    for c in CAT_COLS:
        d[c] = d[c].astype('category')
    return d


def months_between(a, b):
    pa, pb = pd.Period(a, 'M'), pd.Period(b, 'M')
    return (pb - pa).n


def topk(y, s, frac):
    k = max(1, int(round(len(y)*frac)))
    idx = np.argsort(-s)[:k]
    return dict(k=k, hits=int(y[idx].sum()), precision=float(y[idx].mean()), recall=float(y[idx].sum()/max(1, y.sum())))


def metrics(df):
    y, s = df.y.to_numpy(), df.score.to_numpy()
    out = dict(n=len(df), pos=int(y.sum()))
    if 0 < y.sum() < len(y):
        out.update(auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s), base_rate=float(y.mean()))
        out['top1pct'] = topk(y, s, 0.01)
        out['top50'] = topk(y, s, 50/len(y))
    t = df[df.is_tanker]
    if 0 < t.y.sum() < len(t):
        out['tanker'] = dict(n=len(t), pos=int(t.y.sum()), auc=roc_auc_score(t.y, t.score),
                             pr_auc=average_precision_score(t.y, t.score), base_rate=float(t.y.mean()))
    return out


def boot_ci(df, rng):
    # bootstrap theo tàu (cùng tàu lặp qua nhiều tháng test -> lấy mẫu cả cụm)
    groups = df.groupby('group_id').indices
    keys = np.array(list(groups))
    aucs, aps = [], []
    for _ in range(BOOT):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        y, s = df.y.to_numpy()[idx], df.score.to_numpy()[idx]
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, s)); aps.append(average_precision_score(y, s))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(auc_ci95=q(aucs), pr_auc_ci95=q(aps), n_boot=len(aucs))


def main():
    d = load()
    feats = [c for c in d.columns if c not in ID_COLS]
    snaps = sorted(d.snapshot.unique())
    tests = [m for m in snaps if any(months_between(s, m) >= HORIZON for s in snaps)]
    scored, folds, imp = [], [], []
    for m in tests:
        train_snaps = [s for s in snaps if months_between(s, m) >= HORIZON]
        tr, te = d[d.snapshot.isin(train_snaps)], d[d.snapshot == m].copy()
        assert te.snapshot.nunique() == 1 and tr.snapshot.max() < m
        s = np.zeros(len(te))
        for seed in SEEDS:
            model = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            model.fit(tr[feats], tr.y, categorical_feature=CAT_COLS)
            s += model.predict_proba(te[feats])[:, 1]/len(SEEDS)
            imp.append(pd.Series(model.booster_.feature_importance('gain'), index=feats, name=f'{m}_{seed}'))
        te['score'] = s
        fold = dict(test=m, train=train_snaps, train_n=len(tr), train_pos=int(tr.y.sum()), **metrics(te))
        folds.append(fold); scored.append(te[['snapshot', 'vessel_id', 'group_id', 'is_tanker', 'y', 'score']])
        print(json.dumps({k: fold[k] for k in ('test', 'train_n', 'train_pos', 'n', 'pos', 'auc', 'pr_auc') if k in fold}), flush=True)
    allsc = pd.concat(scored, ignore_index=True)
    # gộp: chuẩn hoá thứ hạng trong từng tháng để các tháng so được với nhau
    allsc['score_raw'] = allsc.score
    allsc['score'] = allsc.groupby('snapshot').score.rank(pct=True)
    rng = np.random.default_rng(0)
    pooled = dict(**metrics(allsc), **boot_ci(allsc, rng),
                  tanker_ci=boot_ci(allsc[allsc.is_tanker].reset_index(drop=True), rng))
    importance = pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False)
    out = dict(at=datetime.now(timezone.utc).isoformat(timespec='seconds'), horizon_months=HORIZON,
               rule='train snapshots s with s <= test-6 months (labels fully observed)',
               params=PARAMS, seeds=SEEDS, n_features=len(feats), folds=folds, pooled_rank_within_month=pooled,
               top_features=importance.head(25).round(1).to_dict())
    (ROOT/'results/model_walkforward.json').write_text(json.dumps(out, indent=1, default=str))
    allsc.to_parquet(ROOT/'results/model_scores.parquet', index=False)
    importance.rename('gain').to_csv(ROOT/'results/model_importance.csv')
    print(json.dumps(dict(pooled={k: pooled[k] for k in ('n', 'pos', 'auc', 'pr_auc', 'auc_ci95', 'pr_auc_ci95', 'top1pct')},
                          tanker=pooled.get('tanker'), tanker_ci=pooled['tanker_ci']), default=str), flush=True)


if __name__ == '__main__':
    sys.exit(main())
