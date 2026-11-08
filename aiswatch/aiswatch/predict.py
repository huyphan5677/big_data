"""Chấm điểm một ngày bằng Spark (theo mẫu users/models/predict_base.py): broadcast mô hình, chấm theo lô trong partition.

Dùng rdd.mapPartitions thay vì pandas_udf: Arrow của Spark 3.5 lỗi trên Java 21; đường pickle chạy được trên Java 17/21.
Đầu ra: data/predict/date=D/ (điểm trung bình 5 seed, SHAP seed 0, hạng toàn bộ và hạng danh sách theo dõi).
"""
import json, shutil
import pandas as pd
from .config import DATA, MODEL, WATCH_DAYS

OUT = DATA/'data/predict'
KEEP = ['vessel_id', 'date', 'y', 'first_listed', 'label_id', 'days_since_seen', 'last_seen', 'last_lat', 'last_lon', 'days_present_l1m']
META = json.load(open(MODEL/'meta.json'))
FE = META['features']


def load_model(path):
    if META['algorithm'] == 'XGBoost':
        import xgboost as xgb
        m = xgb.XGBClassifier(); m.load_model(str(path)); return m
    from catboost import CatBoostClassifier
    return CatBoostClassifier().load_model(str(path))


def shap_values(m, X):
    if META['algorithm'] == 'XGBoost':
        import xgboost as xgb
        return m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    from catboost import Pool
    return m.get_feature_importance(Pool(X), type='ShapValues')[:, :-1]


_BC = {}


def score_day(spark, day):
    from pyspark.sql import functions as F, Window
    if 'bc' not in _BC:
        _BC['bc'] = spark.sparkContext.broadcast([load_model(MODEL/f"model_seed{s}.{META['ext']}") for s in META['seeds']])
    bc = _BC['bc']
    schema = ('vessel_id string, date string, y int, first_listed string, label_id string, days_since_seen int, last_seen string, '
              'last_lat double, last_lon double, days_present_l1m double, score double, ' + ', '.join(f'{f} double, shap_{f} double' for f in FE))
    cols = KEEP + ['score'] + [c for f in FE for c in (f, f'shap_{f}')]
    algo = META['algorithm']

    def score(rows):
        import numpy as np, pandas as pd
        ms = bc.value
        pdf = pd.DataFrame([r.asDict() for r in rows], columns=KEEP+FE)
        if not len(pdf): return
        X = pdf[FE].astype(float)
        s = np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)
        if algo == 'XGBoost':
            import xgboost as xgb
            sv = ms[0].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        else:
            from catboost import Pool
            sv = ms[0].get_feature_importance(Pool(X), type='ShapValues')[:, :-1]
        out = pdf[KEEP].copy(); out['days_present_l1m'] = out.days_present_l1m.astype(float); out['score'] = s
        for i, f in enumerate(FE): out[f] = X[f].values; out[f'shap_{f}'] = sv[:, i]
        out = out.astype(object).where(out.notna(), None)
        out['y'] = out.y.map(int); out['days_since_seen'] = out.days_since_seen.map(int)
        yield from (tuple(r) for r in out[cols].itertuples(index=False))

    src = DATA/f'data/features_rolling/date={day}/features.parquet'
    df = spark.read.parquet(str(src)).select(*KEEP, *FE).fillna('', ['first_listed', 'label_id'])
    res = spark.createDataFrame(df.repartition(2).rdd.mapPartitions(score), schema)
    res = res.withColumn('rank_all', F.row_number().over(Window.orderBy(F.desc('score'))))
    watch = F.col('days_since_seen') < WATCH_DAYS
    res = res.withColumn('rank_watch', F.when(watch, F.row_number().over(Window.partitionBy(watch).orderBy(F.desc('score')))))
    target = OUT/f'date={day}'; stage = OUT/f'.staging-{day}'
    shutil.rmtree(stage, ignore_errors=True); shutil.rmtree(target, ignore_errors=True)
    res.coalesce(1).write.parquet(str(stage)); stage.rename(target)
    return pd.read_parquet(target)


def compact():
    """Gộp mọi ngày thành data/predict_all.parquet (web và app đọc file này)."""
    parts = [pd.read_parquet(p).assign(date=p.name.split('=')[1]) for p in sorted(OUT.glob('date=*'))]
    P = pd.concat(parts, ignore_index=True).sort_values(['date', 'rank_all'])
    tmp = DATA/'data/.predict_all.parquet'; P.to_parquet(tmp, index=False, compression='zstd'); tmp.rename(DATA/'data/predict_all.parquet')
    return P
