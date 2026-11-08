"""Chấm điểm hằng ngày bằng Spark, theo mẫu users/models/predict_base.py.

- Broadcast các mô hình (5 seed, thuật toán theo models/final/meta.json) xuống executor.
- mapInPandas tính điểm (trung bình 5 seed) và SHAP (seed 0) cho từng tàu.
- Xếp hạng trong ngày theo hai cách: trên toàn bộ tàu, và trên các tàu thấy trong 3 ngày gần nhất (danh sách theo dõi).
- Lặp theo ngày; ngày nào đã có kết quả thì bỏ qua. Ghi data/predict/date=D/ (staging rồi rename).
"""
import argparse, json, os, sys, shutil
from pathlib import Path
from train_model import ROOT

MODEL = ROOT/'models/final'
OUT = ROOT/'data/predict'
KEEP = ['vessel_id', 'date', 'y', 'first_listed', 'label_id', 'days_since_seen', 'last_seen', 'last_lat', 'last_lon', 'days_present_l1m']  # FE từ meta.json
WATCH_DAYS = 3


def load_model(algo, path):
    if algo == 'XGBoost':
        import xgboost as xgb
        m = xgb.XGBClassifier(); m.load_model(str(path)); return m
    if algo == 'CatBoost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier().load_model(str(path))
    raise ValueError(algo)


def shap_values(algo, m, X):
    """TreeSHAP gốc của thư viện (bỏ cột bias)."""
    if algo == 'XGBoost':
        import xgboost as xgb
        return m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    from catboost import Pool
    return m.get_feature_importance(Pool(X), type='ShapValues')[:, :-1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--start', default='2025-10-01'); ap.add_argument('--end', default='2026-03-31')
    a = ap.parse_args()
    os.environ['PYSPARK_PYTHON'] = sys.executable
    from silver import session
    from pyspark.sql import functions as F, Window
    spark = session(app_name='AIS-predict-daily', cores=2, driver_memory='2g')
    spark.conf.set('spark.sql.shuffle.partitions', '4')
    meta = json.load(open(MODEL/'meta.json')); FE = meta['features']
    ALGO = meta['algorithm']
    models = [load_model(ALGO, MODEL/f"model_seed{s}.{meta['ext']}") for s in meta['seeds']]
    bc = spark.sparkContext.broadcast(models)
    schema = ('vessel_id string, date string, y int, first_listed string, label_id string, days_since_seen int, last_seen string, '
              'last_lat double, last_lon double, days_present_l1m double, score double, '
              + ', '.join(f'{f} double, shap_{f} double' for f in FE))
    COLS = KEEP + ['score'] + [c for f in FE for c in (f, f'shap_{f}')]

    def score(rows):
        # Chấm theo lô trong từng partition. Không dùng mapInPandas/pandas_udf: Arrow của Spark 3.5 lỗi trên Java 21
        # (DirectByteBuffer.<init>(long,int) đã bị bỏ); đi đường pickle qua rdd.mapPartitions thì chạy được.
        import numpy as np, pandas as pd
        ms = bc.value
        pdf = pd.DataFrame([r.asDict() for r in rows], columns=KEEP+FE)
        if len(pdf):
            X = pdf[FE].astype(float)
            s = np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)
            sv = shap_values(ALGO, ms[0], X)
            out = pdf[KEEP].copy()
            out['days_since_seen'] = out.days_since_seen.astype('int32'); out['y'] = out.y.astype('int32'); out['days_present_l1m'] = out.days_present_l1m.astype(float)
            out['score'] = s
            for i, f in enumerate(FE):
                out[f] = X[f].values; out[f'shap_{f}'] = sv[:, i]
            out = out.astype(object).where(out.notna(), None)
            out['y'] = out.y.map(int); out['days_since_seen'] = out.days_since_seen.map(int)   # kiểu Python cho schema int
            yield from (tuple(r) for r in out[COLS].itertuples(index=False))

    import pandas as pd
    for D in pd.date_range(a.start, a.end):
        day = str(D.date()); target = OUT/f'date={day}'
        if target.exists():
            continue
        src = ROOT/f'data/features_rolling/date={day}/features.parquet'
        if not src.exists():
            print(json.dumps(dict(date=day, skipped='no features')), flush=True); continue
        df = spark.read.parquet(str(src)).select(*KEEP, *FE).fillna('', ['first_listed', 'label_id'])
        res = spark.createDataFrame(df.repartition(2).rdd.mapPartitions(score), schema)
        w_all = Window.orderBy(F.desc('score'))
        res = res.withColumn('rank_all', F.row_number().over(w_all))
        watch = F.col('days_since_seen') < WATCH_DAYS
        res = res.withColumn('rank_watch', F.when(watch, F.row_number().over(Window.partitionBy(watch).orderBy(F.desc('score')))))
        stage = OUT/f'.staging-{day}'; shutil.rmtree(stage, ignore_errors=True)
        res.coalesce(1).write.parquet(str(stage))
        stage.rename(target)
        st = spark.read.parquet(str(target)).agg(F.count('*').alias('n'), F.sum('y').alias('pos'),
                                                 F.sum(F.when(F.col('rank_watch') <= 30, F.col('y'))).alias('pos_in_watch30')).first()
        print(json.dumps(dict(date=day, **st.asDict())), flush=True)
    spark.stop()


if __name__ == '__main__':
    main()
