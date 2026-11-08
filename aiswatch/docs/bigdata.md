# Đồ án IT4931: Cảnh báo sớm tàu "shadow fleet" tại eo biển Đan Mạch bằng dữ liệu AIS

> Cập nhật 08/11/2026.
> Báo cáo mô hình chi tiết: `research/report/model_report.md` (bản trong gói: `aiswatch/model/report/`).

---

# Phần 1. Bài toán

## 1.1 Bối cảnh

- Mọi chuyến dầu từ cảng Baltic của Nga đều phải qua eo biển Đan Mạch, ở một trong ba cửa: **Skagen, Great Belt, Øresund**.
- Đan Mạch không chặn được tàu, nhưng kiểm tra được tàu neo hoặc đi qua.
- Mỗi tháng có khoảng 800–1.000 tàu dầu đi qua. Hệ thống trả lời câu hỏi: **nên để mắt tới tàu nào trước?**

## 1.2 Định nghĩa

| Thành phần | Định nghĩa |
|---|---|
| Chấm điểm | Chạy **mỗi ngày**, cho mọi tàu dầu có mặt trong 30 ngày qua và chưa bị cấm |
| Đầu vào | **Chỉ hành vi AIS**: cửa sổ 30 ngày + hành trình qua 3 cửa trong 180 ngày. Không dùng hồ sơ tàu (cờ, tuổi, kích thước) |
| Nhãn | y = 1 nếu tàu bị **EU hoặc Anh** niêm yết lần đầu trong 6 tháng sau ngày chấm (OpenSanctions, ngày pháp lý) |
| Đầu ra 1 | **Danh sách theo dõi** top 30 mỗi ngày |
| Đầu ra 2 | **Cảnh báo theo phút**: tàu trong danh sách đi qua một cửa; mất tín hiệu ≥ 2 giờ trong vùng có sóng tốt |

Hệ thống không kết luận vi phạm. Nó chỉ đưa ra thứ tự ưu tiên để con người kiểm tra.

## 1.3 Dữ liệu

| | Giá trị |
|---|---|
| AIS | Kho Cục Hàng hải Đan Mạch (`aisdata.ais.dk`), 01/04/2025–31/03/2026, 365 ngày, 234 GB zip |
| Số dòng | 6,89 tỷ dòng thô → 3,95 tỷ sau khử trùng (khoảng 40% bản tin trùng do nhiều trạm cùng thu) |
| Lưu trữ | Silver 61 GB Parquet; tầng ngày 198 MB cho cả năm |
| Nhãn | OpenSanctions: `eu_fsf`, `eu_sanctions_map`, `eu_journal_sanctions`, `gb_fcdo_sanctions`. 767 tàu có ngày niêm yết, gộp theo IMO |

## 1.4 Phát hiện từ dữ liệu

- **Shadow fleet ở Đan Mạch không lén lút:**
  - 221.686 phiên gặp nhau trong năm, chỉ 18 phiên có tàu bị cấm.
  - Tàu sắp bị cấm có khoảng mất tín hiệu dài nhất ngắn hơn tàu khác.
- **Chúng chạy như con thoi xuất dầu:**
  - Ra đầy hàng: mớn nước ra − vào khoảng **5,7 m**, tàu khác 0 m.
  - Ở trong Baltic **12 ngày** mỗi lượt, tàu khác 6,5 ngày.
  - Không ghé cảng; quãng đường mỗi ngày dài.
- **Tàu đã bị cấm vẫn đi qua:** từ 26 tàu/tháng (04/2025) lên **124 tàu/tháng (03/2026)**, khoảng 13% số tàu dầu.

## 1.5 Mô hình và kết quả

| | Train | Test |
|---|---|---|
| Snapshot cuối tháng | 202504–202509 | 202510–202603 |
| Dòng / dương / tàu dương | 5.339 / 169 / 99 | 4.818 / 50 / 19 |

**Quy trình:**
- CV 5 fold chia theo tàu, chỉ trên train.
- 60 feature hành vi → chọn còn **10 feature** → so 5 thuật toán → Optuna → **XGBoost**.

**10 feature:**
- quãng đường mỗi ngày;
- mớn nước khi ra so với mức tối đa;
- mớn nước ra − vào;
- số ngày ở trong Baltic;
- tỉ lệ thời gian ở cảng;
- tỉ lệ khai điểm đến Nga;
- tốc độ trung bình;
- khoảng mất tín hiệu dài nhất;
- lượt vào mỗi 30 ngày;
- điểm đến mơ hồ.

**Kết quả:**
- **Test:** ROC-AUC **0,90** (CI 0,86–0,94). PR-AUC 0,076, khoảng 8 lần mức ngẫu nhiên.
- **Chấm hằng ngày** (backtest 182 ngày): ROC-AUC trung vị **0,90**.
- **Cảnh báo sớm:** 24/34 tàu sắp bị cấm lọt top 30 trước ngày niêm yết, sớm hơn **trung vị 81 ngày**. DOVE và HORAE sớm 6 tháng.
- **Tàu điểm cao chưa bị cấm:** tàu từng lọt top 10 về sau bị cấm với tỉ lệ 8%, gấp 4 lần mức nền 2%.

**Hạn chế:**
- Test nhỏ: 19 tàu dương.
- Nhãn EU/UK là nhãn thay thế.
- Mớn nước do tàu tự khai.
- Feature "lượt vào mỗi 30 ngày" trôi (PSI 0,58).

---

# Phần 2. Tool stack cần deploy

## 2.1 Kiến trúc Lambda

```
Batch (Airflow, 1 DAG run / ngày):
  aisdata.ais.dk ─► bronze ─► silver ─► tầng ngày ─► feature 30/180 ngày ─► chấm XGBoost (Spark) ─► MongoDB.watchlist
                     │ GX        │ GX       │ GX                         │ GX
                     └──────────── Parquet trên MinIO ────────────────────┘
Speed:
  producer phát lại (×50) ─► Kafka ─► Spark Structured Streaming ─► MongoDB.alerts / positions
      luật: tàu trong watchlist vào polygon cửa; mất tín hiệu ≥ 2 giờ trong ô có sóng tốt
Phục vụ:   8501 trang giám sát (MapLibre + deck.gl) + 8502 Streamlit, đọc MongoDB
Giám sát:  Prometheus ◄─ node, pod, Airflow, Kafka, Spark, MinIO, MongoDB, GX, mô hình ─► Grafana
```

| Yêu cầu môn | Đáp ứng | Tình trạng |
|---|---|---|
| Lambda | Batch hằng ngày + speed theo phút | Batch xong; speed chưa |
| Spark | Batch ETL + chấm điểm; Structured Streaming | Batch xong; streaming chưa |
| Lưu trữ phân tán | MinIO (S3) | Chưa; hiện là Parquet trên đĩa |
| Message queue | Kafka | Chưa |
| NoSQL | MongoDB | Chưa |
| K8s | k3s | Chưa; hiện chạy venv + systemd |

## 2.2 Danh sách tool

| Tool | Vai trò | Cách dựng |
|---|---|---|
| **k3s** | Cụm K8s | 1 node; có sẵn Traefik (ingress) và local-path (volume) |
| **Docker** + registry | Image chứa gói `aiswatch` | `python:3.12-slim` + `openjdk-17-jre-headless` + `requirements.txt` + `great_expectations` + thư mục `aiswatch/` |
| **Airflow** (+ Postgres) | Điều phối batch | Helm chart `apache-airflow/airflow`, `KubernetesExecutor`; Postgres kèm theo làm metadata DB (mục 2.3) |
| **Great Expectations** | Kiểm chất lượng | Task trong DAG; Data Docs lưu lên MinIO, xem qua nginx (mục 2.4) |
| **MinIO** | Kho dữ liệu | Bucket `aiswatch`. Đường dẫn `s3a://`, jar `hadoop-aws`, cấu hình S3 trong `silver.session()`. Trước khi chuyển: PVC khoảng 40 Gi ở `/data` |
| **Kafka** | Hàng đợi bản tin | 1 broker KRaft; topic `ais.positions`, `ais.dlq` |
| **Producer phát lại** | Đẩy AIS vào Kafka | Script Python đọc silver hoặc quỹ đạo một ngày, gửi theo thời gian với hệ số ×1…×50 |
| **Spark Structured Streaming** | 2 luật theo phút | Broadcast watchlist mới nhất; state theo tàu cho luật mất tín hiệu; watermark và checkpoint trên MinIO. Java 17 (`applyInPandasWithState` cần Arrow) |
| **MongoDB** | Lớp phục vụ | Collection `watchlist`, `alerts`, `positions` (index `2dsphere`, TTL) |
| **Deployment + Service** ×2 | 8501 (`./serve.sh viz`), 8502 (`./serve.sh app`) | Probe `/` và `/_stcore/health` |
| **ConfigMap / Secret** | `config.env` / proxy | Mount đè `/opt/aiswatch/config.env` (subPath); `BIND=0.0.0.0` |
| **Ingress** | Truy cập ngoài | Traefik + basic auth cho 8501, 8502, Airflow, Grafana, Data Docs |
| **Prometheus + Grafana** | Dashboard giám sát hệ thống | `kube-prometheus-stack` (mục 2.5) |

## 2.3 Airflow: DAG `aiswatch_daily`

- **Lịch:**
  - `schedule="0 6 * * *"`, `start_date = X`, `catchup=True`, `max_active_runs=1`.
  - Bật DAG thì Airflow tự xử lý lần lượt từ ngày X tới hôm nay.
- **Ngày xử lý:** mỗi run xử lý ngày `{{ ds }}` − 1, vì kho Đan Mạch có file ngày D vào hôm sau.
- **Cách chạy task:** mỗi task là một `KubernetesPodOperator` dùng image `aiswatch`, với giới hạn CPU/RAM riêng.
- **Lệnh theo bước:** `python -m aiswatch.step <bước> --day D`. Cần viết lớp bọc quanh các hàm đã có: `bronze.build`, `silver.build`, `features_day.build`, `tracks.extract`, `rolling.build`, `predict.score_day`, `publish.*`.

```
refresh_labels ─► download ─► bronze ─► gx_bronze ─► silver ─► gx_silver ─► features_day ─► gx_features_day ─┬─► tracks
                                                                                                             └─► rolling
     rolling ─► gx_features ─► predict ─► gx_predict ─► publish (MongoDB + web) ─► cleanup
```

- **Retry:**
  - `download`: retries=3, retry_delay=30 phút.
  - Các task Spark: retries=1.
- **Phụ thuộc giữa các ngày:**
  - `rolling` cần đủ 30 ngày `features_day` trước đó (`ExternalTaskSensor` hoặc kiểm tra file); chưa đủ thì skip.
  - `predict` đặt `depends_on_past=True`.
- **SLA:** cả DAG xong trong 30 phút.

## 2.4 Great Expectations

Mỗi tầng có một Checkpoint chạy ngay sau bước tạo ra nó.
- Fail ở bronze hoặc silver: **dừng DAG**.
- Fail ở feature hoặc predict: **cảnh báo**, vẫn publish.

| Tầng | Expectation | Nguồn ngưỡng |
|---|---|---|
| Bronze | 26 cột đúng thứ tự; `ts` không null và nằm trong ngày D; số dòng khớp số dòng CSV; 12–30 triệu dòng | `bronze.py` |
| Silver | Tổng `n_copies` = số dòng bronze; tỉ lệ trùng 30–55%; tỉ lệ cờ vị trí lỗi < 2%; MMSI không null | `silver.py`, `_verification.json` |
| Tầng ngày | 3.000–7.000 MMSI; > 100 tàu dầu; quãng đường ≥ 0; lượt qua gate ≥ 0 | `features_day.py`, `_meta.json` |
| Feature | 10 feature mô hình: tỉ lệ null dưới ngưỡng, nằm trong khoảng lúc train; PSI so với train < 0,25 (custom expectation) | `model/meta.json`, `report/final/psi_final.json` |
| Predict | Điểm trong [0, 1]; 300–1.200 tàu được chấm; top 30 không rỗng | `predict.py` |

- **Data Docs:** HTML sinh sau mỗi lần chạy, lưu lên MinIO, xem qua Ingress.
- **Kết quả validation** (số expectation đạt/fail theo tầng): đẩy lên Prometheus Pushgateway.

## 2.5 Dashboard giám sát hệ thống: Prometheus + Grafana

Cài bằng `kube-prometheus-stack`. Grafana Loki là tuỳ chọn, nếu muốn xem log trong Grafana.

| Thành phần | Nguồn metric |
|---|---|
| Node, pod, PVC | node-exporter, kube-state-metrics (có sẵn) |
| Airflow | `statsd` → `statsd-exporter` (tuỳ chọn trong Helm chart) |
| Kafka | `kafka-exporter` (consumer lag) |
| Spark streaming | `spark.ui.prometheus.enabled=true` + `PrometheusServlet`, hoặc `StreamingQueryProgress` → Pushgateway |
| MinIO | `/minio/v2/metrics/cluster` |
| MongoDB | `mongodb-exporter` |
| GX, mô hình | Pushgateway: kết quả validation, số dòng, tỉ lệ trùng, PSI, số tàu được chấm, kích thước watchlist, số cảnh báo |

**Dashboard:**
1. **Hạ tầng:** CPU, RAM, đĩa của node; pod restart; dung lượng PVC/MinIO.
2. **Pipeline batch:** DAG run thành công/thất bại theo ngày; thời gian từng task; số ngày đã xử lý so với số ngày tới hôm nay.
3. **Streaming:** Kafka lag; input rate so với processing rate; batch duration; độ trễ cảnh báo p95; số cảnh báo theo giờ.
4. **Dữ liệu và mô hình:** % expectation đạt theo tầng; dòng thô và dòng sạch mỗi ngày; tỉ lệ trùng; PSI 10 feature theo ngày; số tàu được chấm; phân phối điểm.

**Cảnh báo** (Alertmanager gửi Telegram/Slack/email):
- DAG fail;
- không có ngày mới quá 2 ngày;
- Kafka lag tăng liên tục 10 phút;
- đĩa trống < 15%;
- GX fail ở bronze/silver;
- PSI > 0,25.

## 2.6 Thứ tự triển khai

1. Docker image → `docker run` chế độ demo.
2. k3s: ConfigMap + PVC + Job demo + 2 Deployment + Ingress.
3. `kube-prometheus-stack` → dashboard hạ tầng.
4. Airflow + `aiswatch.step` → DAG backfill khoảng 1 tháng; `statsd-exporter` → dashboard pipeline.
5. GX checkpoint (bronze/silver trước, rồi feature/predict), Data Docs, Pushgateway → dashboard dữ liệu.
6. Kafka + producer phát lại + Structured Streaming → MongoDB; giao diện đọc MongoDB → dashboard streaming.
7. Chuyển PVC sang MinIO.
8. Tuỳ chọn: Loki; producer aisstream live (cần key).

## 2.7 Kịch bản demo

1. Trang giám sát 8501 hiện danh sách theo dõi hôm nay.
2. Phát lại một ngày ở ×50: tàu trong danh sách qua cửa Skagen thì cảnh báo hiện trong vài giây; Grafana thấy input rate và lag.
3. Xoá pod streaming: K8s dựng lại, chạy tiếp từ checkpoint; Grafana thấy pod restart và lag giảm dần.
4. Airflow UI: DAG hằng ngày, backfill từ ngày X; Data Docs của GX (khoảng 40% dòng trùng, cờ lỗi).
5. Trang 8502: kết quả mô hình, cảnh báo sớm (DOVE, HORAE sớm 6 tháng); dashboard PSI.

## 2.8 Lưu ý khi deploy

- **Mạng:** server phải tới được `aisdata.ais.dk` và `data.opensanctions.org`.
- **Tài nguyên:** mỗi ngày dữ liệu tốn khoảng 9 phút, 4 GB RAM cho Spark và khoảng 5 GB đĩa tạm. Lưu lâu dài chỉ khoảng 1,5 MB mỗi ngày.
- **Ngày bắt đầu:**
  - Chấm được từ X + 29.
  - Muốn điểm ổn định từ ngày S thì đặt `START_DATE` = S − 6 tháng và `SCORE_FROM = S`.
- **Nhãn:** khi chạy live, tải lại nhãn mỗi ngày (bỏ `--no-label-download`). Nếu tải lỗi, pipeline dùng bản cũ.
- **Image:** `JAVA_HOME` khác nhau giữa amd64 và arm64.

---

# Phần 3. Code đã làm

## 3.1 Gói triển khai `aiswatch`

- **Mã nguồn:** `aiswatch/`.
- **Kết quả demo:** `demo/aiswatch-demo-data.tar.gz` (33 MB, kết quả 10/2025–03/2026).
- **Chỉ suy luận:** 5 mô hình XGBoost (5 seed) và `model/meta.json` (10 feature, tham số).

```bash
./install.sh                                   # venv + thư viện (cần Java 17+)
./run.sh --start 2025-10-01                    # xử lý từ ngày X tới hôm qua + chấm điểm
./run.sh demo ../demo/aiswatch-demo-data.tar.gz   # nạp kết quả có sẵn
./serve.sh                                     # 8501 trang giám sát, 8502 Streamlit
```

| File | Việc |
|---|---|
| `aiswatch/pipeline.py` | Điều phối từ ngày X tới hôm nay; bỏ qua bước đã xong |
| `aiswatch/bronze.py` | Zip → CSV tạm → Spark → Parquet 26 cột + bảng tóm tắt theo MMSI; đối chiếu số dòng, MMSI, checksum |
| `aiswatch/silver.py` | Spark: khử trùng (`n_copies`), 3 cờ lỗi, đối chiếu; `session()` cấu hình Spark |
| `aiswatch/features_day.py` | Spark: đại lượng cộng được theo MMSI/ngày, lượt qua cửa, ô 5 phút, phiên gặp nhau |
| `aiswatch/labels.py`, `identity.py` | Nhãn EU/UK; ghép MMSI → tàu theo IMO |
| `aiswatch/rolling.py` | Feature 30 ngày + hành trình 180 ngày |
| `aiswatch/predict.py` | Chấm điểm Spark (broadcast mô hình, `rdd.mapPartitions`), gộp kết quả |
| `aiswatch/tracks.py`, `publish.py` | Quỹ đạo 10 phút; dữ liệu cho 2 giao diện |
| `web/index.html` | Trang giám sát 8501: danh sách theo dõi, phát lại quỹ đạo, cảnh báo qua cửa / mất tín hiệu |
| `app/app.py` | Streamlit 8502: câu chuyện, danh sách hằng ngày, chi tiết tàu, mạng gặp gỡ, cảnh báo sớm, đánh giá |
| `systemd/`, `config.env.example`, `README.md` | Chạy nền trên 1 server; hướng dẫn |

**Kiểm thử:**
- Điểm khớp bản nghiên cứu 99,84% (cùng AUC 0,897).
- Ngày thật 31/03/2026: mọi tầng giống hệt bản nghiên cứu.
- Cài mới thì 2 cổng và 6 trang app chạy được.

**Yêu cầu môi trường:**
- PySpark 3.5 trên Python 3.12 cần `shims/distutils` và `setuptools<81`.
- Đặt `PYSPARK_PYTHON=sys.executable`.
- Streamlit cần `--global.developmentMode false`.

## 3.2 Code nghiên cứu

Thư mục `research/`. Code đọc dữ liệu ở `research/data/` (silver 61 GB), không có trong repo; muốn chạy lại phải tải và ETL lại 04/2025–03/2026.

| Bước | File |
|---|---|
| ETL theo ngày | `ingest.py`, `bronze.py`, `silver.py`, `features_day.py` |
| Nhãn | `labels.py` |
| Feature | `features_window.py` (30 ngày), `voyage_features.py` (hành trình 180 ngày), `graph_features.py` (gặp gỡ) |
| Chọn feature, so thuật toán, Optuna, báo cáo | `model_report.py` (`FEATSET=final`) → `report/final/`, `report/model_report.md` |
| Mô hình triển khai | `train_final.py` → `models/final/` |
| Chấm hằng ngày | `features_rolling.py` → `predict_daily.py` (Spark) → `compact_predict.py`; `run_daily_chain.sh` |
| Phân tích, giao diện | `insights.py`, `tracks_extract.py`, `viz_build.py`, `app_prep.py`, `app.py`, `viz/index.html` |

## 3.3 Việc còn lại

1. K8s + Prometheus/Grafana (mục 2.6, bước 1–3).
2. Airflow DAG + `aiswatch.step` + GX checkpoint (mục 2.3–2.4).
3. Kafka + producer phát lại + Structured Streaming → MongoDB.
4. Chuyển lưu trữ sang MinIO.
5. Đối chiếu OFAC/GUR cho các tàu điểm cao chưa bị cấm (mục 6 của `model_report.md`).
6. Train lại khi mọi snapshot có đủ 180 ngày lịch sử.
7. Viết lessons learned 11 mảng (Phụ lục A).

---

# Phụ lục A. Lessons learned

| Mảng | Ghi chú |
|---|---|
| Thu thập | S3 công khai; tải theo ngày, kiểm đủ byte, thử lại 3 lần; xử lý xong xoá zip |
| Xử lý Spark | Tuần tự từng ngày; tầng ngày chỉ lưu đại lượng cộng được rồi gộp thành cửa sổ 30/180 ngày |
| Streaming | Watchlist từ batch + 2 luật theo vị trí, state theo tàu, checkpoint |
| Lưu trữ | Bronze (đủ 26 cột) → silver (khử trùng, gắn cờ) → tầng ngày: 234 GB zip → 61 GB → 198 MB |
| Tích hợp | Giải nén CSV tạm cho Spark; ghép tàu theo IMO hợp lệ checksum, không dùng danh sách cấm để ghép |
| Hiệu năng | Silver chia 64 partition hash MMSI, sắp theo MMSI/ts: tầng ngày 70–90 giây/ngày; broadcast mô hình khi chấm |
| Giám sát | Prometheus/Grafana; mỗi bước ghi `_verification.json` / `_meta.json` / `status.json` |
| Scaling | Các ngày độc lập tới bước tầng ngày, nên chạy song song được; Spark executor động trên K8s |
| Chất lượng dữ liệu | 40% bản tin trùng; lat = 91 / lon = 0; trạm bờ và phao; giá trị UNKNOWN; tàu nhiều MMSI; tàu đổi mã loại tàu. Gắn cờ và đếm, kiểm bằng GX |
| Bảo mật | Dữ liệu công khai; thông tin proxy trong Secret; giao diện sau Ingress có xác thực |
| Chịu lỗi | Mọi bước idempotent (ghi staging rồi đổi tên, bỏ qua bước đã xong); Airflow retry; checkpoint Spark streaming |

# Phụ lục B. Nguồn

- Trang môn: https://datalab.vn/teaching/it4931-luu-tru-va-xu-ly-du-lieu-lon
- AIS Đan Mạch: http://aisdata.ais.dk · OpenSanctions: https://data.opensanctions.org/datasets/latest/
- aisstream: https://aisstream.io/documentation · GUR: https://source.coop/cholmes/gur-shadow-fleet/vessels
- Airflow Helm chart: https://airflow.apache.org/docs/helm-chart/stable/ · Great Expectations: https://docs.greatexpectations.io
- kube-prometheus-stack: https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack
- Spark on K8s: https://spark.apache.org/docs/3.5.1/running-on-kubernetes.html · k3s: https://docs.k3s.io
