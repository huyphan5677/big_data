# aiswatch: giám sát tàu dầu qua eo biển Đan Mạch (gói triển khai)

Gói này **chỉ chạy suy luận**: tải AIS → làm sạch → tính feature → chấm điểm bằng **mô hình đã train sẵn** (`model/`) → hiển thị kết quả.
Không có bước train, chọn feature hay tinh chỉnh tham số.

| Cổng | Giao diện | Nội dung |
|---|---|---|
| **8501** | Trang giám sát (HTML tĩnh: MapLibre + deck.gl) | Danh sách theo dõi mỗi ngày, bản đồ phát lại quỹ đạo 10 phút, cảnh báo theo phút (tàu theo dõi qua cửa Skagen / Great Belt / Øresund, mất tín hiệu ≥ 2 giờ), chi tiết tàu |
| **8502** | Kết quả mô hình (Streamlit) | Câu chuyện, danh sách hằng ngày, chi tiết tàu, mạng lưới gặp nhau, cảnh báo sớm, đánh giá mô hình |

## 1. Yêu cầu

- Linux, **Python ≥ 3.10** (đã kiểm thử với 3.12), **Java 17+** (Spark 3.5 chạy trên JVM).
- 4 CPU, 8 GB RAM. Spark dùng khoảng 4 GB khi xử lý một ngày.
- Ổ đĩa:
  - khoảng 1,5 GB tạm thời cho mỗi ngày đang xử lý (zip, bronze, silver), xoá ngay sau khi xử lý xong;
  - khoảng 1,5 MB lâu dài cho mỗi ngày (feature, kết quả, quỹ đạo).
- Internet tới `aisdata.ais.dk` (AIS) và `data.opensanctions.org` (danh sách trừng phạt). Nếu server cần proxy, đặt `HTTPS_PROXY` / `HTTP_PROXY` trong `config.env`.

## 2. Cài đặt

```bash
tar -xzf aiswatch.tar.gz -C ~ && cd ~/aiswatch
./install.sh          # tạo .venv và cài requirements.txt
nano config.env       # AISWATCH_DATA, START_DATE, cổng...
```

## 3. Chạy

### Chế độ thật: xử lý từ ngày X tới hôm nay

```bash
./run.sh                               # dùng START_DATE trong config.env, chạy tới hôm qua (UTC)
./run.sh --start 2026-01-01            # ghi đè ngày bắt đầu
./run.sh --start 2025-04-01 --score-from 2025-10-01 --end 2026-03-31   # khoảng cố định
./serve.sh                             # mở 8501 + 8502 (hoặc: ./serve.sh viz | ./serve.sh app)
```

Với mỗi ngày D, pipeline (`aiswatch/pipeline.py`) làm lần lượt:
1. Tải `aisdk-D.zip` → **bronze**: đọc CSV từ zip, đối chiếu số dòng, số MMSI và checksum.
2. **silver** (Spark): khử trùng lặp, gắn cờ dòng lỗi. Sau đó xoá zip và bronze.
3. **Tầng ngày** (Spark): các đại lượng cộng được cho từng MMSI.
4. **Quỹ đạo 10 phút** (Spark). Sau đó xoá silver cũ.
5. Với mọi ngày đã có đủ **30 ngày** dữ liệu trước đó:
   - feature trượt 30 ngày cộng hành trình qua cửa trong **180 ngày**;
   - **chấm điểm bằng Spark** (broadcast mô hình, chấm theo lô trong từng partition);
   - xuất dữ liệu cho 2 giao diện.

**Lưu ý về ngày bắt đầu:**
- Chỉ chấm được từ ngày X + 29, vì cần đủ cửa sổ 30 ngày.
- Feature hành trình mạnh nhất khi có đủ 180 ngày lịch sử. Muốn xem kết quả ổn định từ ngày S, đặt `START_DATE` sớm hơn S khoảng 6 tháng và `SCORE_FROM = S`.

**Chạy lại an toàn:** bước nào đã có kết quả thì bỏ qua. Ngày bị lỗi tải sẽ báo lỗi rõ; chạy lại sau là được.

**Chạy tự động mỗi sáng** (systemd user):
```bash
mkdir -p ~/.config/systemd/user && cp systemd/* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aiswatch-viz aiswatch-app aiswatch-pipeline.timer
```
Các unit giả định gói nằm ở `~/aiswatch`. Nếu đặt chỗ khác, sửa `WorkingDirectory` và `ExecStart`.

### Chế độ demo: phát lại kết quả có sẵn, không cần tải dữ liệu hay Spark

```bash
./run.sh demo aiswatch-demo-data.tar.gz   # kết quả 10/2025–03/2026 đã tính bằng chính pipeline này
./serve.sh
```
Ở chế độ demo, giao diện có nút **"Đáp án"** cho biết tàu nào sau đó bị EU/UK cấm thật. Ở chế độ thật, nút này là **"Tình trạng cấm"**, chỉ cho biết các lệnh cấm đã có tới hiện tại.

## 4. Truy cập

Mặc định hai giao diện chỉ nghe trên `127.0.0.1`. Từ máy khác, chuyển tiếp cổng qua SSH:
```bash
ssh -L 8501:localhost:8501 -L 8502:localhost:8502 user@server
```
Rồi mở `http://localhost:8501` (giám sát) và `http://localhost:8502` (kết quả mô hình).

Trang 8501 có tham số `?day=YYYY-MM-DD` để mở thẳng một ngày.

## 5. Cấu trúc

```
aiswatch/
  aiswatch/          code chạy (chỉ suy luận)
    pipeline.py      điều phối từ ngày X tới hôm nay
    bronze.py silver.py features_day.py   ETL theo ngày, toàn bộ bằng Spark (kết quả giống hệt lúc xây dựng)
    labels.py identity.py                 nhãn EU/UK, gom MMSI → tàu theo IMO
    rolling.py       feature 30 ngày + hành trình 180 ngày
    predict.py       chấm điểm Spark
    tracks.py publish.py                  quỹ đạo + dữ liệu cho 2 giao diện
  model/             5 mô hình XGBoost (5 seed) + meta.json (10 feature, tham số) + report/ (kết quả lúc xây dựng)
  maps/              bản đồ cảng và vùng phủ sóng cố định (04/2025)
  web/index.html     trang giám sát (8501)
  app/app.py         trang kết quả (8502)
  shims/             distutils cho PySpark 3.5 trên Python 3.12
  systemd/           unit chạy nền + timer hằng ngày
```

Dữ liệu và kết quả nằm trong `$AISWATCH_DATA`:
- `data/predict/date=D`: điểm từng ngày.
- `data/predict_all.parquet`: điểm gộp.
- `data/app/`: bảng cho app kết quả.
- `web/`: trang giám sát cùng dữ liệu của nó.
- `data/status.json`: lần chạy gần nhất.
- `logs/pipeline.log`: nhật ký.

## 6. Mô hình

- XGBoost, train trên các snapshot 04–09/2025, chỉ xét tàu dầu, chỉ dùng hành vi.
- 10 feature:
  - quãng đường mỗi ngày;
  - mớn nước khi ra so với mức tối đa;
  - mớn nước khi ra − khi vào;
  - số ngày ở trong Baltic;
  - thời gian ở cảng;
  - tỉ lệ khai điểm đến Nga;
  - tốc độ trung bình;
  - khoảng mất tín hiệu dài nhất;
  - lượt đi vào Baltic mỗi 30 ngày;
  - điểm đến mơ hồ.
- Kiểm tra trên 10/2025–03/2026: ROC-AUC 0,90; tàu sắp bị cấm lọt top 30 danh sách theo dõi sớm hơn trung vị 81 ngày.
- Chi tiết ở `model/report/model_report.md`. Bài toán, tool stack cần deploy (K8s, Airflow, GX, Kafka, MinIO, MongoDB, Grafana) và việc còn lại: `docs/bigdata.md`.

**Đầu ra là thứ tự ưu tiên để con người xem xét, không phải kết luận vi phạm.**
