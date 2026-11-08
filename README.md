# aiswatch: cảnh báo sớm tàu "shadow fleet" tại eo biển Đan Mạch (IT4931)

Tài liệu chính: **[docs/bigdata.md](docs/bigdata.md)**, gồm bài toán, tool stack cần deploy và code đã làm.

| Thư mục | Nội dung |
|---|---|
| `aiswatch/` | Gói triển khai (chỉ suy luận, mô hình đã train sẵn). Cách chạy: [aiswatch/README.md](aiswatch/README.md) |
| `demo/aiswatch-demo-data.tar.gz` | Kết quả có sẵn 10/2025–03/2026 cho chế độ demo |
| `research/` | Code nghiên cứu: ETL, feature, chọn feature, train, backtest, giao diện |
| `research/report/` | Báo cáo mô hình `model_report.md` + hình + số liệu gốc |
| `research/models/final/` | 5 mô hình XGBoost + `meta.json` (giống `aiswatch/model/`) |

Chạy nhanh chế độ demo:

```bash
cd aiswatch && ./install.sh
sed -i 's#^AISWATCH_DATA=.*#AISWATCH_DATA='"$PWD"'/../_data#' config.env && mkdir -p ../_data
./run.sh demo ../demo/aiswatch-demo-data.tar.gz
./serve.sh        # http://localhost:8501 (giám sát) · http://localhost:8502 (kết quả mô hình)
```
