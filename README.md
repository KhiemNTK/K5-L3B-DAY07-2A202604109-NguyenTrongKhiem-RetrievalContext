# Day 7 — Agentic RAG

Thí nghiệm dense, TF-IDF và hybrid retrieval trên dữ liệu Day 6. Hiện có benchmark và kết quả retrieval dev;
agent dùng model trả lời, đánh giá test và báo cáo cuối chưa hoàn thành.

## Dữ liệu

| Thành phần | Cấu hình |
|---|---|
| Tài liệu | Git & GitHub Fundamentals — Student Edition, Hitesh Patel, Edition 1 (10/2026) |
| Corpus | 53 trang nội dung, trang vật lý 5–57 trong PDF 65 trang |
| Chunk | B: 179 đoạn, tối đa 500 ký tự, không overlap; chọn từ dev Day 6 |
| Embedding | all-MiniLM-L6-v2, 384 chiều, L2; revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Vector store | PostgreSQL 16 + pgvector 0.8.7, cosine exact search |
| Benchmark | Dev: 15 có đáp án + 2 thiếu evidence; test: 5 + 2 |

[Chunk](data/chunks.json) giữ ID, text, nguồn, trang và metadata Day 6.
[Manifest](data/day6.json) ghi nguồn, attribution CC BY-SA 4.0 và checksum; đường dẫn PDF phân giải từ repo Day 6.
Text trích xuất/chuẩn hóa được chia sẻ tiếp theo CC BY-SA 4.0.

## Cách chạy

Môi trường đã kiểm chứng: Windows, Python 3.12 và dependency Day 6. Máy mới hoặc chưa có database:
[làm theo hướng dẫn thiết lập](docs/README.md). Khi đã có môi trường Day 6, chạy từ gốc repo này:

```powershell
$day6Root = (Resolve-Path '../K5-L3B-DAY06-NguyenTrongKhiem-2A202604109-DataIngestion').Path
$labPython = Join-Path $day6Root 'starter/.venv/Scripts/python.exe'
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose -f starter/docker-compose.yml -p day7_retrieval up -d --wait
$env:HF_HUB_OFFLINE = '1'
& $labPython -X utf8 -B starter/benchmark.py
& $labPython -X utf8 -B -m unittest discover -s tests -v
& $labPython -X utf8 -B starter/retrieval.py --day6-root $day6Root --split dev
```

Compose dùng cổng local `15433` và volume Day 6. Dừng PostgreSQL Day 6 trước nếu cùng dùng volume này.
Adapter truy vấn bằng session chỉ đọc; lệnh dừng dịch vụ là `docker compose -f starter/docker-compose.yml -p day7_retrieval stop`.

## Kết quả dev

Cả ba phương pháp dùng cùng corpus, query gốc và filter Unit được nêu rõ trong câu hỏi.
Hybrid dùng RRF `1 / (60 + rank)`, lấy 10 candidate mỗi thành phần, xuất top-3.
Budget context preview là 2.400 ký tự, gồm ID/nguồn/trang.

[Lượt chạy 16/11/2026](results/retrieval/20261116T025510104675Z-dev/summary.json):

| Phương pháp | Hit@1 | Evidence Hit@3 | Context preview có evidence | Median/query |
|---|---:|---:|---:|---:|
| Dense | 12/15 (80%) | 13/15 (86,67%) | 13/15 (86,67%) | 31,65 ms |
| TF-IDF | 13/15 (86,67%) | 15/15 (100%) | 15/15 (100%) | 0,41 ms |
| Hybrid RRF | 13/15 (86,67%) | 14/15 (93,33%) | 14/15 (93,33%) | 25,02 ms |

Mỗi phương pháp chạy đủ 17 câu, không có lỗi. Câu thiếu evidence không thuộc mẫu số retrieval metrics.
Thời gian dense/hybrid gồm embedding query và kết nối DB; setup nằm riêng trong summary. Chi phí API bằng 0,
không gồm điện/phần cứng. Trace giữ ranking, score, context và các bước thành phần của hybrid.

**41/41 unit test pass.** [Kiểm tra tái lập](results/repro-check.json) đã dựng database mới từ checkout Day 6 đã commit,
đối chiếu đủ 179 chunk B, thử đóng gói lại NPZ và thay telemetry với vector giữ nguyên. Lượt kiểm tra dùng lại dependency đã cài.

## Giới hạn và phần còn lại

TF-IDF đang tốt nhất trên retrieval dev. Context preview chưa phải context cuối gửi model;
chưa có kết quả grounded answer, citation, abstention hoặc tool use của agent thật.
Còn cần nối controller, thử ít nhất hai cấu hình cải thiện, chốt trên dev, đánh giá test và điền báo cáo cá nhân.
Test chưa chạy retrieval. Reviewer nhãn là Codex; học viên chưa xác nhận. Corpus này chưa đủ 100 trang để làm bonus.

[Bộ benchmark và cách chấm](benchmark/README.md) · [Starter](starter/README.md) ·
[Hướng dẫn](GUIDE.md) · [Rubric](RUBRIC.md) · [Quy tắc](RULES.md) · [Cách nộp](SUBMISSION.md)
