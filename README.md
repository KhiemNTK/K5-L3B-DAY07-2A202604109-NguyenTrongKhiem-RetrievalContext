# Day 7 — Agentic RAG

Agent truy xuất dữ liệu Day 6 và trả lời qua OpenRouter tool calling. Đã so sánh ba phương pháp retrieval,
chạy ba cấu hình agent trên toàn bộ dev, chọn cấu hình và đánh giá baseline/selected trên cùng test.
Kết quả, phân tích lỗi và giới hạn ở [báo cáo cá nhân](report/individual/2A202604109_NguyenTrongKhiem.md).

`results/retrieval/` lưu ranking; `results/agent/` lưu trace, câu trả lời và nhãn của từng lượt agent;
`results/experiments/` lưu cấu hình đã chọn trước test.

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

**65/65 unit test pass.** [Kiểm tra tái lập](results/repro-check.json) đã dựng database mới từ checkout Day 6 đã commit,
đối chiếu đủ 179 chunk B, thử đóng gói lại NPZ và thay telemetry với vector giữ nguyên. Lượt kiểm tra dùng lại dependency đã cài.

## Agent và context

[Cấu hình ban đầu](config/agent.json): `openai/gpt-4o-mini`, TF-IDF, tối đa 3 search và 4 model call/câu,
top-3, context 2.400 ký tự, timeout 45 giây mỗi request. Sau so sánh agent dev, giữ cấu hình này vì đạt tiêu chí chọn cao nhất.
Thêm `OPENROUTER_API_KEY` vào `.env`, rồi chạy:

```powershell
& $labPython -X utf8 -B starter/run_agent.py --day6-root $day6Root --case D01 --case D16 --case D08
& $labPython -X utf8 -B starter/run_agent.py --day6-root $day6Root --method dense --case D01
```

Bỏ `--case` để chạy toàn bộ dev. `--method` nhận dense, tfidf hoặc hybrid cho smoke dev.
[Runner thí nghiệm](starter/README.md#1-kết-nối-agent) nhận ba cấu hình dev khai báo sẵn; test yêu cầu selection đã khóa.
Model phải gọi search trước khi trả lời. Mỗi native tool call có kết quả đúng `tool_call_id`;
batch hợp lệ được thực hiện tuần tự trong giới hạn search. Chỉ context tích lũy mới nhất được gửi model.

Context ưu tiên lượt tìm mới, deduplicate theo ID, giữ nguyên đoạn và nguồn/trang. Chunk không vừa budget
bị bỏ, kể cả evidence cũ; ID bị bỏ được lưu trong trace. Citation phải thuộc context cuối.
Adapter có thể gắn marker từ danh sách ID model chọn nếu thiếu marker nội tuyến; raw response và thao tác định dạng được lưu riêng.
Lỗi model/tool được ghi là lỗi, không thay bằng dữ liệu giả. Citation hợp lệ về ID vẫn cần kiểm tra nội dung nguồn.

## So sánh agent và test

Mỗi cấu hình chạy đủ 17 câu dev: baseline strict pass **15/17**, context 800 ký tự **12/17**,
tối đa hai search **14/17**. Tiêu chí chọn và giả thuyết được khai báo trước; lỗi giữ trong mẫu số strict pass.
Giữ baseline, [khóa cấu hình/code/dữ liệu](results/experiments/20261116T043650385003Z/selection.json), rồi chạy đủ bảy câu test cho cả hai nhãn:

| Lượt test | Hit@1 gốc | Evidence Hit@3 gốc | Context đủ evidence | Grounded trung bình | Từ chối đúng | Từ chối nhầm |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 1,6/2 | 2/2 (100%) | 1/5 (20%) |
| Selected = baseline | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 1,6/2 | 2/2 (100%) | 1/5 (20%) |

Cả hai lượt test không lỗi, citation hợp lệ 4/4, tool use 7/7 và strict pass 6/7. T02 bị bỏ lỡ đoạn README ở trang 51.
Đây là hai lượt cùng cấu hình; không kết luận có cải thiện trên test. Tổng test dùng 40 model call, 30.897 token,
API báo 0,00491550 USD; trace và nhãn chấm được dẫn trong báo cáo.

[Kết quả chi tiết](report/individual/2A202604109_NguyenTrongKhiem.md) có nhãn context/answer/citation, trace, thời gian, chi phí và lỗi dev.
Review và metrics được đối chiếu trực tiếp từ trace; sửa nhãn context D02 không đổi lựa chọn baseline hoặc kết quả test.
Selection giữ thời điểm chốt trước test, kiểm tra dữ liệu, config, trace và mã suy luận. Usage thiếu để `null`.

## Giới hạn và phần còn lại

Corpus này chưa đủ 100 trang để làm bonus.
Các thí nghiệm dev có lỗi mạng và chạy chồng thời gian; chi phí đầy đủ của request thiếu usage không xác định.
Chưa kiểm tra cài dependency/tải model từ đầu trên máy mới.

[Bộ benchmark và cách chấm](benchmark/README.md) · [Starter](starter/README.md) ·
[Hướng dẫn](GUIDE.md) · [Rubric](RUBRIC.md) · [Quy tắc](RULES.md) · [Cách nộp](SUBMISSION.md)
