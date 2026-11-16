# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Vòng lặp bounded search, chọn context và kiểm tra citation |
| `controller.py` | Adapter OpenRouter tool calling, validation response và usage |
| `run_agent.py` | CLI chạy dev hoặc test theo selection đã khóa, lưu trace và lỗi |
| `experiments.py` | So sánh dev đã chấm nguồn và khóa cấu hình trước test |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `day6.py` | Adapter dense thật, kiểm tra snapshot và tái sử dụng model/pgvector Day 6 |
| `docker-compose.yml` | Dùng lại volume Day 6 trên cổng local 15433 |
| `benchmark.py` | Kiểm tra schema, split, evidence, filter và checksum benchmark |
| `retrieval.py` | Hybrid RRF và CLI so sánh dense/TF-IDF/hybrid trên dev |

Checkout đã nối vector store, ba phương pháp retrieval và model trả lời thật.
Thiết kế và kết quả ở [báo cáo cá nhân](../report/individual/2A202604109_NguyenTrongKhiem.md). Mock chỉ dùng để test code.

Xem [cấu hình và kết quả](../README.md), [thiết lập từ đầu](../docs/README.md).
`connect_day6(Path(...))` trả `(retriever, verification)`, tương thích `run_agent`.
Chỉ filter `section` được hỗ trợ bởi dense và phải đến từ query/người dùng.

## 1. Kết nối agent

`OpenRouterController.decide` gọi model thật và chuẩn hóa thành `search`, `answer` hoặc `abstain`.
Tool calling theo [giao thức OpenRouter](https://openrouter.ai/docs/guides/features/tool-calling).
Key lấy từ biến môi trường hoặc `.env`, không ghi vào trace. Model chỉ nhận câu hỏi, lịch sử query và context có nguồn.
`run_agent.py` nối controller với retriever theo [config/agent.json](../config/agent.json).

```powershell
& $labPython -X utf8 -B starter/run_agent.py --day6-root $day6Root --case D01 --case D16 --case D08
```

Trace có raw response, action đã chuẩn hóa, query/tool-call ID, kết quả, context cuối, citation, điều kiện dừng,
thời gian và usage. Nếu lỗi xảy ra sau retrieval, context và event trước lỗi vẫn được giữ.
Output bị cắt hoặc tool arguments sai được tính là lỗi; ID citation sai dẫn đến từ chối.
Usage/cost lấy từ [response của OpenRouter](https://openrouter.ai/docs/cookbook/administration/usage-accounting), không có thì để `null`.
Các giới hạn trong config là lựa chọn của bài làm, không phải giới hạn bắt buộc của đề.

Thử nghiệm đầy đủ dùng các biến thể đã khai báo trong `config/experiments.json`:

```powershell
& $labPython -X utf8 -B starter/run_agent.py --day6-root $day6Root --experiment baseline
& $labPython -X utf8 -B starter/run_agent.py --day6-root $day6Root --experiment compact_context
& $labPython -X utf8 -B starter/run_agent.py --day6-root $day6Root --experiment two_searches
```

Đọc nguồn rồi đặt `reviews.json` vào từng thư mục run. Truyền đúng ba thư mục vừa tạo để chọn trên dev:

```powershell
& $labPython -X utf8 -B starter/experiments.py <baseline-run> <compact-run> <two-searches-run>
& $labPython -X utf8 -B starter/run_agent.py --day6-root $day6Root --split test --selection <selection.json> --variant baseline
& $labPython -X utf8 -B starter/run_agent.py --day6-root $day6Root --split test --selection <selection.json> --variant selected
```

Thay `<...>` bằng đường dẫn được in sau mỗi lệnh. Test không nhận `--case`, `--method`, `--experiment`.
Dữ liệu, config, trace và mã suy luận được kiểm tra theo selection; không sửa hệ thống theo test rồi dùng lại test cũ.
Loader đối chiếu config với summary dev đã checksum và cấu hình khai báo. Review/metrics có thể sửa khi đối chiếu nguồn,
không thay đổi thời điểm hoặc cấu hình đã chọn trong selection.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài.
TF-IDF là lexical retrieval, áp dụng filter metadata bằng phép so sánh bằng trước top-k, không thay dense retrieval.
CLI mẫu chỉ nhận câu có đáp án; với bộ đầy đủ có câu thiếu evidence, dùng [runner so sánh](../benchmark/README.md#chạy-và-tái-lập).

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

Với full run của bài này, thêm `--chunks data/chunks.json` để đo ranking câu hỏi gốc và context cuối.
Mỗi câu cần `reason`; mỗi câu có đáp án thêm `context_grade` 0–2. Ghi `citation_reasons` theo thứ tự citation,
`reviewer` và `student_confirmation` ghi người chấm/trạng thái tại thời điểm tạo review.
Giữ nguyên trace/summary của lượt chạy; ghi lý do nếu sửa nhãn sau khi đối chiếu nguồn. Ghi cả câu lỗi trong review.
Nếu lỗi sau retrieval, vẫn chấm context đã giữ lại; lỗi trước retrieval không có context được tính mức 0.
Kết quả mở rộng báo strict pass và grounded mean gồm lỗi bên cạnh các metric rubric trên câu không lỗi.

Có thể chấm lại lượt test đã lưu mà không gọi API:

```powershell
$testRun = 'results/agent/20261116T043711225775Z-test'
& $labPython -X utf8 -B starter/agent_metrics.py --chunks data/chunks.json --cases benchmark/test.json --traces "$testRun/traces.json" --reviews "$testRun/reviews.json" --split test --output .codex-log/baseline-test-metrics.json
```

Unit test tái tính metrics từ review/trace của năm lượt agent, kiểm tra cấu hình, dữ liệu và selection;
không cần API hoặc database.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Agent ưu tiên chunk từ lượt search mới; đoạn cũ được giữ nếu còn budget. `dropped_ids` ghi phần bị loại.
Citation ID hợp lệ chưa chứng minh nguồn hỗ trợ toàn bộ câu trả lời; xem review từng nguồn trước khi chấm.
