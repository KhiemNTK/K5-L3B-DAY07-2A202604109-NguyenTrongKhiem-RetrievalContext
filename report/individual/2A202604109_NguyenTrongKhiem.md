# Báo cáo Day 7 — Retrieval và Context Engineering

**Nguyễn Trọng Khiêm — 2A202604109 — K5-L3B**

## Dữ liệu và benchmark

| Thành phần         | Cấu hình                                                                     |
| ------------------ | ---------------------------------------------------------------------------- |
| Nguồn              | Git & GitHub Fundamentals, Student Edition, Hitesh Patel, Edition 1, 10/2026 |
| Corpus             | 53 trang nội dung, trang vật lý 5–57 trong PDF 65 trang                      |
| Chunk              | B: 179 đoạn, tối đa 500 ký tự, không overlap; chọn trên dev Day 6            |
| Embedding          | `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, chuẩn hóa L2            |
| Revision embedding | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`                                   |
| Vector store       | PostgreSQL 16 + pgvector 0.8.7, cosine exact search; adapter chỉ đọc         |
| Agent              | OpenRouter `openai/gpt-4o-mini`, temperature 0; provider ghi nhận là OpenAI  |
| Dev / test         | Dev 15 có đáp án + 2 thiếu evidence; test 5 + 2                              |

[PDF nguồn](https://sbccas.github.io/git-github-fundamentals/books/git-github-fundamentals-student-edition.pdf)
ghi CC BY-SA 4.0 ở trang 2. Text trích xuất được chọn chương, chuẩn hóa và bỏ footer, chia sẻ cùng giấy phép.
[Manifest](../../data/day6.json) ghi nguồn, model và checksum; [chunk](../../data/chunks.json) giữ ID, text, nguồn và trang.

D01–D15 dùng lại dev Day 6, thuộc Unit 1–4. T01–T05 là chủ đề mới ở Unit 5, không lấy lại test Day 6.
[Benchmark](../../benchmark/README.md) ghi gold và kiểm tra thiếu evidence trên toàn bộ 53 trang.
Filter Unit chỉ áp dụng khi câu hỏi nêu rõ; không gửi gold vào model/retriever.

## Retrieval và thử nghiệm

Ba phương pháp dùng cùng corpus, query gốc và top-3. Hybrid gộp 10 candidate mỗi thành phần bằng
RRF `1 / (60 + rank)`. Context preview dùng 2.400 ký tự, gồm header ID/nguồn/trang.

| Phương pháp / dev |          Hit@1 | Evidence Hit@3 | Context preview có evidence | Median/query |
| ----------------- | -------------: | -------------: | --------------------------: | -----------: |
| Dense             |    12/15 (80%) | 13/15 (86,67%) |              13/15 (86,67%) |     31,65 ms |
| TF-IDF            | 13/15 (86,67%) |   15/15 (100%) |                15/15 (100%) |      0,41 ms |
| Hybrid RRF        | 13/15 (86,67%) | 14/15 (93,33%) |              14/15 (93,33%) |     25,02 ms |

[Kết quả retrieval](../../results/retrieval/20261116T025510104675Z-dev/summary.json): mỗi phương pháp chạy đủ 17 câu, không lỗi, API 0 USD.
Hit chỉ tính 15 câu có đáp án: đúng nguồn/trang và chứa trọn một quote gold thay thế. Dense/hybrid tính cả embedding query và kết nối DB; setup ghi riêng.

Giữ TF-IDF, model, prompt, top-k và corpus để thử ba cấu hình agent trên đủ 17 câu dev:

| Cấu hình    | Context, ký tự gồm header | Tối đa search | Giả thuyết                                        |
| ----------- | ------------------------: | ------------: | ------------------------------------------------- |
| Baseline    |                     2.400 |             3 | Mốc so sánh                                       |
| Context 800 |                       800 |             3 | Giảm token/citation thừa, có nguy cơ mất evidence |
| Hai search  |                     2.400 |             2 | Giảm tìm lặp, có nguy cơ dừng quá sớm             |

Chọn strict pass cao nhất theo [kế hoạch](../../config/experiments.json): answer/context đều mức 2 và mọi citation hợp lệ,
hoặc từ chối đúng câu thiếu evidence; lỗi không đạt. Đây là tiêu chí chọn cấu hình của bài làm.

| Cấu hình / dev |      Hit@1 gốc | Evidence Hit@3 gốc | Context Evidence Rate |    Strict pass | Context mức 0/1/2 |
| -------------- | -------------: | -----------------: | --------------------: | -------------: | ----------------: |
| Baseline       | 13/15 (86,67%) |       15/15 (100%) |        14/15 (93,33%) | 15/17 (88,24%) |            0/1/14 |
| Context 800    | 13/15 (86,67%) |       15/15 (100%) |        11/15 (73,33%) | 12/17 (70,59%) |            1/3/11 |
| Hai search     | 13/15 (86,67%) |       15/15 (100%) |        13/15 (86,67%) | 14/17 (82,35%) |            2/0/13 |

Ranking gốc giống nhau; context cuối phụ thuộc query viết lại, budget và lỗi. Context mức 0/1/2 = không có/thiếu/đủ evidence và nguồn;
vẫn chấm context đã lấy được nếu model lỗi sau retrieval.

### Phân tích lỗi dev và lựa chọn

- **D09, baseline:** context trang 30–31 nói `main` không có commit mới, nhưng answer đổi thành nhánh feature.
  Answer mức 1; citation không hỗ trợ điều kiện sai.
- **D10, context 800:** chỉ giữ trang 32, bỏ đoạn trang 33 về xóa ba dòng marker; context mức 1, từ chối nhầm.

Giữ **baseline** vì strict pass 15/17 cao nhất. Context 800 giảm token đã báo nhưng mất evidence;
hai search giảm lượt gọi quan sát được nhưng có hai lỗi mạng và D05 thêm lợi ích cộng tác chưa có trong context.

## Agent và đánh giá

Baseline gọi `search` trước khi trả lời, tối đa ba search/bốn model call, timeout 45 giây/request.
Context ưu tiên lượt tìm mới, loại ID trùng, giữ nguyên chunk và bỏ đoạn quá budget. Citation phải thuộc context cuối và hỗ trợ nội dung;
trace giữ tool call, raw response và lỗi, không thay lỗi bằng mock.

| Cấu hình / split           | Tool use, câu không lỗi | Grounded 0/1/2, không lỗi; TB /2 | Citation hợp lệ | Answer có citation | Từ chối đúng, toàn bộ mẫu | Từ chối nhầm, toàn bộ mẫu |
| -------------------------- | ----------------------: | -------------------------------- | --------------: | -----------------: | ------------------------: | ------------------------: |
| Baseline / dev             |            16/16 (100%) | 0/1/13; 1,929                    |  14/15 (93,33%) |       14/14 (100%) |                2/2 (100%) |                 0/15 (0%) |
| Context 800 / dev          |            16/16 (100%) | 3/1/11; 1,533                    |  11/12 (91,67%) |       12/12 (100%) |                 1/2 (50%) |                3/15 (20%) |
| Hai search / dev           |            15/15 (100%) | 0/1/12; 1,923                    |  14/15 (93,33%) |       13/13 (100%) |                2/2 (100%) |                 0/15 (0%) |
| Baseline / test            |              7/7 (100%) | 1/0/4; 1,600                     |      4/4 (100%) |         4/4 (100%) |                2/2 (100%) |                 1/5 (20%) |
| Selected = baseline / test |              7/7 (100%) | 1/0/4; 1,600                     |      4/4 (100%) |         4/4 (100%) |                2/2 (100%) |                 1/5 (20%) |

Grounded: 0 = sai/thiếu evidence, 1 = đúng một phần, 2 = đúng và đủ; từ chối câu có đáp án chấm 0.
Dev có bốn lỗi mạng: baseline D02 sau search; context 800 D16 trước search; hai search D01/D11 trước search.
Trung bình grounded gồm lỗi lần lượt là 1,800; 1,533; 1,667 trên /2.
Metrics gốc còn báo abstention/từ chối nhầm trên tập không lỗi: baseline 2/2 và 0/14;
context 800 là 1/1 và 3/15; hai search là 2/2 và 0/13. Bảng dùng mẫu số đầy đủ để thể hiện tác động của lỗi.

### Test sau khi khóa

| Lượt test           | Hit@1 gốc | Evidence Hit@3 gốc | Context Evidence Rate | Context mức 0/1/2 |  Strict pass | Lỗi |
| ------------------- | --------: | -----------------: | --------------------: | ----------------: | -----------: | --: |
| Baseline            | 3/5 (60%) |          4/5 (80%) |             4/5 (80%) |             1/0/4 | 6/7 (85,71%) | 0/7 |
| Selected = baseline | 3/5 (60%) |          4/5 (80%) |             4/5 (80%) |             1/0/4 | 6/7 (85,71%) | 0/7 |

Review đủ 14 kết quả (5 có đáp án + 2 thiếu evidence mỗi lượt), cùng nhãn như sau:

| ID  | Context / answer | Citation         | Lý do                                                                                             |
| --- | ---------------- | ---------------- | ------------------------------------------------------------------------------------------------- |
| T01 | 2 / 2            | Trang 50, hợp lệ | Đủ khác biệt Star: bookmark/quan tâm; Watch: thông báo hoạt động                                  |
| T02 | 0 / 0            | Không có         | Ba query viết lại bỏ lỡ điều kiện checkbox README trang 51; từ chối nhầm                          |
| T03 | 2 / 2            | Trang 51, hợp lệ | Đủ owners và admins                                                                               |
| T04 | 2 / 2            | Trang 52, hợp lệ | Đủ title, bước tái hiện, expected/actual, ảnh/lỗi nếu có; phần workflow thêm là dư nhưng có nguồn |
| T05 | 2 / 2            | Trang 55, hợp lệ | Đúng `git pull upstream main` hoặc Sync fork                                                      |
| T06 | Không áp dụng    | Không có         | Từ chối, không bịa SLA; kiểm tra toàn bộ trang 5–57 không có support/ticket/response/SLA          |
| T07 | Không áp dụng    | Không có         | Từ chối, không bịa giá; không có giá gói Team, các `price` khác thuộc ví dụ quán ăn               |

### Thời gian và chi phí

| Cấu hình / split           | Search / model call | Số câu theo 0/1/2/3 search | Token có usage | Tổng thời gian câu / median | API đã báo, USD |
| -------------------------- | ------------------: | ------------------------: | -------------: | --------------------------: | --------------: |
| Baseline / dev             |             26 / 40 |                  0/11/3/3 |         29.000 |          465,91 s / 26,37 s |      0,00521175 |
| Context 800 / dev          |             25 / 39 |                  1/10/3/3 |         19.876 |          499,80 s / 31,11 s |      0,00377790 |
| Hai search / dev           |             20 / 35 |                  2/10/5/0 |         24.140 |          449,63 s / 26,98 s |      0,00445125 |
| Baseline / test            |             13 / 20 |                   0/4/0/3 |         15.489 |           122,27 s / 9,74 s |      0,00261720 |
| Selected = baseline / test |             13 / 20 |                   0/4/0/3 |         15.408 |          145,99 s / 19,19 s |      0,00229830 |

Phân bố search tính cả câu lỗi; model call tính cả request lỗi. Thời gian không gồm setup.
Dev thiếu usage ở request lỗi nên token/chi phí chỉ là phần đã báo; tổng đầy đủ chưa xác định.
Các lượt chạy chồng thời gian; test selected có 4.096 prompt token được cache, baseline không có.
Hai lượt test dùng cùng cấu hình; chênh lệch thời gian/chi phí chưa chứng minh cải thiện.

### Bằng chứng và tái lập

- Dev: [baseline](../../results/agent/20261116T042530099605Z-dev/summary.json), [context 800](../../results/agent/20261116T042753175840Z-dev/summary.json), [hai search](../../results/agent/20261116T042806131267Z-dev/summary.json).
- Test baseline: [trace](../../results/agent/20261116T043711225775Z-test/traces.json), [review](../../results/agent/20261116T043711225775Z-test/reviews.json), [metrics](../../results/agent/20261116T043711225775Z-test/metrics.json).
- Test selected: [trace](../../results/agent/20261116T043708790355Z-test/traces.json), [review](../../results/agent/20261116T043708790355Z-test/reviews.json), [metrics](../../results/agent/20261116T043708790355Z-test/metrics.json).
- Mỗi thư mục dev có `traces.json`, `reviews.json`, `metrics.json`; review ghi người chấm và lý do từng citation.
- [Thiết lập](../../docs/README.md), [lệnh chạy](../../starter/README.md), [kiểm tra database mới](../../results/repro-check.json).

Sau đối chiếu nguồn, context D02 baseline sửa mức 0 thành 1; không đổi CER, strict pass, cấu hình chọn hoặc kết quả test.
Trace/usage và selection trước test giữ nguyên. Kiểm tra offline **65/65 pass**: `python -X utf8 -B -m unittest discover -s tests -v`.