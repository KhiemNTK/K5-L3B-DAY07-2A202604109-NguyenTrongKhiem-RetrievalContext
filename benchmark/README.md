# Benchmark cá nhân Day 7

| Split | Có đáp án | Thiếu evidence | Tổng |
|---|---:|---:|---:|
| [Dev](dev.json) | 15 | 2 | 17 |
| [Test](test.json) | 5 | 2 | 7 |

Dùng corpus B gồm 179 chunk, 53 trang vật lý (5–57), theo [manifest](../data/day6.json).
Câu hỏi bằng tiếng Anh, cùng ngôn ngữ với tài liệu. Nguồn và attribution CC BY-SA 4.0 nằm trong manifest.

## Nguồn và split

D01–D15 dùng lại câu dev Day 6 thuộc Unit 1–4. T01–T05 là câu mới thuộc Unit 5, với chủ đề đáp án khác
câu dev và test Day 6. Bốn câu thiếu evidence có lý do, từ khóa tìm trên toàn corpus và các trang đã đọc.
Kết luận thiếu thông tin chỉ áp dụng cho 53 trang được index.

Đã đối chiếu 20 đáp án với text PDF và xem ảnh trang 31, 44, 51 để kiểm tra cột/bảng.
Nhãn, lý do và bằng chứng đánh giá được dẫn trong [báo cáo](../report/individual/2A202604109_NguyenTrongKhiem.md#agent-và-đánh-giá).
Filter Unit phải được nêu rõ trong câu hỏi và áp dụng giống nhau cho mọi phương pháp.
Gold chỉ dùng để chấm sau retrieval; không có câu multi-hop trong bộ này.

## Cách chấm

Hit@1 và Evidence Hit@3 chỉ tính câu có đáp án: đúng nguồn, trang và chứa đủ một quote gold.
Các phần tử gold là evidence thay thế. D09 có alternative trang 31, D14 có alternative trang 46.
Trường `region` của D09 chỉ cột trái; cần xem lại PDF khi chấm câu trả lời vì text bị xen hai cột.

Context preview dùng budget 2.400 ký tự, gồm ID/nguồn/trang. Đây chưa phải context cuối của agent.
Câu thiếu evidence vẫn được truy xuất và lưu trace, nhưng không tính retrieval hit hoặc suy ra abstention.
Lỗi được giữ trong trace; lỗi câu có đáp án vẫn nằm trong mẫu số.

## Chạy và tái lập

[plan.json](plan.json) ghi cấu hình; [lock.json](lock.json) khóa input.
Runner dev chỉ đọc file dev; validator kiểm tra cấu trúc cả hai split mà không chạy retrieval trên test.
Lệnh thiết lập và chạy ở [README](../README.md#cách-chạy).
Mỗi lượt tạo `results/retrieval/<UTC>-dev/`, gồm ba trace và summary có checksum code/input.

So sánh agent có ba lượt dev đủ 17 câu và hai lượt test đủ bảy câu sau khi khóa selection;
xem [báo cáo và nhãn nguồn](../report/individual/2A202604109_NguyenTrongKhiem.md). Ranking query gốc được báo riêng với các query model viết lại.
Những lượt này không đổi câu hỏi, gold, split hay filter đã khóa.

## Lịch sử nhãn

Revision 2 bổ sung evidence D09 ở bảng trang 31 sau khi kiểm tra ảnh PDF. Không đổi câu hỏi, split, test hoặc
tham số retrieval. Chỉ giữ [nhãn dev trước sửa](../results/retrieval/20261115T161027146771Z-dev/inputs/dev.json)
để đối chiếu thay đổi D09 với [nhãn hiện tại](dev.json).
