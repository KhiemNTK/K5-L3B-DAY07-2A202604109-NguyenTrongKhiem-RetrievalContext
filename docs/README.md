# Thiết lập và tái lập

Cần Docker Desktop đang chạy, Python 3.12 và checkout Day 6 có các artifact đã commit.
Nếu chưa có, clone [repo Day 6](https://github.com/KhiemNTK/K5-L3B-DAY06-NguyenTrongKhiem-2A202604109-DataIngestion)
ở thư mục ngang cấp với repo này. Các lệnh dưới chạy trong PowerShell từ gốc Day 7.

## Python

```powershell
$day6Root = (Resolve-Path '../K5-L3B-DAY06-NguyenTrongKhiem-2A202604109-DataIngestion').Path
python -m venv (Join-Path $day6Root 'starter/.venv')
$labPython = Join-Path $day6Root 'starter/.venv/Scripts/python.exe'
& $labPython -m pip install -r (Join-Path $day6Root 'starter/requirements.txt')
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Nếu môi trường Day 6 đã có dependency, dùng lại môi trường đó. Không cần API key cho retrieval local.

## Database mới

Compose dùng volume `day6_student_starter_pg_data`, cùng volume Day 6. Dừng container Day 6 trước;
chỉ chạy lệnh `docker stop` dưới đây nếu container đó tồn tại. Nếu đã có database đủ dữ liệu, bỏ qua bước nạp.

```powershell
docker stop day6_student_starter-postgres-1
docker volume create day6_student_starter_pg_data
docker compose -f starter/docker-compose.yml -p day7_retrieval up -d --wait
docker compose -f starter/docker-compose.yml -p day7_retrieval exec -T postgres psql -U day6 -d day6_lab -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

Nạp snapshot bằng runner Day 6. Lệnh này thay các snapshot A/B trong database local bằng artifact Day 6 đã commit,
không tính lại embedding. Nếu đổi tài khoản/cổng, sửa cùng cấu hình Compose, `.env` và URL dưới đây.

```powershell
$env:DATABASE_URL = 'postgresql://day6:day6_local_only@127.0.0.1:15433/day6_lab'
Push-Location (Join-Path $day6Root 'starter')
try { & $labPython -X utf8 -B -m src.run --split dev } finally { Pop-Location }
```

Volume giữ dữ liệu khi dừng container. Không dùng `down -v` để dừng dịch vụ.

## Model và kiểm tra

Lần đầu cần mạng để tải model đúng revision; `MyEmbedder` lưu cache ở `starter/.venv/model_cache` của Day 6.
Truy vấn dưới đây kiểm tra kết nối và tải model nếu cache chưa có, không chạy benchmark test.

```powershell
$env:HF_HUB_OFFLINE = '0'
& $labPython -X utf8 -B starter/day6.py --day6-root $day6Root --query 'How can I inspect the changes staged for the next commit?'
$env:HF_HUB_OFFLINE = '1'
& $labPython -X utf8 -B starter/benchmark.py
& $labPython -X utf8 -B -m unittest discover -s tests -v
& $labPython -X utf8 -B starter/retrieval.py --day6-root $day6Root --split dev
```

Runtime khóa corpus, model/code và nội dung vector B float32. `source_sha256` vẫn giữ checksum của artifact gốc
để đối chiếu lịch sử; số liệu thời gian, báo cáo run và metadata nén NPZ không chặn runtime.
Database phải khớp ID, text, nguồn, trang, metadata và vector của B.

[Kiểm tra database mới](../results/repro-check.json) đã chạy trên container tạm, không gắn volume bài lab.
Dependency được dùng lại từ môi trường sẵn có; cài package và tải model lần đầu chưa được kiểm tra trên máy mới.
