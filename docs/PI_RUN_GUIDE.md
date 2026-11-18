# Hướng dẫn đo trên Raspberry Pi 4 (thí nghiệm G1 + G2)

Mục tiêu:
- **G2:** đo lại các mô hình trong bài bằng script mới. Script này ghi được peak RSS thật, đo end-to-end, và đo riêng 1 luồng và 4 luồng.
- **G1:** kiểm chứng giả thuyết "unroll gây chậm" bằng cách so sánh scan tuần tự với chunked scan, cùng các nhánh identity và conv-only.

Không cần GPU và không cần huấn luyện lại.

## 0. Repo git CHƯA chứa những thứ sau (cố ý, do `.gitignore`)

| Thiếu | Lý do | Lấy ở đâu |
|---|---|---|
| Trọng số đã huấn luyện `best.pt` của Arm 1, 2, 3, 4 | `*.pt` bị ignore | Thư mục `runs/` trên máy đã huấn luyện, ví dụ `runs/detect/runs/two_stage_protocol_study/arm3_stage2_finetune/weights/best.pt` |
| File `.onnx` | `*.onnx` bị ignore | Tự xuất ở bước 1 |
| 100 ảnh val (`Warehouse_015`) | `data/` bị ignore | `data/smartspaces_micro/images/val/` trên máy đã huấn luyện. Hoặc tạo lại bằng `scripts/build_micro_dataset.py` (cần đăng nhập Hugging Face) |

**Kiểm tra SHA-256 của `best.pt`** trước khi xuất. Đây phải là đúng các checkpoint đã dùng để tạo số liệu trong bài.

## 1. Trên PC: xuất ONNX (cần torch + ultralytics)

```bash
pip install "ultralytics==8.4.185" onnx onnxslim onnxruntime
mkdir -p onnx
```

**G2: mô hình đã huấn luyện, scan tuần tự** (cùng dạng với số liệu hiện có trong bài)
```bash
python scripts/export_onnx.py --arm baseline --weights <arm1>/weights/best.pt --out onnx/g2_arm1_baseline.onnx
python scripts/export_onnx.py --arm ss2d     --weights <arm2>/weights/best.pt --out onnx/g2_arm2_ss2d_seq.onnx
python scripts/export_onnx.py --arm case_c   --weights <arm4>/weights/best.pt --out onnx/g2_arm4_casec_seq.onnx
```

**G1: can thiệp. Cùng trọng số Arm 2, chỉ đổi cách viết vòng lặp**
```bash
python scripts/export_onnx.py --arm ss2d --weights <arm2>/weights/best.pt --chunk 20 --out onnx/g1_arm2_ss2d_chunk20.onnx
python scripts/compare_onnx_outputs.py onnx/g2_arm2_ss2d_seq.onnx onnx/g1_arm2_ss2d_chunk20.onnx
```

`compare_onnx_outputs.py` phải in ra `[PASS]`. Nếu ra `[FAIL]`, **dừng lại**: khi đó chunked scan không tương đương với scan tuần tự, nên không được dùng nó để so sánh.

**G1: ablation runtime với trọng số ngẫu nhiên** (latency, số nút và bộ nhớ không phụ thuộc trọng số; KHÔNG báo cáo độ chính xác của các mô hình này)
```bash
python scripts/export_onnx.py --arm identity --untrained --out onnx/g1_identity_untrained.onnx
python scripts/export_onnx.py --arm convonly --untrained --out onnx/g1_convonly_untrained.onnx
```

Mỗi lệnh in ra số nút ONNX và SHA-256; hãy lưu lại phần log này. Ghi chú:
- Xuất bản scan tuần tự có thể mất khoảng 3 phút.
- Chunked scan **chưa từng được xuất thử**, vì máy viết code không có torch. Nếu bị lỗi export, gửi lại log.

## 2. Chép sang Pi

```bash
scp -r onnx/ pi@<ip>:~/onnx
scp -r data/smartspaces_micro/images/val pi@<ip>:~/val_images
```

## 3. Trên Pi: cài đặt và chạy

```bash
git clone https://github.com/h4nsx/nckh.git && cd nckh
git checkout revision/priority-0-text-fixes      # hoặc: git pull nếu đã clone
python3 -m venv ~/venv-edge && source ~/venv-edge/bin/activate
pip install -r requirements-pi.txt
bash scripts/run_pi_benchmarks.sh ~/onnx ~/val_images
```

Trước khi chạy:
- Gắn tản nhiệt/quạt, cắm nguồn chuẩn 5V/3A, tắt các chương trình khác.
- Không chạy qua SSH đang có nhiều tải mạng.
- Script tự ghi nhiệt độ và trạng thái throttling. Nếu `get_throttled` khác `0x0`, kết quả bị ảnh hưởng.

**Thời gian ước tính:** khoảng 2–4 giờ cho 6 mô hình. Mỗi mô hình chạy 1 luồng và 4 luồng, mỗi lần 100 lượt forward cộng 100 ảnh end-to-end, nghỉ 30 giây giữa các lần.

## 4. Gửi kết quả về

Kết quả nằm trong `results/edge/<hostname>_<ngày>/`:
- `system_info.txt`: thông tin Pi, hệ điều hành, governor, phiên bản thư viện, SHA-256 các file ONNX.
- `*_t1.json`, `*_t4.json`: latency forward và end-to-end, `peak_rss_mib`, số nút.
- `profiles/*_profile.json`: thời gian theo từng loại toán tử, số nút trước và sau khi ORT tối ưu, log profiler ORT.
- `thermal_log.txt`: nhiệt độ và throttling sau mỗi lần chạy.

```bash
git add -f results/edge && git commit -m "Pi 4 edge measurements" && git push
```

(Hoặc nén thư mục `results/edge` và gửi qua đường khác.)

## 5. Những gì sẽ được đưa vào bài sau khi có số

| Số đo | Dùng để |
|---|---|
| `peak_rss_mib` (4 luồng) của Arm 1, Arm 2, Case C | Thay cột RSS ở Bảng VIII; trả lời dứt điểm câu hỏi "fits 1 GiB" (D2) |
| Latency và số nút: `g2_arm2_ss2d_seq` so với `g1_arm2_ss2d_chunk20` | Kiểm chứng giả thuyết unroll (D1) |
| identity / conv-only so với baseline | Cho biết riêng phần tích chập của khối mới tốn bao nhiêu |
| 1 luồng so với 4 luồng | Trả lời câu hỏi overhead dispatch thay đổi theo số luồng |
| `end_to_end` | Latency thực tế, gồm cả tiền xử lý và NMS |

Mọi con số đưa vào bài sẽ lấy trực tiếp từ các file JSON này, không nhập tay.
