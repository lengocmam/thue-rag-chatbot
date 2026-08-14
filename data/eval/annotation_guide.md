# Hướng dẫn xây dựng bộ ground-truth (câu hỏi đánh giá retrieval)

## Trạng thái hiện tại
- `data/eval/ground_truth.jsonl`: 46 câu hỏi thật (có đáp án + `chunk_id` đúng)
- `data/eval/trap_questions.jsonl`: 6 câu hỏi bẫy
- **Mục tiêu khóa luận**: 100-150 câu thật + đủ câu bẫy để test hallucination

Đây mới là **khung sườn đã kiểm chứng**, chưa đạt số lượng mục tiêu. Phần này hướng dẫn cách tự mở rộng tiếp bằng đúng script `scripts/generate_ground_truth.py`.

## Nguyên tắc bắt buộc khi thêm câu hỏi mới

1. **Luôn thêm bằng hàm `gt(...)` hoặc `trap(...)` trong script**, KHÔNG sửa tay file `.jsonl` — vì 2 hàm này tự động kiểm tra `chunk_id` có tồn tại thật hay không (`assert cid in VALID_CHUNK_IDS`). Nếu gõ nhầm `chunk_id`, script sẽ báo lỗi ngay khi chạy thay vì âm thầm tạo ground-truth sai.

2. **Tra `chunk_id` đúng trước khi viết câu hỏi** — không đoán tên. Lệnh tra cứu:
   ```bash
   python3 -c "
   import json
   with open('data/processed/chunks_strategy_B_khoan_context.jsonl', encoding='utf-8') as f:
       for l in f:
           c = json.loads(l)
           print(c['chunk_id'], '|', c['so_hieu_van_ban'], '|', c['dieu'], c.get('khoan') or '')
   " > /tmp/chunk_index.txt
   ```
   Rồi tìm bằng `grep` hoặc mở file lên đọc.

3. **Đáp án (`expected_answer`) phải lấy từ NGUYÊN VĂN nội dung chunk**, không tự suy diễn hay nhớ từ nguồn khác. Kiểm tra bằng:
   ```bash
   python3 -c "
   import json
   chunks = {}
   for f in ['data/processed/chunks_strategy_A_dieu.jsonl', 'data/processed/chunks_strategy_B_khoan_context.jsonl']:
       with open(f, encoding='utf-8') as fh:
           for l in fh:
               c = json.loads(l)
               chunks[c['chunk_id']] = c['text']
   print(chunks['TÊN_CHUNK_ID_CẦN_KIỂM_TRA'])
   "
   ```

## Nguồn câu hỏi còn chưa khai thác (ưu tiên khai thác tiếp)

Dựa trên các `*.meta.json` đã viết trước đó, các mảng nội dung sau vẫn còn nhiều Điều/Khoản chưa có câu hỏi:

| Văn bản | Điều còn nhiều tiềm năng | Ghi chú |
|---|---|---|
| Luật 109/2025/QH15 | Điều 4 (22 khoản miễn thuế) | Mới khai thác 7/22 khoản — còn ~15 khoản chưa dùng |
| Luật 109/2025/QH15 | Điều 11 (giảm trừ từ thiện/nhân đạo) | Chưa có câu hỏi nào |
| Luật 109/2025/QH15 | Điều 17, 18, 19, 25, 26, 27 (nhượng quyền, thừa kế, thu nhập khác) | Chưa khai thác |
| TT 87/2026/TT-BTC | Điều 5 (thuế chứng khoán phái sinh) | Chưa có câu hỏi nào |
| TT 91/2026/TT-BTC | Điều 3, 5, 6, 9, 10, 11, 14-16, 18-23 | Còn rất nhiều Điều thủ tục chưa khai thác |
| Luật 09/2026/QH16 | Điều 3 (miễn thuế TNDN), Điều 4 (bảng thuế TTĐB) | Chưa có câu hỏi nào |

## Nguồn câu hỏi BẪY nên bổ sung thêm

Loại bẫy đã làm (6 câu): lỗi thời, hồi tố, không trả lời được, nhầm đối tượng, ngoài phạm vi, nhầm phạm vi văn bản.

**Loại bẫy CHƯA làm, nên bổ sung:**
- **Trộn nhầm 2 văn bản gần giống tên** (VD hỏi về "Nghị định 68/2026" khi bộ dữ liệu chỉ có Nghị định 141/2026 SỬA Nghị định 68/2026, không có bản gốc 68/2026) — kiểm tra hệ thống có tự nhận biết đang thiếu văn bản gốc không.
- **Câu hỏi đánh lừa số liệu tương tự** (VD nhầm ngưỡng "20 triệu đồng" — miễn trừ tính thuế trúng thưởng/thừa kế — với ngưỡng "1 tỷ đồng" — miễn thuế hộ kinh doanh — vì cả 2 đều là "ngưỡng miễn thuế" nhưng khác hẳn bản chất).
- **Câu hỏi yêu cầu tính toán** (VD "lương 25 triệu/tháng, có 1 người phụ thuộc, đóng BHXH 2 triệu thì nộp thuế bao nhiêu?") — kiểm tra hệ thống có tự vượt quá vai trò tra cứu để làm phép tính hay không (rủi ro tính sai).

## Quy trình thêm 1 câu hỏi mới (từng bước)

1. Chọn 1 Điều/Khoản chưa khai thác từ bảng trên.
2. Tra `chunk_id` đúng bằng lệnh ở mục 2.
3. Đọc nội dung `text` thật của chunk đó (mục 3).
4. Viết câu hỏi tự nhiên (như người dùng thật sẽ hỏi, không copy y nguyên câu chữ trong luật).
5. Viết `expected_answer` bám sát nguyên văn.
6. Thêm vào cuối hàm `build_ground_truth()` hoặc `build_trap_questions()` trong `scripts/generate_ground_truth.py`, đặt `id` tiếp theo (VD `GT047`, `GT048`...).
7. Chạy lại: `python scripts/generate_ground_truth.py` — nếu không có lỗi `AssertionError`, câu hỏi đã hợp lệ.

## Sau khi đủ số lượng mục tiêu

Bước tiếp theo là viết `eval/metrics.py` (Recall@K, MRR) chạy trên file `ground_truth.jsonl` này để đo retrieval — đây sẽ là bước kế tiếp trong lộ trình khóa luận.