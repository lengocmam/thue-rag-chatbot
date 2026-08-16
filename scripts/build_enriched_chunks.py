"""
build_enriched_chunks.py
Thí nghiệm "Contextual Retrieval": thêm 1 câu ngữ cảnh cấp văn bản vào ĐẦU
MỖI chunk trước khi index, rồi đo lại xem retrieval có cải thiện không.

NGUYÊN TẮC QUAN TRỌNG: áp dụng ĐỒNG LOẠT cho tất cả chunk của tất cả văn
bản, KHÔNG chỉ sửa riêng chunk đang gặp vấn đề (nd_141_2026_ndcp_dieu1_khoan1).
Nếu chỉ sửa đúng 1 chunk đã biết trước là "khó" để nó lọt vào top-k của
đúng 1 câu hỏi test đã biết trước, đó là OVERFITTING vào bộ test -- không
phản ánh cải thiện thật của phương pháp. Áp dụng đồng loạt mới là thí
nghiệm hợp lệ về mặt khoa học.

Câu ngữ cảnh thêm vào lấy từ metadata cấp văn bản (tên đầy đủ văn bản, đã
có sẵn từ bước viết *.meta.json trước đó) -- không tự chèn từ khóa trùng
với câu hỏi test cụ thể nào.

Cách chạy:
    python scripts/build_enriched_chunks.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_IN = PROJECT_ROOT / "data" / "processed" / "chunks_strategy_B_khoan_context.jsonl"
CHUNKS_OUT = PROJECT_ROOT / "data" / "processed" / "chunks_strategy_B_enriched.jsonl"

# Ngữ cảnh cấp văn bản -- lấy nguyên văn "ten_van_ban" đã ghi trong các
# file *.meta.json ở bước ingest trước đó, KHÔNG tự soạn mới để tránh chèn
# từ khóa thiên vị cho bất kỳ câu hỏi cụ thể nào.
DOC_CONTEXT = {
    "luat_109_2025_qh15": "Luật Thuế thu nhập cá nhân",
    "tt_87_2026_ttbtc": (
        "Thông tư quy định chi tiết một số điều của Luật Thuế thu nhập cá "
        "nhân và Nghị định số 253/2026/NĐ-CP"
    ),
    "luat_09_2026_qh16": (
        "Luật sửa đổi, bổ sung một số điều của Luật Thuế thu nhập cá nhân, "
        "Luật Thuế giá trị gia tăng, Luật Thuế thu nhập doanh nghiệp và "
        "Luật Thuế tiêu thụ đặc biệt"
    ),
    "nd_141_2026_ndcp": (
        "Nghị định sửa đổi, bổ sung Nghị định số 68/2026/NĐ-CP quy định về "
        "chính sách thuế đối với hộ kinh doanh, cá nhân kinh doanh và Nghị "
        "định số 320/2025/NĐ-CP về thuế thu nhập doanh nghiệp"
    ),
    "tt_91_2026_ttbtc": (
        "Thông tư quy định về hóa đơn điện tử, chứng từ điện tử theo Luật "
        "Quản lý thuế"
    ),
}


def enrich_chunk(chunk: dict) -> dict:
    context = DOC_CONTEXT.get(chunk["doc_id"])
    if not context:
        return chunk  # không có ngữ cảnh cho doc_id lạ -- giữ nguyên
    enriched = dict(chunk)
    enriched["text"] = f"[{context}]\n{chunk['text']}"
    enriched["chunk_id"] = chunk["chunk_id"]  # giữ nguyên chunk_id để đối chiếu ground-truth vẫn dùng được
    enriched["_enriched"] = True
    return enriched


def main():
    chunks = []
    with open(CHUNKS_IN, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Nạp {len(chunks)} chunk gốc từ {CHUNKS_IN.name}")

    enriched_chunks = [enrich_chunk(c) for c in chunks]
    n_enriched = sum(1 for c in enriched_chunks if c.get("_enriched"))
    print(f"Đã thêm ngữ cảnh cho {n_enriched}/{len(chunks)} chunk "
          f"(số còn lại không khớp DOC_CONTEXT, giữ nguyên).")

    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        for c in enriched_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Đã lưu {CHUNKS_OUT}")

    # In thử 1 ví dụ để kiểm tra bằng mắt
    example = next(c for c in enriched_chunks if c["chunk_id"] == "nd_141_2026_ndcp_dieu1_khoan1")
    print("\n--- Ví dụ chunk đã enrich (nd_141_2026_ndcp_dieu1_khoan1) ---")
    print(example["text"][:250])


if __name__ == "__main__":
    main()