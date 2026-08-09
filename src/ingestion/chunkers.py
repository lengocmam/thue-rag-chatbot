"""
chunkers.py
3 chiến lược chunking cho văn bản luật, dùng để thí nghiệm so sánh
(xem README §Quyết định kỹ thuật -- không giả định chiến lược nào tốt hơn
mà không đo). Cả 3 hàm trả về list[dict] cùng một schema chunk, khác nhau
ở CÁCH cắt.

Schema chunk chung (đồng bộ với data/raw/*.meta.json ở cấp document):
{
  "chunk_id": str,
  "doc_id": str,
  "so_hieu_van_ban": str,
  "chunking_strategy": "A_dieu" | "B_khoan_context" | "C_fixed",
  "dieu": str | None,
  "khoan": str | None,
  "ten_dieu": str | None,
  "text": str,
  "so_luong_token_uoc_tinh": int,
}
"""

from dataclasses import dataclass


def _estimate_tokens(text: str) -> int:
    # Ước lượng thô: tiếng Việt có dấu, trung bình ~1.3 "từ" / token khi
    # dùng tokenizer subword. Đây chỉ để so sánh tương đối giữa các chiến
    # lược, KHÔNG dùng để tính chi phí API chính xác.
    return int(len(text.split()) * 1.3)


def chunk_strategy_A_dieu(dieu_list, doc_id: str, so_hieu_van_ban: str) -> list:
    """Chiến lược A: mỗi chunk = 1 Điều hoàn chỉnh (gộp toàn bộ khoản/điểm)."""
    chunks = []
    for d in dieu_list:
        parts = [d.text_truc_tiep.strip()] if d.text_truc_tiep.strip() else []
        for k in d.khoan_list:
            khoan_text = f"{k.so}. {k.text.strip()}"
            for p in k.diem_list:
                khoan_text += f" {p.ky_hieu}) {p.text.strip()}"
            parts.append(khoan_text)

        full_text = f"Điều {d.so}. {d.ten}\n" + "\n".join(parts)
        chunks.append({
            "chunk_id": f"{doc_id}_dieu{d.so}",
            "doc_id": doc_id,
            "so_hieu_van_ban": so_hieu_van_ban,
            "chunking_strategy": "A_dieu",
            "dieu": f"Điều {d.so}",
            "khoan": None,
            "ten_dieu": d.ten,
            "text": full_text.strip(),
            "so_luong_token_uoc_tinh": _estimate_tokens(full_text),
        })
    return chunks


def chunk_strategy_B_khoan_context(dieu_list, doc_id: str, so_hieu_van_ban: str) -> list:
    """
    Chiến lược B: mỗi chunk = 1 Khoản, nhưng LUÔN kèm câu ngữ cảnh
    "Điều X quy định về [tên Điều]" ở đầu -- để retrieval không mất ngữ
    cảnh cha khi chỉ khớp đúng 1 khoản nhỏ.
    Điều nào không chia khoản (VD Điều 2) thì tự thành 1 chunk = cả Điều,
    dùng chung logic với chiến lược A cho trường hợp đó.
    """
    chunks = []
    for d in dieu_list:
        if not d.khoan_list:
            full_text = f"Điều {d.so}. {d.ten}\n{d.text_truc_tiep.strip()}"
            chunks.append({
                "chunk_id": f"{doc_id}_dieu{d.so}_full",
                "doc_id": doc_id,
                "so_hieu_van_ban": so_hieu_van_ban,
                "chunking_strategy": "B_khoan_context",
                "dieu": f"Điều {d.so}",
                "khoan": None,
                "ten_dieu": d.ten,
                "text": full_text.strip(),
                "so_luong_token_uoc_tinh": _estimate_tokens(full_text),
            })
            continue

        for k in d.khoan_list:
            context_prefix = f"Điều {d.so} ({d.ten}) quy định:\n"
            khoan_text = f"{k.so}. {k.text.strip()}"
            for p in k.diem_list:
                khoan_text += f"\n  {p.ky_hieu}) {p.text.strip()}"

            full_text = context_prefix + khoan_text
            chunks.append({
                "chunk_id": f"{doc_id}_dieu{d.so}_khoan{k.so}",
                "doc_id": doc_id,
                "so_hieu_van_ban": so_hieu_van_ban,
                "chunking_strategy": "B_khoan_context",
                "dieu": f"Điều {d.so}",
                "khoan": f"Khoản {k.so}",
                "ten_dieu": d.ten,
                "text": full_text.strip(),
                "so_luong_token_uoc_tinh": _estimate_tokens(full_text),
            })
    return chunks


def chunk_strategy_C_fixed(raw_text: str, doc_id: str, so_hieu_van_ban: str,
                            chunk_size_words: int = 350, overlap_words: int = 50) -> list:
    """
    Chiến lược C: fixed-size chunking theo số từ, có overlap -- dùng làm
    BASELINE để so sánh với A/B, không phải vì tin rằng nó tốt hơn.
    """
    words = raw_text.split()
    chunks = []
    start = 0
    idx = 0
    step = max(chunk_size_words - overlap_words, 1)

    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": f"{doc_id}_fixed{idx}",
            "doc_id": doc_id,
            "so_hieu_van_ban": so_hieu_van_ban,
            "chunking_strategy": "C_fixed",
            "dieu": None,
            "khoan": None,
            "ten_dieu": None,
            "text": text,
            "so_luong_token_uoc_tinh": _estimate_tokens(text),
        })
        idx += 1
        if end == len(words):
            break
        start += step

    return chunks