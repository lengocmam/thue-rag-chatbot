"""
diagnose_rank.py
Script chẩn đoán: cho 1 câu hỏi + 1 chunk_id "đáp án đúng" đã biết trước,
tìm xem chunk đó đứng THỨ HẠNG BAO NHIÊU trong từng retriever (BM25, dense,
hybrid) -- không cắt cụt ở top-5 như query demo, để biết "gần đúng" cỡ nào.

Cách dùng:
    python src/retrieval/diagnose_rank.py
"""

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_retriever import (
    load_bm25_index, load_vector_index, bm25_search, dense_search,
    reciprocal_rank_fusion,
)


def find_rank_of_chunk(ranked_list: list, target_chunk_index: int):
    """ranked_list: [(chunk_index, rank), ...]. Trả về rank (0-based) nếu
    tìm thấy, None nếu không nằm trong danh sách đã lấy (nghĩa là đứng
    NGOÀI top_k đã truy vấn, tức còn tệ hơn cả rank cuối cùng)."""
    for idx, rank in ranked_list:
        if idx == target_chunk_index:
            return rank
    return None


def diagnose(strategy: str, query: str, target_chunk_id: str, embed_model,
             top_k_search: int = 232):
    """top_k_search mặc định = TOÀN BỘ số chunk hiện có của chiến lược B,
    để chắc chắn tìm được thứ hạng thật, không bị cắt cụt."""
    bm25_data = load_bm25_index(strategy)
    faiss_index, _ = load_vector_index(strategy)
    chunks = bm25_data["chunks"]

    target_idx = next((i for i, c in enumerate(chunks) if c["chunk_id"] == target_chunk_id), None)
    if target_idx is None:
        print(f"!! Không tìm thấy chunk_id '{target_chunk_id}' trong {strategy}")
        return

    bm25_r = bm25_search(bm25_data, query, top_k=top_k_search)
    dense_r = dense_search(faiss_index, embed_model, query, top_k=top_k_search)
    rrf_scores = reciprocal_rank_fusion(bm25_r, dense_r)
    hybrid_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    hybrid_r = [(idx, rank) for rank, (idx, _) in enumerate(hybrid_ranked)]

    print(f"\nChunk mục tiêu: {target_chunk_id}")
    print(f"  Nội dung: {chunks[target_idx]['text'][:150].strip()}...")
    for name, ranked_list in [("BM25", bm25_r), ("Dense", dense_r), ("Hybrid (RRF)", hybrid_r)]:
        rank = find_rank_of_chunk(ranked_list, target_idx)
        if rank is not None:
            print(f"  -> {name}: hạng #{rank + 1} / {len(chunks)}")
        else:
            print(f"  -> {name}: KHÔNG XUẤT HIỆN trong top-{top_k_search}")


if __name__ == "__main__":
    print("Đang tải model embedding...")
    model = SentenceTransformer("keepitreal/vietnamese-sbert")

    cau_hoi = "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh không phải nộp thuế thu nhập cá nhân?"

    # 2 chunk "đáp án đúng" tiềm năng đã biết trước, so sánh thứ hạng cả 2:
    diagnose("B_khoan_context", cau_hoi, "nd_141_2026_ndcp_dieu1_khoan1", model)  # lệnh tìm-và-thay
    diagnose("B_khoan_context", cau_hoi, "nd_141_2026_ndcp_dieu1_khoan2", model)  # nêu rõ "01 tỷ đồng"