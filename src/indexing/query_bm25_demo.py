"""
query_bm25_demo.py
Script thử nghiệm nhanh: truy vấn BM25 index vừa dựng, in top-K kết quả.
Dùng để KIỂM TRA index hoạt động đúng trước khi viết eval chính thức.

Cách chạy:
    python src/indexing/query_bm25_demo.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_bm25_index import simple_tokenize

INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "index"


def query(strategy: str, question: str, top_k: int = 3):
    with open(INDEX_DIR / f"bm25_{strategy}.pkl", "rb") as f:
        index_data = pickle.load(f)

    bm25 = index_data["bm25_model"]
    chunks = index_data["chunks"]

    tokens = simple_tokenize(question)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    print(f"\nCâu hỏi: {question}")
    print(f"Chiến lược: {strategy}")
    for rank, idx in enumerate(top_indices, 1):
        c = chunks[idx]
        print(f"  #{rank} (score={scores[idx]:.2f}) [{c['chunk_id']}] "
              f"{c.get('so_hieu_van_ban')} - {c.get('dieu')} {c.get('khoan') or ''}")
        print(f"      {c['text'][:150].strip()}...")


if __name__ == "__main__":
    # Câu hỏi bẫy đã biết trước: câu trả lời ĐÚNG nằm ở Nghị định 141/2026,
    # KHÔNG nằm ở Luật 109/2025 (vốn ghi "500 triệu đồng" đã lỗi thời) hay
    # Luật 09/2026 (chỉ nói "Chính phủ quy định" không nêu số) -- test xem
    # BM25 lexical thô có tìm đúng văn bản chứa câu trả lời hiện hành không.
    cau_hoi_test = [
        "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh không phải nộp thuế thu nhập cá nhân?",
        "Mức giảm trừ gia cảnh cho người phụ thuộc là bao nhiêu?",
        "Thuế suất chuyển nhượng chứng khoán là bao nhiêu phần trăm?",
    ]
    for q in cau_hoi_test:
        query("B_khoan_context", q, top_k=3)
        print("-" * 70)