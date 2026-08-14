"""
hybrid_retriever.py
Kết hợp BM25 (lexical) + dense embedding (semantic) bằng Reciprocal Rank
Fusion (RRF) -- KHÔNG cộng điểm số trực tiếp (vì thang điểm BM25 và cosine
similarity không cùng đơn vị, cộng trực tiếp sẽ thiên vị 1 bên).

Công thức RRF cho 1 chunk d xuất hiện ở hạng r trong danh sách kết quả:
    RRF_score(d) = sum over các retriever [ 1 / (k + rank_trong_retriever_do) ]
    (k = hằng số làm mượt, thường chọn 60 theo paper gốc Cormack et al. 2009)

Cách dùng:
    python src/retrieval/hybrid_retriever.py
"""

import pickle
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent / "indexing"))
from build_bm25_index import simple_tokenize

PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"

RRF_K = 60  # hằng số làm mượt chuẩn theo paper gốc, không phải số tuỳ chọn


def load_bm25_index(strategy: str) -> dict:
    with open(INDEX_DIR / f"bm25_{strategy}.pkl", "rb") as f:
        return pickle.load(f)


def load_vector_index(strategy: str):
    index = faiss.read_index(str(INDEX_DIR / f"faiss_{strategy}.index"))
    with open(INDEX_DIR / f"faiss_{strategy}.meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return index, meta


def bm25_search(bm25_data: dict, query: str, top_k: int) -> list:
    """Trả về list[(chunk_index, rank_bat_dau_tu_0)]."""
    tokens = simple_tokenize(query)
    scores = bm25_data["bm25_model"].get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(idx, rank) for rank, idx in enumerate(ranked)]


def dense_search(index, embed_model: SentenceTransformer, query: str, top_k: int) -> list:
    q_vec = embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    _, indices = index.search(q_vec, top_k)
    return [(int(idx), rank) for rank, idx in enumerate(indices[0]) if idx != -1]


def reciprocal_rank_fusion(*ranked_lists, k: int = RRF_K) -> dict:
    """ranked_lists: mỗi list là [(chunk_index, rank), ...] từ 1 retriever.
    Trả về dict {chunk_index: rrf_score}, CHƯA sắp xếp."""
    rrf_scores = {}
    for ranked_list in ranked_lists:
        for chunk_idx, rank in ranked_list:
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
    return rrf_scores


def hybrid_search(strategy: str, query: str, embed_model: SentenceTransformer,
                   top_k_each: int = 20, top_k_final: int = 5) -> list:
    """Chạy BM25 + dense độc lập (lấy top_k_each mỗi bên, nên rộng hơn
    top_k_final để RRF có đủ ứng viên trộn), rồi hợp nhất bằng RRF."""
    bm25_data = load_bm25_index(strategy)
    faiss_index, vec_meta = load_vector_index(strategy)
    chunks = bm25_data["chunks"]  # 2 index dùng chung 1 danh sách chunk gốc, thứ tự phải khớp

    bm25_results = bm25_search(bm25_data, query, top_k_each)
    dense_results = dense_search(faiss_index, embed_model, query, top_k_each)

    rrf_scores = reciprocal_rank_fusion(bm25_results, dense_results)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_final]

    return [
        {"chunk": chunks[idx], "rrf_score": score, "chunk_index": idx}
        for idx, score in ranked
    ]


if __name__ == "__main__":
    print("Đang tải model embedding...")
    model = SentenceTransformer("keepitreal/vietnamese-sbert")

    cau_hoi = "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh không phải nộp thuế thu nhập cá nhân?"
    results = hybrid_search("B_khoan_context", cau_hoi, model, top_k_final=5)

    print(f"\nCâu hỏi: {cau_hoi}\n")
    for rank, r in enumerate(results, 1):
        c = r["chunk"]
        print(f"#{rank} (RRF={r['rrf_score']:.4f}) [{c['chunk_id']}] "
              f"{c.get('so_hieu_van_ban')} - {c.get('dieu')} {c.get('khoan') or ''}")
        print(f"    {c['text'][:150].strip()}...")