"""
build_vector_index.py
Dựng dense embedding index (semantic retrieval) từ 1 file chunks_strategy_*.jsonl,
dùng FAISS để lưu và tìm kiếm theo cosine similarity.

Model mặc định: 'keepitreal/vietnamese-sbert' -- sentence-transformer huấn
luyện riêng cho tiếng Việt. Có thể đổi sang model đa ngôn ngữ tổng quát hơn
(VD 'intfloat/multilingual-e5-base') để SO SÁNH -- đây chính là 1 biến thí
nghiệm của khóa luận (model tiếng Việt chuyên biệt vs đa ngôn ngữ tổng quát).

Cách chạy (build cho cả 3 chiến lược cùng lúc):
    pip install sentence-transformers faiss-cpu
    python src/indexing/build_vector_index.py

LƯU Ý: lần chạy đầu sẽ tải model (~500MB-1GB tuỳ model) từ HuggingFace Hub,
cần kết nối mạng. Các lần sau dùng cache cục bộ, không cần mạng nữa.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

STRATEGIES = ["A_dieu", "B_khoan_context", "C_fixed"]
MODEL_NAME = "keepitreal/vietnamese-sbert"
BATCH_SIZE = 32


def load_chunks(strategy: str) -> list:
    path = CHUNKS_DIR / f"chunks_strategy_{strategy}.jsonl"
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_texts(model: SentenceTransformer, texts: list) -> np.ndarray:
    """Encode + CHUẨN HOÁ VECTOR (L2 norm = 1) -- bắt buộc để dùng
    IndexFlatIP (inner product) như cosine similarity thật sự. Nếu bỏ bước
    normalize này, kết quả tìm kiếm sẽ SAI (thiên vị theo độ dài vector)."""
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def build_vector_index_for_strategy(model: SentenceTransformer, strategy: str) -> dict:
    chunks = load_chunks(strategy)
    print(f"  -> {len(chunks)} chunk nạp từ chunks_strategy_{strategy}.jsonl")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product trên vector đã normalize = cosine similarity
    index.add(embeddings)

    return {
        "faiss_index": index,
        "chunks": chunks,
        "strategy": strategy,
        "model_name": MODEL_NAME,
        "embedding_dim": dim,
    }


def save_index(index_data: dict, strategy: str):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index_data["faiss_index"], str(INDEX_DIR / f"faiss_{strategy}.index"))
    # Lưu riêng metadata (không lưu được faiss_index object bằng pickle trực tiếp)
    meta = {k: v for k, v in index_data.items() if k != "faiss_index"}
    with open(INDEX_DIR / f"faiss_{strategy}.meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"  -> Đã lưu faiss_{strategy}.index + faiss_{strategy}.meta.pkl")


def main():
    print(f"Đang tải model '{MODEL_NAME}' (lần đầu sẽ tải từ HuggingFace Hub)...")
    model = SentenceTransformer(MODEL_NAME)
    print("Tải model xong.\n")

    for strategy in STRATEGIES:
        print(f"=== Chiến lược {strategy} ===")
        chunks_path = CHUNKS_DIR / f"chunks_strategy_{strategy}.jsonl"
        if not chunks_path.exists():
            print(f"  !! Bỏ qua -- không tìm thấy {chunks_path}")
            continue
        index_data = build_vector_index_for_strategy(model, strategy)
        save_index(index_data, strategy)


if __name__ == "__main__":
    main()