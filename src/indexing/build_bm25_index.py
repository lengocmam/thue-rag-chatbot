"""
build_bm25_index.py
Dựng BM25 index (sparse/lexical retrieval) từ 1 file chunks_strategy_*.jsonl.

Cách chạy (build cho cả 3 chiến lược cùng lúc):
    python src/indexing/build_bm25_index.py

Tokenizer: dùng tách từ ĐƠN GIẢN theo khoảng trắng + lowercase (giống baseline
TF-IDF/BM25 "thô" trong nhiều nghiên cứu IR tiếng Việt). Đây là lựa chọn CÓ
CHỦ ĐÍCH cho baseline đầu tiên -- không dùng word segmentation (VD
underthesea/pyvi tách "thu_nhập" thành 1 token) vì:
  (1) muốn baseline đơn giản nhất có thể để so sánh:
  (2) việc thêm word segmentation là một BIẾN THÍ NGHIỆM RIÊNG có thể thêm
      sau này (so sánh "BM25 + segmentation" vs "BM25 không segmentation")
      -- không nên trộn 2 thay đổi cùng lúc khi mới bắt đầu.
Nếu sau này muốn thử segmentation, chỉ cần thay hàm `simple_tokenize`.
"""

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

STRATEGIES = ["A_dieu", "B_khoan_context", "C_fixed"]


def simple_tokenize(text: str) -> list:
    """Tách từ đơn giản: lowercase + tách theo khoảng trắng/dấu câu.
    Giữ nguyên dấu tiếng Việt (không bỏ dấu) -- bỏ dấu là 1 lựa chọn khác
    có thể thử nghiệm riêng (ảnh hưởng đến khả năng khớp khi người dùng gõ
    không dấu)."""
    text = text.lower()
    tokens = re.findall(r"[\wÀ-ỹ]+", text, re.UNICODE)
    return tokens


def load_chunks(strategy: str) -> list:
    path = CHUNKS_DIR / f"chunks_strategy_{strategy}.jsonl"
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_bm25_for_strategy(strategy: str) -> dict:
    chunks = load_chunks(strategy)
    print(f"  -> {len(chunks)} chunk nạp từ chunks_strategy_{strategy}.jsonl")

    tokenized_corpus = [simple_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Lưu riêng: model BM25 + danh sách chunk gốc (để tra lại text/metadata
    # khi có kết quả truy hồi theo index số nguyên)
    index_data = {
        "bm25_model": bm25,
        "chunks": chunks,  # giữ nguyên metadata đầy đủ (chunk_id, dieu, khoan...)
        "strategy": strategy,
        "tokenizer": "simple_tokenize (lowercase + word-char split, giữ dấu)",
    }
    return index_data


def save_index(index_data: dict, strategy: str):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INDEX_DIR / f"bm25_{strategy}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(index_data, f)
    print(f"  -> Đã lưu {out_path}")


def main():
    for strategy in STRATEGIES:
        print(f"=== Chiến lược {strategy} ===")
        chunks_path = CHUNKS_DIR / f"chunks_strategy_{strategy}.jsonl"
        if not chunks_path.exists():
            print(f"  !! Bỏ qua -- không tìm thấy {chunks_path}")
            continue
        index_data = build_bm25_for_strategy(strategy)
        save_index(index_data, strategy)


if __name__ == "__main__":
    main()