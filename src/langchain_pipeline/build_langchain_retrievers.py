"""
build_langchain_retrievers.py
Dựng lại retriever bằng LangChain thay vì tự viết tay -- tái dùng ĐÚNG dữ
liệu chunk đã có (chunks_strategy_*.jsonl), không tạo dữ liệu mới.

So với bản tự viết tay (build_bm25_index.py, build_vector_index.py,
hybrid_retriever.py), bản LangChain này thay thế:
    - Tự viết BM25Okapi + tokenize thủ công -> langchain BM25Retriever
    - Tự viết FAISS index + normalize vector thủ công -> langchain FAISS
      vectorstore (tự lo việc embedding + lưu/nạp index)
    - Tự viết Reciprocal Rank Fusion thủ công -> langchain EnsembleRetriever
      (đã cài sẵn RRF bên trong, chỉ cần khai báo trọng số mỗi retriever)

Cách chạy:
    pip install langchain langchain-community langchain-huggingface faiss-cpu
    python src/langchain_pipeline/build_langchain_retrievers.py
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain tái cấu trúc lớn ở bản 1.x: EnsembleRetriever (và các retriever
# "cổ điển" khác) chuyển từ langchain.retrievers sang package riêng
# langchain_classic.retrievers. Thử cả 2 đường dẫn để script chạy được
# trên cả bản LangChain cũ (0.x, phổ biến trong hầu hết tutorial) lẫn mới.
try:
    from langchain_classic.retrievers import EnsembleRetriever  # LangChain >= 1.0
except ImportError:
    from langchain.retrievers import EnsembleRetriever  # LangChain 0.x

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "index_langchain"

EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
STRATEGY = "B_khoan_context"  # đổi thành A_dieu/C_fixed nếu muốn build cho chiến lược khác

# Trọng số Ensemble: mặc định bằng nhau (0.5/0.5) giống RRF thường dùng.
# Kết quả thí nghiệm trước đó (BM25 > Hybrid > Dense trên domain này) gợi
# ý có thể thử tăng trọng số BM25 (VD 0.7/0.3) như MỘT THÍ NGHIỆM RIÊNG --
# không đổi mặc định ở đây để giữ so sánh công bằng với bản tự viết tay.
ENSEMBLE_WEIGHTS = [0.5, 0.5]


def load_chunks_as_documents(strategy: str) -> list:
    """Chuyển chunk JSONL đã có thành list[Document] của LangChain --
    giữ nguyên toàn bộ metadata (chunk_id, dieu, khoan, so_hieu_van_ban...)
    để sau này trích dẫn nguồn khi sinh câu trả lời."""
    path = CHUNKS_DIR / f"chunks_strategy_{strategy}.jsonl"
    documents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            documents.append(Document(
                page_content=c["text"],
                metadata={
                    "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                    "so_hieu_van_ban": c["so_hieu_van_ban"],
                    "dieu": c.get("dieu"), "khoan": c.get("khoan"),
                    "ten_dieu": c.get("ten_dieu"),
                },
            ))
    return documents


def build_bm25_retriever(documents: list, k: int = 10) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def build_vector_retriever(documents: list, k: int = 10):
    print(f"  Đang tải embedding model '{EMBEDDING_MODEL}' (lần đầu tải từ HuggingFace Hub)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(bm25_retriever, vector_retriever, weights=ENSEMBLE_WEIGHTS):
    """EnsembleRetriever của LangChain tự làm Reciprocal Rank Fusion bên
    trong -- tương đương hybrid_retriever.py tự viết tay ở bước trước,
    nhưng không cần tự cài đặt công thức RRF thủ công."""
    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=weights)


def save_vectorstore(vectorstore, strategy: str):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR / f"faiss_langchain_{strategy}"))
    print(f"  -> Đã lưu vectorstore tại {INDEX_DIR / f'faiss_langchain_{strategy}'}")


def main():
    print(f"=== Dựng retriever LangChain cho chiến lược {STRATEGY} ===")
    documents = load_chunks_as_documents(STRATEGY)
    print(f"Nạp {len(documents)} document.")

    print("Dựng BM25Retriever...")
    bm25_retriever = build_bm25_retriever(documents)

    print("Dựng vector retriever (FAISS + embedding)...")
    vectorstore = FAISS.from_documents(
        documents, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
    save_vectorstore(vectorstore, STRATEGY)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    print("Dựng Hybrid EnsembleRetriever...")
    hybrid_retriever = build_hybrid_retriever(bm25_retriever, vector_retriever)

    # Demo nhanh
    cau_hoi = "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh không phải nộp thuế thu nhập cá nhân?"
    print(f"\n--- Demo truy vấn: {cau_hoi} ---")
    results = hybrid_retriever.invoke(cau_hoi)
    for i, doc in enumerate(results[:5], 1):
        print(f"#{i} [{doc.metadata['chunk_id']}] {doc.metadata['so_hieu_van_ban']} "
              f"- {doc.metadata['dieu']} {doc.metadata.get('khoan') or ''}")
        print(f"    {doc.page_content[:120].strip()}...")


if __name__ == "__main__":
    main()