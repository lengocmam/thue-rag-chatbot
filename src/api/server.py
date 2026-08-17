"""
server.py
FastAPI serving layer -- expose toàn bộ pipeline RAG (retriever + cổng
lọc off-topic + LLM) qua REST API để BE/FE gọi vào, thay vì phải chạy
script Python trực tiếp.

Thiết kế quan trọng: NẠP MODEL 1 LẦN DUY NHẤT lúc khởi động server (qua
`lifespan`), không nạp lại mỗi request -- vì tải embedding model + LLM
tốn vài giây đến vài chục giây, nếu nạp lại mỗi request thì API sẽ chậm
không dùng được trong thực tế.

Endpoints:
    GET  /health          -- kiểm tra server còn sống, model đã nạp xong chưa
    POST /api/chat         -- hỏi đáp đầy đủ (gate -> retrieve -> LLM sinh câu trả lời)
    POST /api/search       -- chỉ retrieval, KHÔNG gọi LLM (để debug/hiển thị nguồn nhanh)
    GET  /api/stats         -- thống kê nhanh (số chunk đã index, model đang dùng)

Cách chạy:
    pip install fastapi uvicorn
    uvicorn src.api.server:app --reload --port 8000
    (mở http://127.0.0.1:8000/docs để xem Swagger UI tự sinh)
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "langchain_pipeline"))

# Biến toàn cục giữ các đối tượng đã nạp -- gán giá trị thật trong lifespan()
_state = {
    "retriever": None,
    "rag_chain": None,
    "gate": None,
    "startup_time": None,
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chạy 1 LẦN DUY NHẤT khi server khởi động -- nạp toàn bộ model nặng
    (embedding, FAISS, LLM, cổng lọc) vào _state để mọi request sau đó chỉ
    cần TRA CỨU, không phải NẠP LẠI."""
    print("Đang khởi động server -- nạp model (có thể mất 10-60 giây)...")
    start = time.time()

    from build_langchain_retrievers import (
        load_chunks_as_documents, build_bm25_retriever, build_hybrid_retriever, STRATEGY,
    )
    from rag_chain import build_rag_chain
    from offtopic_gate import OffTopicGate

    documents = load_chunks_as_documents(STRATEGY)
    bm25_retriever = build_bm25_retriever(documents, k=5)

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert"))
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    hybrid_retriever = build_hybrid_retriever(bm25_retriever, vector_retriever)
    rag_chain = build_rag_chain(hybrid_retriever)

    try:
        gate = OffTopicGate()
    except FileNotFoundError:
        print("  !! Chưa có model cổng lọc -- chạy không có gate.")
        gate = None

    _state["retriever"] = hybrid_retriever
    _state["rag_chain"] = rag_chain
    _state["gate"] = gate
    _state["startup_time"] = time.time() - start
    _state["ready"] = True
    print(f"Server sẵn sàng sau {_state['startup_time']:.1f} giây.")

    yield  # server chạy trong khoảng thời gian này

    print("Đang tắt server...")


app = FastAPI(
    title="Thuế RAG Chatbot API",
    description="API hỏi đáp về thuế thu nhập cá nhân, thuế hộ kinh doanh, hóa đơn điện tử",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Schema request/response (Pydantic -- FastAPI tự validate + sinh docs) ----------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Câu hỏi của người dùng")
    max_sources_shown: int = Field(
        5, ge=1, le=20,
        description="Số nguồn tối đa hiển thị cho người dùng. KHÔNG ảnh hưởng "
                     "context đưa vào LLM (LLM vẫn dùng toàn bộ chunk truy hồi "
                     "được để trả lời) -- chỉ cắt bớt danh sách sources trả về."
    )


class SourceChunk(BaseModel):
    chunk_id: str
    so_hieu_van_ban: str
    dieu: str | None = None
    khoan: str | None = None
    text_preview: str


class ChatResponse(BaseModel):
    answer: str
    gated: bool = Field(..., description="True nếu bị cổng lọc off-topic chặn (không gọi LLM)")
    gate_confidence: float | None = None
    sources: list[SourceChunk] = []
    latency_ms: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)


class SearchResponse(BaseModel):
    results: list[SourceChunk]
    latency_ms: float


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok" if _state["ready"] else "starting", "ready": _state["ready"]}


@app.get("/api/stats")
def stats():
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Server đang khởi động, thử lại sau vài giây.")
    return {
        "startup_time_seconds": round(_state["startup_time"], 2),
        "gate_enabled": _state["gate"] is not None,
        "embedding_model": "keepitreal/vietnamese-sbert",
    }


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Chỉ chạy retrieval, KHÔNG gọi LLM -- dùng để debug nhanh hoặc khi
    FE chỉ cần hiển thị danh sách nguồn liên quan mà chưa cần câu trả lời
    tổng hợp (tiết kiệm thời gian/token so với /api/chat)."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Server đang khởi động, thử lại sau vài giây.")

    start = time.time()
    docs = _state["retriever"].invoke(req.query)[: req.top_k]
    results = [
        SourceChunk(
            chunk_id=d.metadata["chunk_id"], so_hieu_van_ban=d.metadata["so_hieu_van_ban"],
            dieu=d.metadata.get("dieu"), khoan=d.metadata.get("khoan"),
            text_preview=d.page_content[:200],
        )
        for d in docs
    ]
    return SearchResponse(results=results, latency_ms=(time.time() - start) * 1000)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Hỏi đáp đầy đủ: cổng lọc off-topic -> (nếu qua) retrieval -> LLM
    sinh câu trả lời có trích dẫn nguồn."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Server đang khởi động, thử lại sau vài giây.")

    start = time.time()
    gate = _state["gate"]

    if gate is not None:
        in_scope, confidence = gate.is_in_scope(req.question)
        if not in_scope:
            from offtopic_gate import REFUSAL_MESSAGE
            return ChatResponse(
                answer=REFUSAL_MESSAGE, gated=True, gate_confidence=confidence,
                sources=[], latency_ms=(time.time() - start) * 1000,
            )

    docs = _state["retriever"].invoke(req.question)
    # QUAN TRỌNG: sources hiển thị bị CẮT theo max_sources_shown, nhưng
    # rag_chain.invoke() bên dưới vẫn tự gọi lại retriever với ĐẦY ĐỦ
    # docs (không bị cắt) để LLM có toàn bộ context -- 2 việc tách biệt.
    sources = [
        SourceChunk(
            chunk_id=d.metadata["chunk_id"], so_hieu_van_ban=d.metadata["so_hieu_van_ban"],
            dieu=d.metadata.get("dieu"), khoan=d.metadata.get("khoan"),
            text_preview=d.page_content[:200],
        )
        for d in docs[: req.max_sources_shown]
    ]

    try:
        answer = _state["rag_chain"].invoke(req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Lỗi khi gọi LLM: {type(e).__name__}: {str(e)[:200]}")

    return ChatResponse(
        answer=answer, gated=False,
        gate_confidence=(gate.is_in_scope(req.question)[1] if gate else None),
        sources=sources, latency_ms=(time.time() - start) * 1000,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)