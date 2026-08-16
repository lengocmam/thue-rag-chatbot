"""
rag_chain.py
Hoàn thiện chữ "AG" (Augmented Generation) còn thiếu trong RAG -- trước đó
dự án mới dừng ở "R" (Retrieval). Module này ghép retriever đã dựng
(build_langchain_retrievers.py) với LLM qua LCEL (LangChain Expression
Language) -- cú pháp chain hiện đại của LangChain, dùng toán tử `|` để nối
các bước xử lý.

Thiết kế prompt (phần "Prompt Engineering" theo đúng yêu cầu công việc):
    1. Ép LLM CHỈ trả lời dựa trên context được cung cấp, không dùng kiến
       thức nền có sẵn (chống hallucination).
    2. Bắt buộc trích dẫn nguồn (Điều/Khoản, số hiệu văn bản) cho mỗi câu
       trả lời -- để người dùng tự kiểm chứng lại luật gốc.
    3. Nếu context không đủ thông tin, PHẢI thừa nhận không biết, không
       được bịa (đây là yêu cầu quan trọng nhất trong domain pháp luật).

Cách chạy:
    export OPENAI_API_KEY=sk-...      # hoặc set trên Windows: set OPENAI_API_KEY=...
    python src/langchain_pipeline/rag_chain.py
"""

import os
import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Đổi tên model Ollama ở đây nếu muốn dùng model khác llama3.1 (VD:
# "qwen2.5", "qwen2.5:7b"...) -- không cần sửa sâu trong hàm get_llm().
OLLAMA_MODEL_NAME = "llama3.1"
OPENAI_MODEL_NAME = "gpt-4o-mini"

sys.path.insert(0, str(Path(__file__).parent))
from build_langchain_retrievers import (
    load_chunks_as_documents, build_bm25_retriever, build_hybrid_retriever, STRATEGY,
)
from offtopic_gate import OffTopicGate, REFUSAL_MESSAGE

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn thuế, CHỈ trả lời dựa trên các đoạn văn bản pháp luật được cung cấp dưới đây. Tuân thủ nghiêm ngặt các quy tắc sau:

1. CHỈ sử dụng thông tin có trong phần "Ngữ cảnh" bên dưới. KHÔNG dùng kiến thức có sẵn của bạn để bổ sung hay suy đoán.
2. Nếu ngữ cảnh không đủ thông tin để trả lời, PHẢI trả lời: "Tôi không tìm thấy đủ thông tin trong dữ liệu hiện có để trả lời chính xác câu hỏi này." KHÔNG được bịa ra con số, mức thuế suất, hay quy định nào không có trong ngữ cảnh.
3. Với mỗi thông tin quan trọng (số liệu, mức thuế suất, ngưỡng...), PHẢI trích dẫn rõ nguồn theo định dạng: (Điều X, [Số hiệu văn bản]).
4. Nếu các đoạn ngữ cảnh có dấu hiệu MÂU THUẪN nhau (VD một văn bản cũ và một văn bản sửa đổi mới hơn), hãy ưu tiên văn bản có ngày ban hành GẦN ĐÂY hơn hoặc được ghi rõ là "sửa đổi, bổ sung" cho văn bản kia, và nêu rõ điều này trong câu trả lời.
5. Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_docs(docs: list) -> str:
    """Định dạng danh sách Document thành text đưa vào prompt, kèm thông
    tin nguồn ngay đầu mỗi đoạn để LLM dễ trích dẫn đúng."""
    parts = []
    for doc in docs:
        meta = doc.metadata
        nguon = f"[{meta.get('so_hieu_van_ban')} - {meta.get('dieu')} {meta.get('khoan') or ''}]"
        parts.append(f"{nguon}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def get_llm():
    """Chọn LLM theo thứ tự ưu tiên: OpenAI API (nếu có OPENAI_API_KEY)
    -> HuggingFace local pipeline (chạy offline, không cần API key, nhưng
    cần model tải về và máy đủ mạnh) -- giống đúng tinh thần fallback chain
    'OpenAI -> Ollama -> tổng hợp offline' đã thiết kế từ đầu dự án."""
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0)

    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0)
    except Exception:
        pass

    raise RuntimeError(
        "Chưa cấu hình LLM nào khả dụng. Cách khắc phục:\n"
        "  (1) Set biến môi trường OPENAI_API_KEY, hoặc\n"
        "  (2) Cài Ollama (https://ollama.com) và chạy 'ollama pull llama3.1', hoặc\n"
        "  (3) Tự thay get_llm() bằng HuggingFacePipeline với model local."
    )


def build_rag_chain(retriever, llm=None):
    """Chain LCEL: câu hỏi -> retriever lấy context -> format -> đưa vào
    prompt -> LLM -> parse text kết quả. Toán tử `|` nối các bước, mỗi
    bước là 1 Runnable -- đây chính là cú pháp LCEL hiện đại của LangChain."""
    llm = llm or get_llm()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def answer_question(query: str, rag_chain, gate: "OffTopicGate | None" = None) -> dict:
    """Hàm điều phối cấp cao -- ĐÂY LÀ PHẦN THỂ HIỆN 'TƯ DUY AGENT': hệ
    thống tự quyết định bước tiếp theo dựa trên kết quả bước trước, thay
    vì luôn đi thẳng một đường retrieve -> generate.

    Luồng quyết định:
        1. Cổng lọc off-topic chấm điểm câu hỏi.
        2. NẾU ngoài phạm vi -> DỪNG NGAY, trả lời từ chối cố định,
           KHÔNG gọi retriever/LLM (tiết kiệm token, đây là lý do chính
           để có cổng lọc thay vì để LLM tự nhận ra và từ chối).
        3. NẾU trong phạm vi -> đi tiếp toàn bộ chain RAG như bình thường."""
    if gate is not None:
        in_scope, confidence = gate.is_in_scope(query)
        if not in_scope:
            return {
                "answer": REFUSAL_MESSAGE, "gated": True,
                "gate_confidence": confidence, "llm_called": False,
            }

    answer = rag_chain.invoke(query)
    return {"answer": answer, "gated": False, "llm_called": True}


def main():
    print(f"=== Dựng RAG chain đầy đủ (chiến lược {STRATEGY}) ===")
    documents = load_chunks_as_documents(STRATEGY)
    bm25_retriever = build_bm25_retriever(documents, k=5)

    print("Đang tải embedding model để dựng vector retriever...")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert"))
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    hybrid_retriever = build_hybrid_retriever(bm25_retriever, vector_retriever)

    print("Đang khởi tạo LLM...")
    rag_chain = build_rag_chain(hybrid_retriever)

    print("Đang nạp cổng lọc off-topic...")
    try:
        gate = OffTopicGate()
    except FileNotFoundError:
        print("  !! Chưa có model cổng lọc (chạy scripts/train_offtopic_gate.py trước) "
              "-- bỏ qua bước gate, coi mọi câu hỏi đều trong phạm vi.")
        gate = None

    test_queries = [
        "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh không phải nộp thuế thu nhập cá nhân?",
        "Hôm nay thời tiết thế nào?",  # câu hỏi bẫy off-topic, kiểm tra gate có chặn đúng không
    ]
    for cau_hoi in test_queries:
        print(f"\nCâu hỏi: {cau_hoi}")

        # --- DEBUG: in ra context THẬT SỰ được đưa vào prompt, để biết
        # lỗi (nếu có) nằm ở retriever (không tìm ra đúng chunk) hay ở LLM
        # (có context đúng nhưng vẫn từ chối trả lời) ---
        retrieved_docs = hybrid_retriever.invoke(cau_hoi)
        print(f"  [DEBUG] Retriever trả về {len(retrieved_docs)} chunk:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"    #{i} [{doc.metadata['chunk_id']}] {doc.metadata['so_hieu_van_ban']} "
                  f"- {doc.metadata['dieu']} {doc.metadata.get('khoan') or ''}")

        result = answer_question(cau_hoi, rag_chain, gate)
        print(f"  [gated={result['gated']}, llm_called={result['llm_called']}]")
        print(f"  Trả lời: {result['answer']}")


if __name__ == "__main__":
    main()