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

sys.path.insert(0, str(Path(__file__).parent))
from build_langchain_retrievers import (
    load_chunks_as_documents, build_bm25_retriever, build_hybrid_retriever, STRATEGY,
)

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
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1", temperature=0)
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

    cau_hoi = "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh không phải nộp thuế thu nhập cá nhân?"
    print(f"\nCâu hỏi: {cau_hoi}\n")
    answer = rag_chain.invoke(cau_hoi)
    print("Trả lời:")
    print(answer)


if __name__ == "__main__":
    main()