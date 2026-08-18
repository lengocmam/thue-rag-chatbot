# Thuế RAG Chatbot

Chatbot hỏi-đáp pháp luật thuế Việt Nam (thuế thu nhập cá nhân, thuế hộ kinh doanh, hóa đơn điện tử) bằng kiến trúc RAG (Retrieval-Augmented Generation) — xây dựng như một dự án nghiên cứu có đo lường, không phải demo minh họa.

**Nguyên tắc viết README này**: mọi con số đều đo được và tái lập được bằng script tương ứng trong repo; mọi kết luận đều kèm cách đo hoặc nói rõ giới hạn — kể cả khi kết quả không như kỳ vọng ban đầu.

`Python` `LangChain` `FastAPI` `Ollama` `FAISS` `BM25` `scikit-learn`

---

## Kết quả nổi bật

| Chỉ số | Kết quả | Đo bằng |
|---|---|---|
| BM25 vs Dense embedding (Recall@5) | BM25 thắng có ý nghĩa thống kê ở cả 2 chiến lược chunking | Paired bootstrap, p < 0.001 |
| BM25 vs Hybrid RRF (Recall@5) | BM25 đơn lẻ thắng Hybrid | Paired bootstrap, p < 0.05 (4/4 phép so sánh) |
| Cổng lọc off-topic (TF-IDF + Logistic Regression) | AUC-ROC 0.986, F1 0.971 | Stratified 5-Fold CV, 90 mẫu |
| Contextual Retrieval (enrichment) | Cải thiện 1 chunk cụ thể, KHÔNG có ý nghĩa thống kê trên diện rộng | Paired bootstrap trên test set giữ kín, p = 0.26 |
| Độ trễ end-to-end (Ollama llama3.1, CPU) | ~22.6 giây/câu hỏi | Đo qua FastAPI `/api/chat` |

---

## Mục lục

- [Quyết định kỹ thuật đáng chú ý](#quyết-định-kỹ-thuật-đáng-chú-ý)
- [Kiến trúc pipeline](#kiến-trúc-pipeline)
- [Đánh giá khoa học](#đánh-giá-khoa-học)
- [Tech stack](#tech-stack)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cách chạy](#cách-chạy)
- [Giới hạn đã biết](#giới-hạn-đã-biết)
- [Tác giả](#tác-giả)

---

## Quyết định kỹ thuật đáng chú ý

Không liệt kê công nghệ đã dùng — đây là các quyết định có đánh đổi thật, kèm số đo đằng sau, không phải chọn vì "nghe hay hơn":

**BM25 làm retrieval mặc định, không phải dense embedding**: benchmark ban đầu (1 câu hỏi cụ thể) khiến dense embedding có vẻ đáng thử, nhưng đo trên 32 câu test set giữ kín cho thấy BM25 thắng có ý nghĩa thống kê ở toàn bộ 4 phép so sánh (p < 0.05). Kết luận không đến từ giả định "embedding hiện đại hơn nên tốt hơn", mà từ đo lường thực tế trên domain văn bản luật hành chính tiếng Việt.

**Hybrid RRF không được dùng làm mặc định dù về lý thuyết "kết hợp ưu điểm 2 bên"**: đo thực tế cho thấy Hybrid RRF *luôn* kém hơn BM25 đơn lẻ trên domain này — vì phải "hòa trộn" với tín hiệu dense yếu hơn. Giữ BM25 làm baseline mặc định, Hybrid chỉ dùng khi có bằng chứng ngược lại.

**Contextual Retrieval (enrichment chunk) không được áp dụng dù đã thử nghiệm**: thêm ngữ cảnh cấp văn bản vào đầu mỗi chunk cải thiện rõ rệt cho 1 chunk cụ thể (hạng #56 → #27/232), nhưng đo trên test set giữ kín (chưa từng dùng để thiết kế enrichment) cho kết quả không có ý nghĩa thống kê (p = 0.26) — không đủ bằng chứng để đưa vào production.

**OCR cho văn bản scan thay vì bỏ qua**: 1/5 văn bản gốc (Luật 109/2025/QH15) không có lớp text nhúng (PDF scan thuần). Thay vì loại bỏ, dựng pipeline OCR riêng (Tesseract + gói tiếng Việt) — chấp nhận chất lượng text thấp hơn (có lỗi chính tả rải rác) đổi lấy việc có được văn bản luật gốc quan trọng nhất trong bộ dữ liệu.

**3 chiến lược chunking song song, không chọn 1 ngay từ đầu**: theo Điều (A), theo Khoản kèm ngữ cảnh (B), fixed-size (C) — được đo độc lập vì giả định "chunk nhỏ hơn tìm chính xác hơn" hoá ra không đơn giản: tiêu chí đối chiếu ground-truth khác nhau giữa 3 chiến lược khiến không thể so sánh trực tiếp Recall@K giữa chúng (xem phần Giới hạn).

**Phát hiện và tự sửa lỗi test-set leakage**: thí nghiệm enrichment ban đầu được thiết kế dựa trên quan sát 1 câu hỏi cụ thể trong chính bộ ground-truth, rồi đánh giá lại trên cùng bộ đó — vi phạm nguyên tắc cơ bản. Đã khắc phục bằng cách tách dev set (14 câu, được phép "nhìn" để debug) và test set (32 câu, giữ kín tuyệt đối, chỉ dùng 1 lần để báo cáo số liệu cuối).

## Kiến trúc pipeline

```
PDF luật (text layer hoặc scan)
        │
        ▼
┌─────────────────────┐   PyMuPDF / pdfplumber (đối chiếu chéo khi nghi ngờ lỗi)
│   Ingestion          │   Tesseract OCR (riêng cho văn bản scan)
│   src/ingestion/      │   Regex parser: Chương > Điều > Khoản > Điểm
└─────────────────────┘
        │
        ▼
┌─────────────────────┐   Chiến lược A: theo Điều
│   Chunking            │   Chiến lược B: theo Khoản + ngữ cảnh Điều
│   src/ingestion/      │   Chiến lược C: fixed-size (baseline)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐   BM25Retriever (rank_bm25 / LangChain)
│   Indexing            │   FAISS + vietnamese-sbert (dense)
│   src/indexing/       │   EnsembleRetriever (Hybrid RRF, k=60)
│   src/langchain_pipeline/
└─────────────────────┘
        │
        ▼
┌─────────────────────┐   TF-IDF/Embedding + Logistic Regression
│   Cổng lọc off-topic  │   Chặn câu hỏi ngoài phạm vi TRƯỚC khi gọi LLM
│   src/langchain_pipeline/offtopic_gate.py
└─────────────────────┘
        │ (nếu qua cổng lọc)
        ▼
┌─────────────────────┐   LCEL chain: retriever | format | prompt | LLM | parser
│   Sinh câu trả lời     │   Prompt ép trích dẫn nguồn + từ chối khi thiếu dữ liệu
│   Ollama (llama3.1) hoặc OpenAI
│   src/langchain_pipeline/rag_chain.py
└─────────────────────┘
        │
        ▼
┌─────────────────────┐   FastAPI: /api/chat · /api/search · /health · /api/stats
│   Serving             │   Nạp model 1 lần lúc khởi động (lifespan)
│   src/api/server.py
└─────────────────────┘
```

## Đánh giá khoa học

Khung đánh giá dùng metric chuẩn ngành (Recall@K, MRR) và kiểm định thống kê (paired bootstrap, 10.000 lần lặp) — không chỉ nhận xét định tính. Toàn bộ script trong `eval/` và `scripts/`.

### Retrieval: BM25 vs Dense vs Hybrid (test set giữ kín, 32 câu)

| Chiến lược | Retriever | Recall@5 | MRR |
|---|---|---|---|
| A_dieu | BM25 | 0.969 | 0.784 |
| A_dieu | Dense | 0.688 | 0.580 |
| A_dieu | Hybrid RRF | 0.812 | 0.655 |
| B_khoan_context | BM25 | 0.906 | 0.757 |
| B_khoan_context | Dense | 0.531 | 0.406 |
| B_khoan_context | Hybrid RRF | 0.719 | 0.543 |

Kiểm định paired bootstrap: **cả 4/4 phép so sánh (BM25 vs Dense, BM25 vs Hybrid, ở cả 2 chiến lược) đều có ý nghĩa thống kê (p < 0.05)**, khẳng định BM25 vượt trội không phải do ngẫu nhiên mẫu nhỏ.

*(Chiến lược C_fixed bị loại khỏi so sánh thuật toán — xem phần Giới hạn.)*

### Cổng lọc off-topic: so sánh 2 phương án đặc trưng đầu vào

| Phương án | AUC-ROC | F1 |
|---|---|---|
| TF-IDF + Logistic Regression | **0.986** | **0.971** |
| Vietnamese-SBERT Embedding + Logistic Regression | 0.981 | 0.951 |

Đánh giá bằng Stratified 5-Fold Cross-Validation (90 mẫu: 52 trong phạm vi thuế, 38 ngoài phạm vi — cố tình đưa nhiều câu "khó" như luật khác/tài chính không liên quan thuế để test ranh giới thật). Pattern TF-IDF thắng embedding lặp lại nhất quán với kết quả retrieval — 2 bằng chứng độc lập cùng chỉ về hạn chế của `vietnamese-sbert` trên domain này.

### Thí nghiệm bổ sung: Contextual Retrieval (enrichment)

Giả thuyết: thêm 1 câu ngữ cảnh cấp văn bản vào đầu mỗi chunk trước khi index có cải thiện retrieval không?

| | Trước enrich | Sau enrich |
|---|---|---|
| Hạng chunk mục tiêu (case cụ thể) | #56/232 | #27/232 |
| Recall@5 (test set giữ kín, 32 câu) | 0.906 | 0.844 |
| MRR (test set giữ kín) | 0.757 | 0.767 |
| p-value (paired bootstrap) | — | 0.264 |

**Kết luận trung thực**: cải thiện rõ cho 1 trường hợp cụ thể, nhưng KHÔNG đủ ý nghĩa thống kê trên diện rộng — không nên khái quát hoá thành "enrichment luôn giúp ích" chỉ từ 1 quan sát.

## Tech stack

| Tầng | Công nghệ |
|---|---|
| PDF/OCR | pdfplumber, PyMuPDF, Tesseract OCR (+ gói tiếng Việt), poppler-utils |
| Retrieval | rank_bm25, sentence-transformers (`keepitreal/vietnamese-sbert`), FAISS |
| RAG framework | LangChain (`langchain`, `langchain-community`, `langchain-huggingface`), LCEL |
| LLM | Ollama (`llama3.1`) — self-host, hoặc OpenAI (`gpt-4o-mini`) qua `langchain-openai` |
| Off-topic gate | scikit-learn (TF-IDF/Logistic Regression, so sánh với Embedding) |
| Đánh giá | numpy thuần — Recall@K/MRR, paired bootstrap significance test |
| API | FastAPI, Uvicorn, Pydantic |

## Cấu trúc thư mục

```
thue-rag-chatbot/
├── data/
│   ├── raw/                  # PDF gốc + *.meta.json (metadata từng văn bản)
│   ├── processed/            # chunks_strategy_{A,B,C}*.jsonl, *.noi_dung.json
│   ├── index/                # BM25 (.pkl) + FAISS (.index/.meta.pkl)
│   ├── index_langchain/      # vectorstore dựng bằng LangChain
│   └── eval/                 # ground_truth(_dev/_test).jsonl, trap_questions.jsonl,
│                              #   offtopic_training_data.jsonl, annotation_guide.md
├── src/
│   ├── ingestion/             # pdf_parser.py, legal_structure_parser.py, chunkers.py
│   ├── indexing/               # build_bm25_index.py, build_vector_index.py
│   ├── retrieval/              # hybrid_retriever.py, diagnose_rank.py
│   ├── langchain_pipeline/     # build_langchain_retrievers.py, rag_chain.py, offtopic_gate.py
│   └── api/                    # server.py (FastAPI)
├── eval/
│   └── metrics.py              # Recall@K, MRR, paired_bootstrap_test
├── scripts/                    # extract_*.py (1 script/văn bản), generate_ground_truth.py,
│                                #   run_retrieval_eval.py, train_offtopic_gate.py, ...
├── models/                     # offtopic_gate.pkl
└── reports/                    # retrieval_comparison.json, offtopic_gate_comparison.json
```

## Cách chạy

```bash
# 1. Cài đặt
pip install -r requirements.txt
# Riêng OCR (chỉ cần cho Luật 109/2025/QH15, văn bản dạng scan):
#   Windows: cài Tesseract-OCR (kèm gói Vietnamese) + Poppler, thêm vào PATH
#   apt: sudo apt install tesseract-ocr tesseract-ocr-vie poppler-utils

# 2. Ingest + parse toàn bộ văn bản (mỗi văn bản 1 script riêng)
python scripts/ingest_one_doc.py data/raw/87-2026TTBTC.pdf
python scripts/extract_09_2026_qh16.py
python scripts/extract_91_2026_ttbtc.py
python scripts/extract_141_2026_ndcp.py
python scripts/extract_109_2025_qh15.py   # cần Tesseract (OCR)
python scripts/chunk_remaining_docs.py     # gộp 4 văn bản còn lại vào chunk chung

# 3. Dựng ground-truth + tách dev/test
python scripts/generate_ground_truth.py
python scripts/split_dev_test.py

# 4. Dựng index
python scripts/build_bm25_index.py
python src/indexing/build_vector_index.py           # cần internet lần đầu (tải model)
python src/langchain_pipeline/build_langchain_retrievers.py

# 5. Train cổng lọc off-topic
python scripts/build_offtopic_training_data.py
python scripts/train_offtopic_gate.py

# 6. Đánh giá retrieval (Recall@K, MRR, kiểm định thống kê)
python scripts/run_retrieval_eval.py

# 7. Chạy LLM (chọn 1 trong 2)
#   (a) Ollama (miễn phí, local): cài từ https://ollama.com, sau đó:
ollama pull llama3.1
pip install langchain-ollama
#   (b) OpenAI:
export OPENAI_API_KEY=sk-...    # Windows: setx OPENAI_API_KEY "sk-..."

# 8. Chạy thử pipeline đầy đủ qua CLI
python src/langchain_pipeline/rag_chain.py

# 9. Chạy API server
uvicorn src.api.server:app --reload --port 8000
# Mở http://127.0.0.1:8000/docs để test qua Swagger UI
```

## Giới hạn đã biết

Gom hết vào 1 chỗ để dễ soi, nói thẳng thay vì để người đọc tự phát hiện:

- **Chunk dạng "tìm-và-thay" khó retrieve ở mọi phương pháp**: các Điều/Khoản chỉ chứa lệnh sửa đổi tham chiếu chéo (VD *"Sửa đổi cụm từ '500 triệu đồng' thành '01 tỷ đồng' tại Điều 3, Điều 4..."*) xếp hạng rất thấp (#56-120/232) ở cả BM25, Dense, lẫn Hybrid — vì bản thân đoạn văn không chứa từ khóa tự nhiên khớp với câu hỏi người dùng thật. Đây là hạn chế cấu trúc của chunking theo Khoản, không phải lỗi retrieval.
- **Chuỗi phiên bản văn bản (temporal versioning) chưa được xử lý tường minh**: cùng 1 quy định có thể xuất hiện ở 3 tầng văn bản khác nhau theo thời gian (VD ngưỡng doanh thu: Luật gốc ghi "500 triệu" → Luật sửa đổi bỏ số, giao Chính phủ quy định → Nghị định mới ghi "01 tỷ") — hệ thống hiện KHÔNG có cơ chế tự động biết văn bản nào đang hiệu lực, dựa hoàn toàn vào retrieval tìm đúng văn bản mới nhất và prompt yêu cầu LLM ưu tiên diễn giải theo văn bản gần đây.
- **Đánh giá chiến lược C_fixed dùng tiêu chí quá lỏng lẻo**: do fixed-size chunking không có ranh giới Điều/Khoản rõ ràng, ground-truth đối chiếu theo tiêu chí "cùng văn bản" thay vì "cùng Khoản" — khiến Recall@5 của C_fixed luôn gần kịch trần một cách giả tạo, KHÔNG so sánh được trực tiếp với A_dieu/B_khoan_context.
- **Chỉ kiểm chứng 1 model embedding** (`vietnamese-sbert`) — kết luận "dense embedding kém hơn BM25" chỉ đúng cho model này trên domain này, chưa thể khái quát hoá cho mọi model embedding tiếng Việt.
- **OCR cho Luật 109/2025/QH15 có lỗi chính tả rải rác** (nhầm dấu thanh, nhầm chữ hình dạng tương tự) — cấu trúc Điều/Khoản/Điểm đáng tin cậy, nhưng nội dung câu chữ cần đối chiếu bản gốc trước khi dùng làm ground-truth chính thức cho các ứng dụng đòi hỏi độ chính xác pháp lý cao.
- **Bộ ground-truth còn nhỏ** (46 câu, 32 câu trong test set giữ kín) — đủ để phát hiện khác biệt lớn (như BM25 vs Dense) nhưng có thể thiếu năng lực thống kê (statistical power) để phát hiện khác biệt nhỏ hơn giữa các cấu hình gần nhau.
- **LLM sinh câu trả lời (`llama3.1` 8B) là model tổng quát, chưa fine-tune riêng cho domain pháp luật thuế tiếng Việt** — đã quan sát trường hợp model từ chối trả lời đúng thiết kế (an toàn) khi retrieval không tìm đủ chunk liên quan, nhưng chưa được benchmark hệ thống về độ chính xác câu trả lời khi có đủ context (mới dừng ở đánh giá retrieval, chưa đánh giá answer quality bằng LLM-as-judge hay con người).
- **Hạ tầng chỉ ở quy mô dev/demo**: chạy trên 1 máy, Ollama CPU-only (~22.6 giây/câu hỏi), chưa có cơ chế cache, rate-limit, hay xử lý đồng thời nhiều request.

## Hướng phát triển tiếp theo

Những việc đã xác định rõ nhưng chưa làm (khác với "Giới hạn đã biết" — đây là việc CÓ THỂ làm để khắc phục trực tiếp từng giới hạn):

- **Đánh giá chất lượng câu trả lời (answer quality), không chỉ retrieval**: hiện mới đo Recall@K/MRR của retrieval; chưa có benchmark có hệ thống cho câu trả lời cuối cùng của LLM (VD: LLM-as-judge chấm điểm "faithfulness" — câu trả lời có bám sát context hay không — kết hợp đối chiếu thủ công một phần, theo đúng mô hình đã dùng để đánh giá cổng lọc off-topic).
- **Xử lý tường minh quan hệ "văn bản nào đang hiệu lực"**: xây một lớp metadata quan hệ giữa các văn bản (văn bản A sửa đổi văn bản B, hiệu lực từ ngày nào) để retrieval/prompt có thể chủ động loại bỏ hoặc hạ ưu tiên nội dung đã lỗi thời, thay vì phó mặc hoàn toàn cho LLM tự suy luận từ context thô.
- **Mở rộng ground-truth lên 100-150 câu** (hiện 46 câu) để có đủ năng lực thống kê phát hiện khác biệt nhỏ hơn giữa các cấu hình gần nhau (VD so sánh 2 model embedding khác nhau).
- **Thử thêm ít nhất 1 model embedding khác** (VD `multilingual-e5-base`) để biết kết luận "BM25 thắng embedding" là do bản chất domain hay do riêng model `vietnamese-sbert`.
- **Enrichment tinh vi hơn**: thí nghiệm hiện tại (thêm ngữ cảnh cấp văn bản, đồng loạt mọi chunk) không có ý nghĩa thống kê — có thể thử enrichment CHỌN LỌC (chỉ áp dụng cho nhóm chunk "tìm-và-thay" đã xác định là khó retrieve) thay vì áp dụng tràn lan.
- **Cache + rate limit cho API**: hiện `/api/chat` chạy đồng bộ, không cache câu hỏi trùng lặp, không giới hạn số request đồng thời — cần thiết nếu triển khai thật.

## Tác giả

Dự án cá nhân — RAG chatbot pháp luật thuế Việt Nam, xây dựng với trọng tâm đo lường khoa học và tư duy hệ thống (retrieval, chunking, agent gating, LLM serving) hơn là chỉ lắp ráp công cụ có sẵn.

Bản quyền © 2026. Xem `LICENSE` để biết điều khoản sử dụng.