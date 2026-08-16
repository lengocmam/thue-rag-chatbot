"""
offtopic_gate.py
Nạp model cổng lọc off-topic đã huấn luyện (train_offtopic_gate.py), cung
cấp hàm is_in_scope() để dùng trong pipeline RAG -- đây là bước "Agent"
đơn giản nhất: hệ thống tự quyết định có nên đi tiếp (gọi LLM, tốn token)
hay dừng lại sớm và trả lời cố định.

Cách dùng:
    from offtopic_gate import OffTopicGate
    gate = OffTopicGate()
    in_scope, confidence = gate.is_in_scope("Ngưỡng doanh thu miễn thuế là bao nhiêu?")
"""

import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "offtopic_gate.pkl"

REFUSAL_MESSAGE = (
    "Xin lỗi, câu hỏi này nằm ngoài phạm vi tư vấn của tôi. Tôi chỉ có thể "
    "hỗ trợ các câu hỏi liên quan đến thuế thu nhập cá nhân, thuế hộ kinh "
    "doanh và hóa đơn điện tử theo quy định pháp luật hiện hành."
)


class OffTopicGate:
    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = 0.5):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        self.pipeline = saved["pipeline"]
        self.feature_type = saved["feature_type"]
        self.threshold = threshold

    def is_in_scope(self, query: str) -> tuple:
        """Trả về (in_scope: bool, confidence: float). confidence là xác
        suất mô hình cho là 'trong phạm vi' (label=1)."""
        proba = self.pipeline.predict_proba([query])[0][1]
        return proba >= self.threshold, float(proba)


if __name__ == "__main__":
    gate = OffTopicGate()
    test_queries = [
        "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh không phải nộp thuế thu nhập cá nhân?",
        "Hôm nay thời tiết thế nào?",
        "Mức phạt tù cho tội trộm cắp tài sản là bao nhiêu năm?",
        "Mức giảm trừ gia cảnh cho người phụ thuộc là bao nhiêu?",
    ]
    for q in test_queries:
        in_scope, conf = gate.is_in_scope(q)
        status = "TRONG PHẠM VI" if in_scope else "NGOÀI PHẠM VI (từ chối)"
        print(f"[{status}] (conf={conf:.3f}) {q}")