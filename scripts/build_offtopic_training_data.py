"""
build_offtopic_training_data.py
Sinh dữ liệu huấn luyện cho CỔNG LỌC OFF-TOPIC -- 1 classifier nhẹ chạy
TRƯỚC khi gọi LLM, để chặn câu hỏi ngoài phạm vi thuế (tiết kiệm token,
tránh LLM cố trả lời bừa cho câu hỏi không liên quan). Đây là ví dụ cho
"tư duy Agent" -- hệ thống tự quyết định có nên đi tiếp bước gọi LLM hay
dừng lại sớm.

Nhãn dương (label=1, TRONG phạm vi): lấy trực tiếp từ ground_truth.jsonl +
trap_questions.jsonl đã có -- KHÔNG viết lại tay, tận dụng dữ liệu đã kiểm
chứng.

Nhãn âm (label=0, NGOÀI phạm vi): viết tay, cố tình đa dạng nhiều loại để
classifier học được ranh giới thật, không chỉ học "khác chủ đề thuế nói
chung" một cách hời hợt:
    - Chit-chat / hỏi thăm thông thường
    - Kiến thức phổ thông không liên quan luật
    - Luật KHÁC không thuộc phạm vi thuế (hình sự, dân sự, lao động...)
      -- đây là nhóm KHÓ NHẤT, dễ bị classifier nhầm là "liên quan luật
      nói chung" nên "trong phạm vi" -- cố tình đưa nhiều để test giới hạn
    - Câu hỏi về công nghệ/AI/lập trình
    - Câu hỏi thể thao/giải trí/nấu ăn

Cách chạy:
    python scripts/build_offtopic_training_data.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
GT_PATH = PROJECT_ROOT / "data" / "eval" / "ground_truth.jsonl"
TRAP_PATH = PROJECT_ROOT / "data" / "eval" / "trap_questions.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "eval" / "offtopic_training_data.jsonl"


OFF_TOPIC_EXAMPLES = [
    # --- Chit-chat / hỏi thăm thông thường ---
    "Xin chào, bạn khỏe không?",
    "Hôm nay thời tiết Hà Nội thế nào?",
    "Bạn tên gì?",
    "Bạn có thể giúp tôi việc gì?",
    "Cảm ơn bạn nhiều nhé",
    "Tôi buồn quá, bạn an ủi tôi được không?",

    # --- Kiến thức phổ thông không liên quan luật ---
    "Thủ đô của Nhật Bản là gì?",
    "Trái đất quay quanh mặt trời mất bao lâu?",
    "Ai là người phát minh ra bóng đèn?",
    "Công thức tính diện tích hình tròn là gì?",
    "Nước sôi ở bao nhiêu độ C?",
    "1 năm ánh sáng bằng bao nhiêu km?",

    # --- Luật KHÁC không thuộc phạm vi thuế (nhóm khó nhất) ---
    "Mức phạt tù cho tội trộm cắp tài sản là bao nhiêu năm?",
    "Thủ tục ly hôn đơn phương cần giấy tờ gì?",
    "Người lao động được nghỉ phép năm bao nhiêu ngày theo Luật Lao động?",
    "Độ tuổi kết hôn tối thiểu theo pháp luật Việt Nam là bao nhiêu?",
    "Thời hiệu khởi kiện tranh chấp đất đai là bao lâu?",
    "Mức xử phạt vi phạm giao thông khi vượt đèn đỏ là bao nhiêu?",
    "Quy định về đăng ký kết hôn với người nước ngoài như thế nào?",
    "Thủ tục xin cấp lại chứng minh nhân dân bị mất ra sao?",
    "Luật Bảo vệ môi trường quy định gì về xử lý rác thải công nghiệp?",
    "Hợp đồng lao động thử việc tối đa bao nhiêu ngày?",

    # --- Công nghệ / AI / lập trình ---
    "Python và Java ngôn ngữ nào dễ học hơn?",
    "Cách cài đặt Docker trên Windows như thế nào?",
    "Sự khác biệt giữa AI và Machine Learning là gì?",
    "Làm sao để tối ưu tốc độ website?",
    "Git merge và git rebase khác nhau thế nào?",

    # --- Thể thao / giải trí / nấu ăn ---
    "Đội tuyển Việt Nam đá vòng loại World Cup khi nào?",
    "Cách nấu phở bò truyền thống như thế nào?",
    "Bộ phim nào đoạt giải Oscar năm ngoái?",
    "Cầu thủ nào ghi nhiều bàn thắng nhất lịch sử bóng đá?",
    "Công thức làm bánh mì tại nhà đơn giản nhất là gì?",

    # --- Y tế / sức khỏe (dễ nhầm là "cần tư vấn chuyên môn" nói chung) ---
    "Triệu chứng của bệnh cảm cúm là gì?",
    "Uống bao nhiêu nước mỗi ngày là đủ?",
    "Tập thể dục bao lâu mỗi ngày là tốt cho sức khỏe?",

    # --- Tài chính KHÔNG liên quan thuế (dễ nhầm nhất vì cùng nhóm "tài chính") ---
    "Nên đầu tư vào cổ phiếu hay vàng trong năm nay?",
    "Lãi suất gửi tiết kiệm ngân hàng hiện nay là bao nhiêu?",
    "Cách lập kế hoạch chi tiêu cá nhân hiệu quả?",
]


def load_in_domain_questions() -> list:
    questions = []
    for path in (GT_PATH, TRAP_PATH):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                questions.append(q["question"])
    return questions


def main():
    in_domain = load_in_domain_questions()
    print(f"Nạp {len(in_domain)} câu TRONG phạm vi (từ ground_truth + trap_questions).")
    print(f"Có {len(OFF_TOPIC_EXAMPLES)} câu NGOÀI phạm vi (viết tay).")

    records = []
    for q in in_domain:
        records.append({"text": q, "label": 1})
    for q in OFF_TOPIC_EXAMPLES:
        records.append({"text": q, "label": 0})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTổng {len(records)} mẫu -> {OUT_PATH}")
    print(f"  Label 1 (trong phạm vi): {len(in_domain)}")
    print(f"  Label 0 (ngoài phạm vi): {len(OFF_TOPIC_EXAMPLES)}")
    if len(in_domain) / max(len(OFF_TOPIC_EXAMPLES), 1) > 2 or len(OFF_TOPIC_EXAMPLES) / max(len(in_domain), 1) > 2:
        print("  !! CẢNH BÁO: 2 lớp lệch nhau khá nhiều (imbalanced) -- "
              "cần dùng class_weight='balanced' khi train, và ưu tiên "
              "đọc AUC-ROC/F1 thay vì Accuracy khi đánh giá.")


if __name__ == "__main__":
    main()