"""
split_dev_test.py
Tách ground_truth.jsonl thành DEV set (được phép "nhìn" để phát hiện lỗi,
thiết kế cải tiến) và TEST set (giữ kín hoàn toàn, chỉ dùng để báo cáo số
liệu CUỐI CÙNG) -- khắc phục lỗi phương pháp luận đã phát hiện: câu hỏi
GT001 (ngưỡng doanh thu) được dùng để PHÁT HIỆN vấn đề retrieval, rồi lại
bị dùng để ĐÁNH GIÁ enrichment -- đây là "test set leakage" (nhìn thấy dữ
liệu test khi thiết kế), làm số liệu enrichment mất tính khách quan.

Nguyên tắc tách:
    - GT001 và các câu đã "bị nhìn thấy" trong quá trình debug thủ công
      trước đó BẮT BUỘC nằm trong dev set (không thể giả vờ chưa từng
      thấy).
    - Phần còn lại chia ngẫu nhiên theo tỷ lệ ~30% dev / ~70% test, có
      seed cố định để tái lập được.
    - TỪ THỜI ĐIỂM NÀY TRỞ ĐI: mọi thiết kế cải tiến (enrichment, prompt,
      chunking...) chỉ được phép nhìn dev set. Test set CHỈ dùng 1 lần
      duy nhất để báo cáo số liệu cuối cùng trong khóa luận.

Cách chạy:
    python scripts/split_dev_test.py
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
GT_PATH = PROJECT_ROOT / "data" / "eval" / "ground_truth.jsonl"
DEV_OUT = PROJECT_ROOT / "data" / "eval" / "ground_truth_dev.jsonl"
TEST_OUT = PROJECT_ROOT / "data" / "eval" / "ground_truth_test.jsonl"

# Các câu ĐÃ BỊ "NHÌN THẤY" trong quá trình debug thủ công trước đó --
# bắt buộc đưa vào dev set, không được tính vào test set dù ngẫu nhiên có
# chọn trúng hay không.
ALREADY_SEEN_IDS = {"GT001"}  # câu ngưỡng doanh thu, dùng để phát hiện vấn đề dieu1_khoan1

DEV_RATIO = 0.30
SEED = 42


def main():
    with open(GT_PATH, encoding="utf-8") as f:
        gt = [json.loads(l) for l in f]

    seen = [q for q in gt if q["id"] in ALREADY_SEEN_IDS]
    unseen = [q for q in gt if q["id"] not in ALREADY_SEEN_IDS]

    rng = random.Random(SEED)
    rng.shuffle(unseen)

    n_dev_needed = round(len(gt) * DEV_RATIO) - len(seen)
    n_dev_needed = max(n_dev_needed, 0)

    dev = seen + unseen[:n_dev_needed]
    test = unseen[n_dev_needed:]

    with open(DEV_OUT, "w", encoding="utf-8") as f:
        for q in dev:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(TEST_OUT, "w", encoding="utf-8") as f:
        for q in test:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Tổng {len(gt)} câu -> dev {len(dev)} câu / test {len(test)} câu")
    print(f"  Dev bao gồm (đã bị 'nhìn thấy'): {[q['id'] for q in seen]}")
    print(f"\nTỪ NAY: mọi thiết kế cải tiến chỉ được nhìn {DEV_OUT.name}.")
    print(f"{TEST_OUT.name} CHỈ dùng 1 lần để báo cáo số liệu cuối cùng.")


if __name__ == "__main__":
    main()