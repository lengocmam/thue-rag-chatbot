"""
metrics.py
Các hàm đo lường retrieval: Recall@K, MRR, và kiểm định thống kê paired
bootstrap để so sánh 2 cấu hình (retriever/chunking) có khác biệt CÓ Ý
NGHĨA THỐNG KÊ hay không -- không chỉ nhìn số trung bình rồi kết luận vội
(đúng tinh thần "không giả định, phải đo" xuyên suốt dự án).

Dùng numpy thuần, không phụ thuộc thư viện đánh giá IR chuyên dụng (VD
pytrec_eval) để dễ kiểm tra logic tính toán bằng tay khi cần.
"""

import numpy as np


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """retrieved_ids: list chunk_id đã XẾP HẠNG (rank 0 = tốt nhất).
    relevant_ids: set chunk_id được coi là đúng cho câu hỏi này.
    Trả về 1.0 nếu có ít nhất 1 chunk đúng nằm trong top-k, 0.0 nếu không.
    (Đây là "Recall@K nhị phân" -- phù hợp khi mỗi câu hỏi thường chỉ cần
    tìm ra 1 đoạn chứa câu trả lời, không phải tìm hết TOÀN BỘ đoạn liên
    quan như các bài toán multi-relevant-doc kinh điển.)"""
    if not relevant_ids:
        return None  # không có ground truth hợp lệ -- bỏ qua câu này
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """MRR thành phần cho 1 câu hỏi: 1/hạng của kết quả đúng ĐẦU TIÊN xuất
    hiện trong danh sách xếp hạng. Trả về 0.0 nếu không có chunk đúng nào
    xuất hiện trong toàn bộ danh sách đã truy vấn."""
    if not relevant_ids:
        return None
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def aggregate_metrics(per_query_results: list) -> dict:
    """per_query_results: list[dict] mỗi dict có ít nhất các khóa
    'recall@3', 'recall@5', 'recall@10', 'rr' (reciprocal rank).
    Trả về giá trị trung bình, bỏ qua None (câu hỏi không có ground truth
    hợp lệ cho chiến lược đang xét)."""
    result = {}
    for key in ["recall@3", "recall@5", "recall@10", "rr"]:
        values = [r[key] for r in per_query_results if r.get(key) is not None]
        result[key if key != "rr" else "mrr"] = float(np.mean(values)) if values else None
        result[f"so_cau_hop_le_{key}"] = len(values)
    return result


def paired_bootstrap_test(scores_a: list, scores_b: list, n_iterations: int = 10000,
                           seed: int = 42) -> dict:
    """Kiểm định paired bootstrap: scores_a, scores_b là 2 list điểm số
    CÙNG THỨ TỰ CÂU HỎI (VD recall@5 của cấu hình A so với cấu hình B trên
    từng câu hỏi). Trả về p-value 2 phía + khoảng tin cậy 95% của độ
    chênh lệch trung bình (mean_diff).

    Cách đọc: nếu p-value < 0.05, có thể kết luận sự khác biệt giữa A và B
    có ý nghĩa thống kê (không phải do ngẫu nhiên/nhiễu mẫu nhỏ) -- ĐÚNG
    tinh thần khóa luận: không kết luận "A tốt hơn B" chỉ vì trung bình A
    nhỉnh hơn B một chút trên vài chục câu hỏi."""
    assert len(scores_a) == len(scores_b), "2 danh sách điểm phải cùng số câu hỏi (paired)"
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    n = len(a)
    observed_diff = float(np.mean(a) - np.mean(b))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)  # resample có hoàn lại, giữ ghép cặp
        diffs[i] = np.mean(a[idx]) - np.mean(b[idx])

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # p-value 2 phía: tỷ lệ mẫu bootstrap có dấu NGƯỢC với chênh lệch quan sát được
    p_value = float(np.mean(diffs <= 0) if observed_diff > 0 else np.mean(diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "observed_diff": observed_diff,
        "ci_95_low": float(ci_low), "ci_95_high": float(ci_high),
        "p_value": p_value,
        "ket_luan": (
            "Khác biệt CÓ Ý NGHĨA thống kê (p < 0.05)" if p_value < 0.05
            else "KHÔNG đủ bằng chứng khác biệt có ý nghĩa thống kê (p >= 0.05) "
                 "-- không nên kết luận A tốt/kém hơn B chỉ từ mẫu này"
        ),
    }