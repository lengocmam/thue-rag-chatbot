"""
compare_enrichment_effect.py
So sánh thứ hạng BM25 của các chunk "khó retrieve" (dạng tìm-và-thay)
TRƯỚC và SAU khi enrich ngữ cảnh cấp văn bản -- đo trên TOÀN BỘ 46 câu
ground-truth, không chỉ 1 câu đã biết trước, để tránh kết luận vội từ 1
mẫu (đúng nguyên tắc paired bootstrap đã dùng ở phần retrieval).

Cách chạy:
    python scripts/compare_enrichment_effect.py
"""

import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "src" / "indexing"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "eval"))
from build_bm25_index import simple_tokenize
from metrics import recall_at_k, reciprocal_rank, aggregate_metrics, paired_bootstrap_test

DATA_DIR = PROJECT_ROOT.parent / "data" / "processed"
GT_PATH = PROJECT_ROOT.parent / "data" / "eval" / "ground_truth_test.jsonl"  # CHỈ dùng test set giữ kín


def load_chunks(filename: str) -> list:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_bm25(chunks: list):
    tokenized = [simple_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def rank_ids(bm25, chunks, query: str) -> list:
    scores = bm25.get_scores(simple_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunks[i]["chunk_id"] for i in order]


def resolve_relevant_ids(gt_entry: dict, chunks: list) -> set:
    """Đối chiếu (doc_id, dieu, khoan) -- giống run_retrieval_eval.py,
    nhưng đơn giản hoá cho trường hợp chỉ có 1 chiến lược B."""
    targets = gt_entry["relevant_dieu_khoan"]
    relevant = set()
    for c in chunks:
        for t in targets:
            if c["doc_id"] == t["doc_id"] and c["dieu"] == t["dieu"] and (t["khoan"] is None or c["khoan"] == t["khoan"]):
                relevant.add(c["chunk_id"])
    return relevant


def main():
    gt_list = [json.loads(l) for l in open(GT_PATH, encoding="utf-8")]
    print(f"Nạp {len(gt_list)} câu hỏi ground-truth.\n")

    print("=== 1. [CHỈ MANG TÍNH MINH HỌA — KHÔNG PHẢI SỐ LIỆU ĐÁNH GIÁ] ===")
    print("    Câu hỏi GT001 thuộc DEV SET (đã dùng để phát hiện vấn đề), KHÔNG nằm")
    print("    trong test set giữ kín -- chỉ in ra đây để minh hoạ động cơ thí nghiệm,")
    print("    KHÔNG được trích dẫn số liệu này làm bằng chứng hiệu quả trong báo cáo.")
    original_chunks = load_chunks("chunks_strategy_B_khoan_context.jsonl")
    enriched_chunks = load_chunks("chunks_strategy_B_enriched.jsonl")
    bm25_original = build_bm25(original_chunks)
    bm25_enriched = build_bm25(enriched_chunks)

    cau_hoi = "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh không phải nộp thuế thu nhập cá nhân?"
    target = "nd_141_2026_ndcp_dieu1_khoan1"

    rank_before = rank_ids(bm25_original, original_chunks, cau_hoi)
    rank_after = rank_ids(bm25_enriched, enriched_chunks, cau_hoi)
    pos_before = rank_before.index(target) + 1
    pos_after = rank_after.index(target) + 1
    print(f"  Chunk mục tiêu: {target}")
    print(f"  Hạng TRƯỚC enrich : #{pos_before} / {len(original_chunks)}")
    print(f"  Hạng SAU enrich   : #{pos_after} / {len(enriched_chunks)}")
    print(f"  {'CẢI THIỆN' if pos_after < pos_before else 'KHÔNG cải thiện' if pos_after > pos_before else 'KHÔNG đổi'}")

    print("\n=== 2. ĐÁNH GIÁ CHÍNH THỨC: trên TEST SET GIỮ KÍN (chưa từng dùng để thiết kế enrichment) ===")
    relevant_before = [resolve_relevant_ids(gt, original_chunks) for gt in gt_list]
    relevant_after = [resolve_relevant_ids(gt, enriched_chunks) for gt in gt_list]

    per_query_before, per_query_after = [], []
    for gt, rel_b, rel_a in zip(gt_list, relevant_before, relevant_after):
        ranked_b = rank_ids(bm25_original, original_chunks, gt["question"])
        ranked_a = rank_ids(bm25_enriched, enriched_chunks, gt["question"])
        per_query_before.append({"recall@5": recall_at_k(ranked_b, rel_b, 5), "rr": reciprocal_rank(ranked_b, rel_b)})
        per_query_after.append({"recall@5": recall_at_k(ranked_a, rel_a, 5), "rr": reciprocal_rank(ranked_a, rel_a)})

    agg_before = aggregate_metrics(per_query_before)
    agg_after = aggregate_metrics(per_query_after)
    print(f"  TRƯỚC enrich : Recall@5={agg_before['recall@5']:.3f}  MRR={agg_before['mrr']:.3f}")
    print(f"  SAU enrich   : Recall@5={agg_after['recall@5']:.3f}  MRR={agg_after['mrr']:.3f}")

    print("\n=== 3. Kiểm định thống kê (paired bootstrap, recall@5) ===")
    scores_before = [r["recall@5"] for r in per_query_before if r["recall@5"] is not None]
    scores_after = [r["recall@5"] for r in per_query_after if r["recall@5"] is not None]
    # Đảm bảo ghép cặp đúng thứ tự (chỉ giữ câu có ground-truth hợp lệ ở CẢ 2 bên)
    paired = [(b["recall@5"], a["recall@5"]) for b, a in zip(per_query_before, per_query_after)
              if b["recall@5"] is not None and a["recall@5"] is not None]
    b_scores = [p[0] for p in paired]
    a_scores = [p[1] for p in paired]
    result = paired_bootstrap_test(a_scores, b_scores)  # a=sau, b=trước
    print(f"  Mean sau enrich={result['mean_a']:.3f}  Mean trước enrich={result['mean_b']:.3f}  "
          f"diff={result['observed_diff']:+.3f}  p={result['p_value']:.4f}  (n={len(paired)})")
    print(f"  {result['ket_luan']}")


if __name__ == "__main__":
    main()