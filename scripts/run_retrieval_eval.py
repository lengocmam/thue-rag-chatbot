"""
run_retrieval_eval.py
Chạy toàn bộ ma trận thí nghiệm: {BM25, Dense, Hybrid} x {A_dieu, B_khoan_context,
C_fixed} trên bộ ground_truth.jsonl, tính Recall@3/5/10 + MRR, xuất báo cáo.

VẤN ĐỀ KỸ THUẬT QUAN TRỌNG cần giải quyết: ground_truth.jsonl ghi
chunk_id theo 1 chiến lược cụ thể (thường là B), nhưng A/C có chunk_id
HOÀN TOÀN KHÁC cho cùng nội dung (A gộp cả Điều, C cắt cố định không theo
ranh giới Điều/Khoản). Không thể so sánh chunk_id trực tiếp xuyên chiến
lược -- phải đối chiếu qua (doc_id, dieu, khoan) đã lưu sẵn trong
relevant_dieu_khoan của mỗi câu hỏi (xem generate_ground_truth.py).

Cách đối chiếu cho từng chiến lược:
    - A_dieu: 1 chunk A được coi là ĐÚNG nếu cùng (doc_id, dieu) -- vì A
      gộp cả Điều thành 1 chunk, không phân biệt Khoản.
    - B_khoan_context: đối chiếu chính xác (doc_id, dieu, khoan).
    - C_fixed: KHÔNG có ranh giới Điều/Khoản rõ ràng (cắt cố định theo số
      từ) -- coi 1 chunk C là ĐÚNG nếu nó chứa (overlap) ít nhất 1 phần
      đáng kể văn bản trùng với chunk B/A tương ứng. Cách đối chiếu XẤP XỈ
      này là MỘT GIỚI HẠN CẦN GHI RÕ trong khóa luận (không hoàn hảo như 2
      chiến lược kia).

Cách chạy:
    python scripts/run_retrieval_eval.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "indexing"))
from metrics import recall_at_k, reciprocal_rank, aggregate_metrics, paired_bootstrap_test

from build_bm25_index import simple_tokenize

PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"
# Dùng TEST SET GIỮ KÍN (32 câu) làm số liệu CHÍNH THỨC báo cáo trong khóa
# luận -- đồng bộ phương pháp luận với thí nghiệm enrichment (xem
# split_dev_test.py). Dev set (14 câu, có GT001) CHỈ dùng khi cần khám phá
# vấn đề/thiết kế thử nghiệm mới, KHÔNG dùng để báo cáo số liệu cuối cùng.
GT_PATH = PROJECT_ROOT / "data" / "eval" / "ground_truth_test.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "retrieval_comparison.json"

STRATEGIES = ["A_dieu", "B_khoan_context", "C_fixed"]
K_VALUES = [3, 5, 10]

# Có dense/hybrid hay không tuỳ máy đã cài sentence-transformers/faiss chưa
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    DENSE_AVAILABLE = True
except ImportError:
    DENSE_AVAILABLE = False


def load_ground_truth() -> list:
    with open(GT_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_bm25(strategy: str) -> dict:
    with open(INDEX_DIR / f"bm25_{strategy}.pkl", "rb") as f:
        return pickle.load(f)


def load_dense(strategy: str):
    index = faiss.read_index(str(INDEX_DIR / f"faiss_{strategy}.index"))
    with open(INDEX_DIR / f"faiss_{strategy}.meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return index, meta


# ---------- Đối chiếu ground-truth (doc/dieu/khoan) sang chunk_id của TỪNG chiến lược ----------

def resolve_relevant_ids(gt_entry: dict, strategy_chunks: list, strategy: str) -> set:
    relevant_ids = set()
    targets = gt_entry["relevant_dieu_khoan"]

    for chunk in strategy_chunks:
        for t in targets:
            if chunk["doc_id"] != t["doc_id"]:
                continue
            if strategy == "A_dieu":
                if chunk["dieu"] == t["dieu"]:
                    relevant_ids.add(chunk["chunk_id"])
            elif strategy == "B_khoan_context":
                if chunk["dieu"] == t["dieu"] and (t["khoan"] is None or chunk["khoan"] == t["khoan"]):
                    relevant_ids.add(chunk["chunk_id"])
            elif strategy == "C_fixed":
                # Xấp xỉ: coi là liên quan nếu chunk C thuộc CÙNG doc_id --
                # KHÔNG chính xác bằng 2 chiến lược trên, chỉ để có con số
                # tham khảo. Ghi rõ giới hạn này khi báo cáo kết quả.
                if chunk["doc_id"] == t["doc_id"]:
                    relevant_ids.add(chunk["chunk_id"])
    return relevant_ids


# ---------- Chạy retrieval cho 1 câu hỏi ----------

def bm25_rank(bm25_data: dict, query: str) -> list:
    tokens = simple_tokenize(query)
    scores = bm25_data["bm25_model"].get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [bm25_data["chunks"][i]["chunk_id"] for i in order]


def dense_rank(index, meta, embed_model, query: str) -> list:
    q_vec = embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    _, indices = index.search(q_vec, index.ntotal)
    return [meta["chunks"][i]["chunk_id"] for i in indices[0] if i != -1]


def rrf_rank(bm25_ranked: list, dense_ranked: list, k: int = 60) -> list:
    scores = {}
    for rank, cid in enumerate(bm25_ranked):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(dense_ranked):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ---------- Vòng lặp chính ----------

def evaluate_retriever(retriever_name: str, ranked_ids_per_query: list, gt_list: list,
                        relevant_ids_per_query: list) -> dict:
    per_query = []
    for ranked_ids, relevant_ids in zip(ranked_ids_per_query, relevant_ids_per_query):
        row = {f"recall@{k}": recall_at_k(ranked_ids, relevant_ids, k) for k in K_VALUES}
        row["rr"] = reciprocal_rank(ranked_ids, relevant_ids)
        per_query.append(row)
    agg = aggregate_metrics(per_query)
    agg["retriever"] = retriever_name
    agg["per_query_recall@5"] = [r["recall@5"] for r in per_query]  # giữ lại để paired bootstrap sau này
    return agg


def main():
    gt_list = load_ground_truth()
    print(f"Nạp {len(gt_list)} câu hỏi ground-truth.\n")

    all_results = []

    for strategy in STRATEGIES:
        bm25_path = INDEX_DIR / f"bm25_{strategy}.pkl"
        if not bm25_path.exists():
            print(f"!! Bỏ qua {strategy} -- chưa có index BM25 (chạy build_bm25_index.py trước)")
            continue

        print(f"=== Chiến lược chunking: {strategy} ===")
        bm25_data = load_bm25(strategy)
        strategy_chunks = bm25_data["chunks"]

        relevant_ids_per_query = [resolve_relevant_ids(gt, strategy_chunks, strategy) for gt in gt_list]
        n_no_gt = sum(1 for r in relevant_ids_per_query if not r)
        if n_no_gt:
            print(f"  !! CẢNH BÁO: {n_no_gt}/{len(gt_list)} câu hỏi không đối chiếu được "
                  f"chunk liên quan nào trong chiến lược này (bỏ qua khi tính điểm)")

        # --- BM25 ---
        bm25_ranked = [bm25_rank(bm25_data, gt["question"]) for gt in gt_list]
        result_bm25 = evaluate_retriever("BM25", bm25_ranked, gt_list, relevant_ids_per_query)
        result_bm25["strategy"] = strategy
        all_results.append(result_bm25)
        print(f"  BM25   : Recall@5={result_bm25['recall@5']:.3f}  MRR={result_bm25['mrr']:.3f}")

        # --- Dense + Hybrid (nếu có) ---
        faiss_path = INDEX_DIR / f"faiss_{strategy}.index"
        if DENSE_AVAILABLE and faiss_path.exists():
            try:
                index, meta = load_dense(strategy)
                embed_model = SentenceTransformer(meta["model_name"])
                dense_ranked = [dense_rank(index, meta, embed_model, gt["question"]) for gt in gt_list]
                result_dense = evaluate_retriever("Dense", dense_ranked, gt_list, relevant_ids_per_query)
                result_dense["strategy"] = strategy
                all_results.append(result_dense)
                print(f"  Dense  : Recall@5={result_dense['recall@5']:.3f}  MRR={result_dense['mrr']:.3f}")

                hybrid_ranked = [rrf_rank(b, d) for b, d in zip(bm25_ranked, dense_ranked)]
                result_hybrid = evaluate_retriever("Hybrid_RRF", hybrid_ranked, gt_list, relevant_ids_per_query)
                result_hybrid["strategy"] = strategy
                all_results.append(result_hybrid)
                print(f"  Hybrid : Recall@5={result_hybrid['recall@5']:.3f}  MRR={result_hybrid['mrr']:.3f}")
            except Exception as e:
                # KHÔNG để lỗi tải model (VD mất mạng, chưa tải model lần
                # đầu) làm mất luôn kết quả BM25 đã tính được của các
                # chiến lược còn lại -- chỉ bỏ qua Dense/Hybrid, báo rõ lý do.
                print(f"  !! Bỏ qua Dense/Hybrid cho {strategy} -- lỗi: "
                      f"{type(e).__name__}: {str(e)[:150]}")
        else:
            print("  (Bỏ qua Dense/Hybrid -- chưa có index FAISS hoặc chưa cài sentence-transformers)")
        print()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu báo cáo đầy đủ tại {REPORT_PATH}")

    # Bảng tổng hợp cuối
    print("\n=== BẢNG TỔNG HỢP ===")
    print(f"{'Chiến lược':<18}{'Retriever':<14}{'R@3':<8}{'R@5':<8}{'R@10':<8}{'MRR':<8}")
    for r in all_results:
        print(f"{r['strategy']:<18}{r['retriever']:<14}"
              f"{r['recall@3']:<8.3f}{r['recall@5']:<8.3f}{r['recall@10']:<8.3f}{r['mrr']:<8.3f}")

    # --- Kiểm định thống kê: BM25 vs Hybrid, BM25 vs Dense ---
    # CHỈ so sánh trong A_dieu và B_khoan_context -- C_fixed bị loại vì
    # tiêu chí đối chiếu quá lỏng lẻo (gần kịch trần, không phân biệt được),
    # xem ghi chú resolve_relevant_ids().
    print("\n=== KIỂM ĐỊNH THỐNG KÊ (paired bootstrap, so trên recall@5) ===")
    by_key = {(r["strategy"], r["retriever"]): r for r in all_results}

    def _paired_no_none(list_a, list_b):
        """Loại bỏ cặp có None (câu hỏi không đối chiếu được ground-truth
        hợp lệ cho chiến lược đang xét) trước khi đưa vào bootstrap, giữ
        đúng thứ tự ghép cặp giữa 2 danh sách."""
        pairs = [(a, b) for a, b in zip(list_a, list_b) if a is not None and b is not None]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    for strategy in ["A_dieu", "B_khoan_context"]:
        bm25_r = by_key.get((strategy, "BM25"))
        hybrid_r = by_key.get((strategy, "Hybrid_RRF"))
        dense_r = by_key.get((strategy, "Dense"))
        if not (bm25_r and hybrid_r):
            continue

        print(f"\n--- {strategy}: BM25 vs Hybrid_RRF ---")
        a_scores, b_scores = _paired_no_none(bm25_r["per_query_recall@5"], hybrid_r["per_query_recall@5"])
        res = paired_bootstrap_test(a_scores, b_scores)
        print(f"  Mean BM25={res['mean_a']:.3f}  Mean Hybrid={res['mean_b']:.3f}  "
              f"diff={res['observed_diff']:+.3f}  p={res['p_value']:.4f}  (n={len(a_scores)})")
        print(f"  {res['ket_luan']}")

        if dense_r:
            print(f"--- {strategy}: BM25 vs Dense ---")
            a2, b2 = _paired_no_none(bm25_r["per_query_recall@5"], dense_r["per_query_recall@5"])
            res2 = paired_bootstrap_test(a2, b2)
            print(f"  Mean BM25={res2['mean_a']:.3f}  Mean Dense={res2['mean_b']:.3f}  "
                  f"diff={res2['observed_diff']:+.3f}  p={res2['p_value']:.4f}  (n={len(a2)})")
            print(f"  {res2['ket_luan']}")


if __name__ == "__main__":
    main()