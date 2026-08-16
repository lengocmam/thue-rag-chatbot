"""
train_offtopic_gate.py
Huấn luyện classifier nhẹ (Logistic Regression) để lọc câu hỏi off-topic
TRƯỚC khi gọi LLM. So sánh 2 cách trích đặc trưng đầu vào -- ĐÚNG tinh
thần "đo mới biết, không giả định" xuyên suốt dự án:
    (A) TF-IDF -- nhẹ, nhanh, không cần tải model, baseline hợp lý.
    (B) Sentence embedding (vietnamese-sbert) -- nặng hơn, cần tải model,
        nhưng có thể nắm ngữ nghĩa tốt hơn TF-IDF thuần từ khóa.

Vì dữ liệu huấn luyện NHỎ (~90 mẫu), dùng k-fold cross-validation (thay vì
1 lần train/test split) để có ước lượng đáng tin cậy hơn, không phụ thuộc
may rủi của 1 lần chia tập.

Cách chạy:
    python scripts/train_offtopic_gate.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "eval" / "offtopic_training_data.jsonl"
MODEL_DIR = PROJECT_ROOT / "models"
N_FOLDS = 5
RANDOM_STATE = 42


def load_data():
    texts, labels = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(r["label"])
    return texts, np.array(labels)


def evaluate_cv(pipeline, texts, labels, name: str) -> dict:
    """Đánh giá bằng Stratified K-Fold CV -- 'stratified' để mỗi fold vẫn
    giữ đúng tỷ lệ 2 lớp như tập gốc (quan trọng vì dữ liệu hơi lệch lớp)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    pred = cross_val_predict(pipeline, texts, labels, cv=skf)
    proba = cross_val_predict(pipeline, texts, labels, cv=skf, method="predict_proba")[:, 1]

    metrics = {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred),
        "recall": recall_score(labels, pred),
        "f1": f1_score(labels, pred),
        "auc_roc": roc_auc_score(labels, proba),
    }
    cm = confusion_matrix(labels, pred)

    print(f"\n=== {name} (Stratified {N_FOLDS}-Fold CV) ===")
    for k, v in metrics.items():
        print(f"  {k:<10}: {v:.4f}")
    print(f"  Confusion matrix (hàng=thật, cột=dự đoán, thứ tự [off-topic, in-domain]):")
    print(f"    {cm}")

    # Liệt kê các trường hợp dự đoán SAI -- quan trọng để phân tích lỗi
    # (đúng tinh thần: không chỉ báo cáo số điểm mà phải hiểu lỗi ở đâu)
    wrong_idx = np.where(pred != labels)[0]
    if len(wrong_idx) > 0:
        print(f"  Các câu dự đoán SAI ({len(wrong_idx)} câu):")
        for i in wrong_idx:
            print(f"    - '{texts[i][:70]}...' (thật={labels[i]}, dự đoán={pred[i]}, proba={proba[i]:.3f})")

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def train_final_model(pipeline, texts, labels):
    """Train lại trên TOÀN BỘ dữ liệu (không chỉ 1 fold) để lấy model cuối
    cùng dùng thật -- k-fold ở trên chỉ để ĐÁNH GIÁ, không phải model dùng
    để triển khai."""
    pipeline.fit(texts, labels)
    return pipeline


def main():
    texts, labels = load_data()
    print(f"Nạp {len(texts)} mẫu ({labels.sum()} trong phạm vi / {len(labels) - labels.sum()} ngoài phạm vi).")

    # --- Phương án A: TF-IDF + Logistic Regression ---
    tfidf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    metrics_tfidf = evaluate_cv(tfidf_pipeline, texts, labels, "TF-IDF + Logistic Regression")

    # --- Phương án B: Sentence Embedding + Logistic Regression ---
    # Cần sentence-transformers -- nếu chưa cài/chưa tải được model thì bỏ
    # qua phương án này, không làm hỏng kết quả phương án A đã có.
    metrics_embedding = None
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.base import BaseEstimator, TransformerMixin

        class EmbeddingTransformer(BaseEstimator, TransformerMixin):
            """Bọc SentenceTransformer thành 1 bước trong sklearn Pipeline
            -- để dùng chung cơ chế cross_val_predict như TF-IDF."""
            def __init__(self, model_name="keepitreal/vietnamese-sbert"):
                self.model_name = model_name

            def fit(self, X, y=None):
                self.model_ = SentenceTransformer(self.model_name)
                return self

            def transform(self, X):
                return self.model_.encode(list(X), normalize_embeddings=True, show_progress_bar=False)

        embedding_pipeline = Pipeline([
            ("embed", EmbeddingTransformer()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ])
        metrics_embedding = evaluate_cv(embedding_pipeline, texts, labels, "Vietnamese-SBERT Embedding + Logistic Regression")
    except Exception as e:
        print(f"\n!! Bỏ qua phương án Embedding -- lỗi: {type(e).__name__}: {str(e)[:150]}")

    # --- So sánh và chọn model tốt hơn để triển khai ---
    print("\n=== SO SÁNH 2 PHƯƠNG ÁN ===")
    print(f"TF-IDF     : AUC-ROC={metrics_tfidf['auc_roc']:.4f}  F1={metrics_tfidf['f1']:.4f}")
    if metrics_embedding:
        print(f"Embedding  : AUC-ROC={metrics_embedding['auc_roc']:.4f}  F1={metrics_embedding['f1']:.4f}")
        best_name = "embedding" if metrics_embedding["auc_roc"] > metrics_tfidf["auc_roc"] else "tfidf"
    else:
        best_name = "tfidf"
    print(f"-> Chọn triển khai: {best_name.upper()} "
          f"(LƯU Ý: chênh lệch nhỏ trên ~90 mẫu KHÔNG đủ để kết luận chắc "
          f"chắn phương án nào tốt hơn về bản chất -- cần thêm dữ liệu và "
          f"kiểm định thống kê paired bootstrap như đã làm ở phần retrieval "
          f"nếu muốn kết luận nghiêm túc cho khóa luận.)")

    # --- Lưu model cuối cùng để dùng trong RAG pipeline ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_pipeline = tfidf_pipeline if best_name == "tfidf" else embedding_pipeline
    final_pipeline = train_final_model(final_pipeline, texts, labels)
    with open(MODEL_DIR / "offtopic_gate.pkl", "wb") as f:
        pickle.dump({"pipeline": final_pipeline, "feature_type": best_name}, f)
    print(f"\nĐã lưu model cuối tại {MODEL_DIR / 'offtopic_gate.pkl'}")

    # Lưu báo cáo đầy đủ
    report = {"tfidf": metrics_tfidf, "embedding": metrics_embedding, "chosen": best_name}
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "offtopic_gate_comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()