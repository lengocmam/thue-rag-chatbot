"""
ingest_one_doc.py
Chạy toàn bộ pipeline ingest cho 1 văn bản: PDF -> text -> cấu trúc Điều/Khoản
-> 3 chiến lược chunk -> lưu JSONL vào data/processed/.

Cách chạy:
    python scripts/ingest_one_doc.py data/raw/87-2026TTBTC.pdf
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_parser import extract_text_from_pdf
from src.ingestion.legal_structure_parser import parse_legal_structure, validate_structure
from src.ingestion.chunkers import (
    chunk_strategy_A_dieu,
    chunk_strategy_B_khoan_context,
    chunk_strategy_C_fixed,
)


def load_doc_metadata(pdf_path: Path) -> dict:
    meta_path = pdf_path.parent / (pdf_path.stem + ".meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def append_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(pdf_path_str: str):
    pdf_path = Path(pdf_path_str)
    doc_meta = load_doc_metadata(pdf_path)
    doc_id = doc_meta.get("doc_id", pdf_path.stem)
    so_hieu = doc_meta.get("so_hieu", pdf_path.stem)

    print(f"[1/4] Trích xuất text từ {pdf_path.name}...")
    raw_text = extract_text_from_pdf(str(pdf_path))
    print(f"      -> {len(raw_text)} ký tự")

    print("[2/4] Parse cấu trúc Điều/Khoản/Điểm...")
    dieu_list = parse_legal_structure(raw_text)
    report = validate_structure(dieu_list)
    print(f"      -> {json.dumps(report, ensure_ascii=False)}")
    if not dieu_list:
        print("      !! CẢNH BÁO: không tìm thấy Điều nào -- chỉ chunk theo "
              "chiến lược C (fixed-size) cho văn bản này.")

    print("[3/4] Sinh chunks theo 3 chiến lược...")
    chunks_A = chunk_strategy_A_dieu(dieu_list, doc_id, so_hieu) if dieu_list else []
    chunks_B = chunk_strategy_B_khoan_context(dieu_list, doc_id, so_hieu) if dieu_list else []
    chunks_C = chunk_strategy_C_fixed(raw_text, doc_id, so_hieu)
    print(f"      -> A (theo Điều): {len(chunks_A)} chunks")
    print(f"      -> B (theo Khoản + ngữ cảnh): {len(chunks_B)} chunks")
    print(f"      -> C (fixed-size baseline): {len(chunks_C)} chunks")

    print("[4/4] Ghi ra data/processed/...")
    out_dir = Path("data/processed")
    append_jsonl(out_dir / "chunks_strategy_A_dieu.jsonl", chunks_A)
    append_jsonl(out_dir / "chunks_strategy_B_khoan_context.jsonl", chunks_B)
    append_jsonl(out_dir / "chunks_strategy_C_fixed.jsonl", chunks_C)

    # Báo cáo QA nhỏ để bạn tự soát lại bằng mắt trước khi dùng cho embedding
    qa_report = {
        "doc_id": doc_id,
        "file_goc": pdf_path.name,
        "so_ky_tu_raw_text": len(raw_text),
        "validate_structure": report,
        "so_chunk_A": len(chunks_A),
        "so_chunk_B": len(chunks_B),
        "so_chunk_C": len(chunks_C),
    }
    with open(out_dir / f"{doc_id}.qa_report.json", "w", encoding="utf-8") as f:
        json.dump(qa_report, f, ensure_ascii=False, indent=2)

    print(f"\nXong. Xem báo cáo QA tại data/processed/{doc_id}.qa_report.json")
    return chunks_A, chunks_B, chunks_C


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/87-2026TTBTC.pdf"
    main(path)