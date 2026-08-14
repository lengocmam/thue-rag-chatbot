"""
chunk_remaining_docs.py
Gộp 4 văn bản đã parse (Luật 09/2026/QH16, TT 91/2026, NĐ 141/2026,
Luật 109/2025/QH15) vào CHUNG 3 file chunks_strategy_A/B/C.jsonl đã có
sẵn của TT 87/2026/TT-BTC -- để toàn bộ 6 văn bản nằm trong 1 tập dữ liệu
thống nhất, dùng chung cho bước embedding/index tiếp theo.

Khó khăn chính: 4 file .noi_dung.json có 3 SCHEMA KHÁC NHAU (do đặc thù
từng văn bản đã xử lý riêng trước đó):
    (1) chuong_list -> dieu_list -> khoan_list -> diem_list
        (Luật 109/2025, TT 91/2026)
    (2) dieu_list -> khoan_list -> diem_list, KHÔNG có Chương
        (NĐ 141/2026)
    (3) dieu_van_xuoi (dict phẳng, KHÔNG chia khoản) + bảng riêng
        (Luật 09/2026/QH16)
Script này viết 1 hàm chunk DÙNG CHUNG cho dict (khác với chunkers.py gốc
vốn dùng dataclass), nhận diện tự động theo schema của từng file.

Cách chạy: python scripts/chunk_remaining_docs.py
"""

import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

SOURCE_FILES = [
    "luat_109_2025_qh15.noi_dung.json",
    "tt_91_2026_ttbtc.noi_dung.json",
    "nd_141_2026_ndcp.noi_dung.json",
    "luat_09_2026_qh16.noi_dung.json",
    "nd_253_2026_ndcp.json",
]

OUT_A = PROCESSED_DIR / "chunks_strategy_A_dieu.jsonl"
OUT_B = PROCESSED_DIR / "chunks_strategy_B_khoan_context.jsonl"
OUT_C = PROCESSED_DIR / "chunks_strategy_C_fixed.jsonl"


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


# ---------- Chuẩn hoá: mọi schema về 1 list[dieu_dict] phẳng (bỏ Chương) ----------

def _flatten_dieu_list(doc: dict) -> list:
    """Trả về list các dieu_dict thống nhất {so, ten, khoan_list,
    text_truc_tiep, chuong_so (có thể None)}, bất kể văn bản gốc có Chương
    hay không, và bất kể lồng trong chuong_list hay dieu_list trực tiếp."""
    flat = []

    if "chuong_list" in doc:  # schema (1)
        for c in doc["chuong_list"]:
            for d in c["dieu_list"]:
                d = dict(d)
                d["chuong_so"] = c["so"]
                flat.append(d)

    elif "dieu_list" in doc:  # schema (2)
        for d in doc["dieu_list"]:
            d = dict(d)
            d["chuong_so"] = None
            flat.append(d)

    elif "dieu_van_xuoi" in doc:  # schema (3) -- Luật 09/2026/QH16
        for so, info in doc["dieu_van_xuoi"].items():
            flat.append({
                "so": so, "ten": info["ten"], "chuong_so": None,
                "khoan_list": [],  # không có cấu trúc khoản trong schema này
                "text_truc_tiep": info["noi_dung"],
            })

    return flat


def _dieu_id_prefix(doc_id: str, chuong_so, dieu_so: str) -> str:
    if chuong_so and chuong_so != "?":
        return f"{doc_id}_chuong{chuong_so}_dieu{dieu_so}"
    return f"{doc_id}_dieu{dieu_so}"


# ---------- Chiến lược A: mỗi chunk = 1 Điều hoàn chỉnh ----------

def chunk_A(doc_id: str, so_hieu: str, flat_dieu: list) -> list:
    chunks = []
    for d in flat_dieu:
        parts = [d["text_truc_tiep"].strip()] if d.get("text_truc_tiep", "").strip() else []
        for k in d.get("khoan_list", []):
            khoan_text = f"{k['so']}. {k['text'].strip()}"
            for p in k.get("diem_list", []):
                khoan_text += f" {p['ky_hieu']}) {p['text'].strip()}"
            parts.append(khoan_text)

        full_text = f"Điều {d['so']}. {d['ten']}\n" + "\n".join(parts)
        chunk_id = _dieu_id_prefix(doc_id, d.get("chuong_so"), d["so"])
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "so_hieu_van_ban": so_hieu,
            "chunking_strategy": "A_dieu",
            "dieu": f"Điều {d['so']}",
            "khoan": None,
            "ten_dieu": d["ten"],
            "text": full_text.strip(),
            "so_luong_token_uoc_tinh": _estimate_tokens(full_text),
        })
    return chunks


# ---------- Chiến lược B: mỗi chunk = 1 Khoản kèm ngữ cảnh Điều ----------

def chunk_B(doc_id: str, so_hieu: str, flat_dieu: list) -> list:
    chunks = []
    for d in flat_dieu:
        if not d.get("khoan_list"):
            # Điều không chia khoản (hoặc schema không hỗ trợ) -> cả Điều là 1 chunk
            full_text = f"Điều {d['so']}. {d['ten']}\n{d.get('text_truc_tiep', '').strip()}"
            chunk_id = _dieu_id_prefix(doc_id, d.get("chuong_so"), d["so"]) + "_full"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "so_hieu_van_ban": so_hieu,
                "chunking_strategy": "B_khoan_context",
                "dieu": f"Điều {d['so']}",
                "khoan": None,
                "ten_dieu": d["ten"],
                "text": full_text.strip(),
                "so_luong_token_uoc_tinh": _estimate_tokens(full_text),
            })
            continue

        for k in d["khoan_list"]:
            context_prefix = f"Điều {d['so']} ({d['ten']}) quy định:\n"
            khoan_text = f"{k['so']}. {k['text'].strip()}"
            for p in k.get("diem_list", []):
                khoan_text += f"\n  {p['ky_hieu']}) {p['text'].strip()}"

            full_text = context_prefix + khoan_text
            chunk_id = _dieu_id_prefix(doc_id, d.get("chuong_so"), d["so"]) + f"_khoan{k['so']}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "so_hieu_van_ban": so_hieu,
                "chunking_strategy": "B_khoan_context",
                "dieu": f"Điều {d['so']}",
                "khoan": f"Khoản {k['so']}",
                "ten_dieu": d["ten"],
                "text": full_text.strip(),
                "so_luong_token_uoc_tinh": _estimate_tokens(full_text),
            })
    return chunks


# ---------- Chiến lược C: fixed-size trên toàn bộ text tái dựng từ cấu trúc ----------

def _reconstruct_full_text(flat_dieu: list) -> str:
    """Dựng lại text liền mạch từ cấu trúc đã parse (không có raw_text gốc
    lưu riêng trong noi_dung.json) -- dùng làm input cho chunking cố định."""
    parts = []
    for d in flat_dieu:
        parts.append(f"Điều {d['so']}. {d['ten']}")
        if d.get("text_truc_tiep", "").strip():
            parts.append(d["text_truc_tiep"].strip())
        for k in d.get("khoan_list", []):
            khoan_text = f"{k['so']}. {k['text'].strip()}"
            for p in k.get("diem_list", []):
                khoan_text += f" {p['ky_hieu']}) {p['text'].strip()}"
            parts.append(khoan_text)
    return "\n".join(parts)


def chunk_C(doc_id: str, so_hieu: str, full_text: str,
            chunk_size_words: int = 350, overlap_words: int = 50) -> list:
    words = full_text.split()
    chunks = []
    start, idx = 0, 0
    step = max(chunk_size_words - overlap_words, 1)

    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": f"{doc_id}_fixed{idx}",
            "doc_id": doc_id,
            "so_hieu_van_ban": so_hieu,
            "chunking_strategy": "C_fixed",
            "dieu": None, "khoan": None, "ten_dieu": None,
            "text": text,
            "so_luong_token_uoc_tinh": _estimate_tokens(text),
        })
        idx += 1
        if end == len(words):
            break
        start += step
    return chunks


# ---------- Xử lý riêng bảng số liệu (không có Khoản/Điểm chuẩn) ----------

def chunk_bang_rieng(doc_id: str, so_hieu: str, doc: dict) -> list:
    """Một số văn bản có bảng số liệu tách riêng khỏi cấu trúc Điều/Khoản
    (Điều 9 Luật 109/2025, Điều 4 Luật 09/2026) -- đóng gói thành 1 chunk
    độc lập dạng mô tả câu để không mất thông tin quan trọng này."""
    chunks = []

    if "bieu_thue_luy_tien_dieu_9" in doc and doc["bieu_thue_luy_tien_dieu_9"]:
        rows = doc["bieu_thue_luy_tien_dieu_9"]
        lines = ["Điều 9. Biểu thuế luỹ tiến từng phần (áp dụng cho thu nhập từ tiền lương, tiền công):"]
        for r in rows:
            khoang = f"{r['muc_thang_tu']} {r['muc_thang_gia_tri']}"
            if r.get("muc_thang_den"):
                khoang += f" đến {r['muc_thang_den']}"
            lines.append(f"- {khoang} triệu đồng/tháng: thuế suất {r['thue_suat_percent']}%")
        text = "\n".join(lines)
        chunks.append({
            "chunk_id": f"{doc_id}_dieu9_bangthue",
            "doc_id": doc_id, "so_hieu_van_ban": so_hieu,
            "chunking_strategy": "A_dieu",  # đối xử như 1 Điều hoàn chỉnh
            "dieu": "Điều 9", "khoan": None, "ten_dieu": "Biểu thuế luỹ tiến từng phần",
            "text": text, "so_luong_token_uoc_tinh": _estimate_tokens(text),
        })

    if "dieu_4_bang_thue_suat" in doc and doc["dieu_4_bang_thue_suat"].get("hang_du_lieu"):
        info = doc["dieu_4_bang_thue_suat"]
        lines = [f"Điều 4. {info['ten']}"]
        for row in info["hang_du_lieu"]:
            muc = "; ".join(f"từ {m['tu_ngay']}: {m['thue_suat_percent']}%" for m in row["muc_thue_suat"])
            lines.append(f"- {row['mo_ta_loai_xe']}: {muc}")
        text = "\n".join(lines)
        chunks.append({
            "chunk_id": f"{doc_id}_dieu4_bangthue",
            "doc_id": doc_id, "so_hieu_van_ban": so_hieu,
            "chunking_strategy": "A_dieu",
            "dieu": "Điều 4", "khoan": None, "ten_dieu": info["ten"],
            "text": text, "so_luong_token_uoc_tinh": _estimate_tokens(text),
        })

    return chunks


# ---------- Main ----------

def append_jsonl(path: Path, records: list):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    total_A, total_B, total_C = 0, 0, 0

    for fname in SOURCE_FILES:
        path = PROCESSED_DIR / fname
        if not path.exists():
            print(f"!! Bỏ qua {fname} -- không tìm thấy file")
            continue

        with open(path, encoding="utf-8") as f:
            doc = json.load(f)

        doc_id = doc["doc_id"]
        so_hieu = doc["so_hieu"]
        print(f"--- {so_hieu} ({fname}) ---")

        flat_dieu = _flatten_dieu_list(doc)
        chunks_A = chunk_A(doc_id, so_hieu, flat_dieu)
        chunks_B = chunk_B(doc_id, so_hieu, flat_dieu)
        full_text = _reconstruct_full_text(flat_dieu)
        chunks_C = chunk_C(doc_id, so_hieu, full_text)
        chunks_bang = chunk_bang_rieng(doc_id, so_hieu, doc)

        chunks_A += chunks_bang  # bảng số liệu tính vào chiến lược A

        append_jsonl(OUT_A, chunks_A)
        append_jsonl(OUT_B, chunks_B)
        append_jsonl(OUT_C, chunks_C)

        print(f"    A: {len(chunks_A)} chunk (gồm {len(chunks_bang)} chunk bảng)")
        print(f"    B: {len(chunks_B)} chunk")
        print(f"    C: {len(chunks_C)} chunk")

        total_A += len(chunks_A)
        total_B += len(chunks_B)
        total_C += len(chunks_C)

    print(f"\nTổng cộng đã thêm: A={total_A}, B={total_B}, C={total_C}")
    print("(Số dòng cũ của TT 87/2026/TT-BTC vẫn giữ nguyên, được nối thêm vào)")


if __name__ == "__main__":
    main()