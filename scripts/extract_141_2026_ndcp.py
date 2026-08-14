"""
extract_141_2026_ndcp.py
Script RIÊNG, ĐỘC LẬP cho file 141-2026NĐ-CP.pdf.

Đặc điểm văn bản: thuần văn xuôi, không Chương, không bảng -- cấu trúc đơn
giản nhất trong 4 văn bản đã xử lý (Điều > Khoản > Điểm chuẩn). Vẫn viết
script riêng (không dùng chung ingest_one_doc.py) để giữ đúng nguyên tắc
"mỗi văn bản 1 script độc lập" của dự án, và vì văn bản này có một điểm
CẦN XỬ LÝ CẨN THẬN: Điều 1 khoản 1 là một câu "tìm-và-thay" áp dụng cho
NHIỀU điều khoản của văn bản KHÁC (Nghị định 68/2026/NĐ-CP) -- không phải
nội dung tự thân, nên cần tách riêng để không làm chunk bị hiểu sai.

Cách chạy: python scripts/extract_141_2026_ndcp.py
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.pdf_parser import is_noise_line

PDF_PATH = "data/raw/141-2026NĐ-CP.pdf"
OUT_PATH = "data/processed/nd_141_2026_ndcp.noi_dung.json"

SIGNATURE_WATERMARK_PREFIXES = ("Người ký:", "Email:", "Cơ quan:", "Thời gian ký:")
STOP_MARKER = "Nơi nhận:"


# ---------- Bước 1: trích text thô bằng PyMuPDF ----------

def extract_raw_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    lines_out = []
    for page in doc:
        for line in page.get_text().split("\n"):
            if is_noise_line(line):
                continue
            if line.strip().startswith(SIGNATURE_WATERMARK_PREFIXES):
                continue
            lines_out.append(line)
    doc.close()
    text = "\n".join(lines_out)
    idx = text.find(STOP_MARKER)
    return text[:idx] if idx != -1 else text


# ---------- Bước 2: parse Điều / Khoản / Điểm ----------

RE_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
RE_KHOAN = re.compile(r"^(\d+)\.\s+(.*)$")
RE_DIEM = re.compile(r"^([a-zđ](?:\.\d+)*)\)\s+(.*)$")

DANGLING_LAST_WORDS = {
    "của", "về", "cho", "tại", "theo", "và", "hoặc", "là", "được",
    "có", "trong", "mà", "đến", "từ", "để", "với", "giảm", "khoản", "số",
    "nghị", "định", "quản", "chính", "sách", "thu", "nhập", "doanh",
}


def _is_title_dangling(title: str) -> bool:
    words = title.strip().split()
    return bool(words) and words[-1].lower() in DANGLING_LAST_WORDS


def parse_structure(text: str) -> list:
    lines = text.split("\n")
    dieu_list = []
    current_dieu = None
    current_khoan = None
    current_diem = None
    title_pending = False
    title_merge_count = 0
    MAX_TITLE_MERGE_LINES = 4  # chặn gộp vô hạn nếu heuristic đoán sai

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_dieu = RE_DIEU.match(line)
        if m_dieu:
            ten = m_dieu.group(2).strip()
            current_dieu = {"so": m_dieu.group(1), "ten": ten, "khoan_list": [], "text_truc_tiep": ""}
            dieu_list.append(current_dieu)
            title_pending = _is_title_dangling(ten)
            title_merge_count = 0
            current_khoan = current_diem = None
            continue

        # QUAN TRỌNG: luôn kiểm tra Khoản/Điểm TRƯỚC khi coi là phần tiếp
        # của tiêu đề -- nếu không, một tiêu đề "may mắn" kết thúc bằng từ
        # lửng (VD "...kinh doanh") có thể nuốt luôn cả Khoản 1 thật sự.
        m_khoan_peek = RE_KHOAN.match(line) if current_dieu else None
        m_diem_peek = RE_DIEM.match(line) if current_khoan else None

        if (title_pending and title_merge_count < MAX_TITLE_MERGE_LINES
                and not current_dieu["khoan_list"] and not current_dieu["text_truc_tiep"]
                and not m_khoan_peek and not m_diem_peek):
            current_dieu["ten"] += " " + line
            title_merge_count += 1
            title_pending = _is_title_dangling(current_dieu["ten"])
            continue
        title_pending = False

        m_khoan = RE_KHOAN.match(line) if current_dieu else None
        if m_khoan:
            current_khoan = {"so": m_khoan.group(1), "text": m_khoan.group(2).strip(), "diem_list": []}
            current_dieu["khoan_list"].append(current_khoan)
            current_diem = None
            continue

        m_diem = RE_DIEM.match(line) if current_khoan else None
        if m_diem:
            current_diem = {"ky_hieu": m_diem.group(1), "text": m_diem.group(2).strip()}
            current_khoan["diem_list"].append(current_diem)
            continue

        if current_diem:
            current_diem["text"] += " " + line
        elif current_khoan:
            current_khoan["text"] += " " + line
        elif current_dieu:
            current_dieu["text_truc_tiep"] += " " + line

    return dieu_list


def validate_structure(dieu_list: list) -> dict:
    so_dieu = [int(d["so"]) for d in dieu_list]
    lien_tuc = so_dieu == list(range(so_dieu[0], so_dieu[-1] + 1)) if so_dieu else True
    return {
        "tong_so_dieu": len(dieu_list),
        "danh_sach_so_dieu": so_dieu,
        "so_dieu_lien_tuc": lien_tuc,
        "dieu_khong_co_khoan": [d["so"] for d in dieu_list if not d["khoan_list"]],
    }


# ---------- Bước 3: đánh dấu riêng nội dung "tìm-và-thay" trong Điều 1 khoản 1 ----------

RE_TIM_VA_THAY = re.compile(r"Sửa đổi cụm từ\s*[“\"](.+?)[”\"]\s*thành\s*[“\"](.+?)[”\"]")


def extract_tim_va_thay(dieu_list: list) -> list:
    """Điều 1 khoản 1 của văn bản này là dạng 'tìm-và-thay' (không phải nội
    dung mới, mà là lệnh thay thế cụm từ trong văn bản khác) -- trích riêng
    ra để không bị hiểu nhầm là nội dung tự thân khi chunk."""
    results = []
    for d in dieu_list:
        for k in d["khoan_list"]:
            for m in RE_TIM_VA_THAY.finditer(k["text"]):
                results.append({
                    "dieu": f"Điều {d['so']}", "khoan": f"Khoản {k['so']}",
                    "cum_tu_cu": m.group(1), "cum_tu_moi": m.group(2),
                    "context_day_du": k["text"],
                })
    return results


# ---------- Ghép kết quả ----------

def main():
    print(f"[1/4] Trích text thô từ {PDF_PATH} (PyMuPDF)...")
    raw_text = extract_raw_text(PDF_PATH)
    print(f"      -> {len(raw_text)} ký tự")

    print("[2/4] Parse cấu trúc Điều/Khoản/Điểm...")
    dieu_list = parse_structure(raw_text)
    report = validate_structure(dieu_list)
    print(f"      -> {json.dumps(report, ensure_ascii=False)}")

    print("[3/4] Trích các lệnh 'tìm-và-thay' (Điều 1 khoản 1)...")
    tim_va_thay = extract_tim_va_thay(dieu_list)
    for t in tim_va_thay:
        print(f"      -> '{t['cum_tu_cu']}' -> '{t['cum_tu_moi']}'")

    print(f"[4/4] Ghi kết quả ra {OUT_PATH}...")
    output = {
        "doc_id": "nd_141_2026_ndcp",
        "so_hieu": "141/2026/NĐ-CP",
        "validate_structure": report,
        "dieu_list": dieu_list,
        "lenh_tim_va_thay": tim_va_thay,
        "ghi_chu": (
            "Điều 1 khoản 1 chứa lệnh thay '500 triệu đồng' -> '01 tỷ đồng' "
            "áp dụng cho nhiều Điều của Nghị định 68/2026/NĐ-CP. Đây là nguồn "
            "chứa NGƯỠNG DOANH THU CỤ THỂ mà các văn bản luật gốc chỉ nói "
            "'Chính phủ quy định' không nêu số -- quan trọng cho retrieval."
        ),
    }
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Xong.")
    return output


if __name__ == "__main__":
    main()