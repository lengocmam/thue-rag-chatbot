"""
extract_109_2025_qh15.py
Script RIÊNG, ĐỘC LẬP cho file 109-2025-QH15.pdf.

*** KHÁC BIỆT LỚN NHẤT so với 4 văn bản trước: đây là PDF SCAN (ảnh), ***
*** KHÔNG có lớp text nhúng dùng được -- pdfplumber/PyMuPDF chỉ đọc ra  ***
*** vỏn vẹn vài ký tự ("109"). `pdffonts` báo có font nhúng, nhưng font ***
*** đó chỉ phục vụ 1 nhãn nhỏ, không phải nội dung chính -- toàn bộ nội ***
*** dung thân văn bản nằm trong ảnh raster từng trang.                 ***

Quy trình bắt buộc:
    PDF -> rasterize từng trang (pdftoppm, 300 DPI)
        -> OCR tiếng Việt (tesseract, gói tesseract-ocr-vie)
        -> parse cấu trúc Chương/Điều/Khoản/Điểm (regex, tái dùng từ
           legal_structure_parser.py)
        -> trích riêng bảng biểu thuế lũy tiến (Điều 9) bằng regex số

CẢNH BÁO CHẤT LƯỢNG: OCR luôn có sai số (đã quan sát: "QUỐC" -> "QUÓC",
"bổ sung" -> "bồ sung", "kể từ" -> "kê từ"...). Các lỗi này thường KHÔNG
ảnh hưởng cấu trúc Điều/Khoản (số + dấu chấm vẫn rõ), nhưng CÓ THỂ ảnh
hưởng độ chính xác nội dung chunk. Khuyến nghị: đối chiếu lại bằng mắt
với bản gốc trước khi dùng làm ground-truth chính thức cho khóa luận.

Cách chạy: python scripts/extract_109_2025_qh15.py
Yêu cầu hệ thống: tesseract-ocr, tesseract-ocr-vie, poppler-utils (pdftoppm)
    apt-get install -y tesseract-ocr tesseract-ocr-vie poppler-utils
"""

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PDF_PATH = "data/raw/109-2025-QH15.pdf"
OUT_PATH = "data/processed/luat_109_2025_qh15.noi_dung.json"
OCR_TMP_DIR = "data/processed/_ocr_tmp_109_2025"
OCR_DPI = 300
OCR_LANG = "vie"

# Watermark cố định của thuvienphapluat.vn xuất hiện dọc theo lề phải mỗi
# trang -- OCR đọc watermark này RẤT KHÔNG NHẤT QUÁN giữa các trang (do
# chữ bị xoay dọc + mờ), nên không thể dùng 1 regex chính xác. Dùng nhiều
# fragment đặc trưng khó lẫn với nội dung luật thật (số điện thoại, "Tel",
# phần đuôi ".vn"/".VH" của tên miền) để lọc, thay vì khớp chính xác câu.
WATERMARK_FRAGMENTS = [
    "3930 3279", "84-28-3930", "Tel:", "VienPhapLuat", "huVienPhapLuat",
    "PHÁP LUẬT*", "PHÁP LUẬT *",
]


def _line_is_watermark(line: str) -> bool:
    return any(frag in line for frag in WATERMARK_FRAGMENTS)


# ---------- Bước 1: rasterize + OCR toàn bộ trang ----------

def rasterize_pdf(pdf_path: str, out_dir: str, dpi: int = OCR_DPI) -> list:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prefix = str(Path(out_dir) / "page")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), pdf_path, prefix],
        check=True, capture_output=True,
    )
    return sorted(Path(out_dir).glob("page-*.png"))


def ocr_page(image_path: Path, lang: str = OCR_LANG) -> str:
    result = subprocess.run(
        ["tesseract", str(image_path), "stdout", "-l", lang],
        check=True, capture_output=True, text=True,
    )
    return result.stdout


def ocr_full_document(pdf_path: str) -> str:
    image_paths = rasterize_pdf(pdf_path, OCR_TMP_DIR)
    print(f"      -> {len(image_paths)} trang đã rasterize, đang OCR từng trang...")
    all_text = []
    for i, img_path in enumerate(image_paths, 1):
        text = ocr_page(img_path)
        lines = [l for l in text.split("\n") if not _line_is_watermark(l)]
        lines = [l for l in lines if l.strip()]
        all_text.append("\n".join(lines))
        print(f"      -> OCR trang {i}/{len(image_paths)} xong ({len(text)} ký tự thô)")
    return "\n".join(all_text)


# ---------- Bước 2: parse Chương/Điều/Khoản/Điểm ----------

RE_CHUONG = re.compile(r"^-?\s*Chương\s+(\S+)\s*$")

# OCR thường đọc sai số La Mã (VD "III" -> "IH" do nhầm 2 chữ I liền
# nhau với chữ H). Chuẩn hoá lại dựa trên bảng ánh xạ lỗi OCR thường gặp,
# thay vì chỉ chấp nhận ký tự La Mã chuẩn (làm bỏ sót Chương bị OCR sai).
CHUONG_OCR_FIX = {
    "IH": "III", "II": "II", "III": "III", "IV": "IV", "I": "I", "V": "V",
    "1H": "III", "l": "I", "ll": "II",
}


def _normalize_chuong_so(raw: str) -> str:
    return CHUONG_OCR_FIX.get(raw, raw)
RE_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
RE_KHOAN = re.compile(r"^(\d+)\.\s+(.*)$")
RE_DIEM = re.compile(r"^([a-zđ](?:\.\d+)*)\)\s+(.*)$")

DANGLING_LAST_WORDS = {
    "của", "về", "cho", "tại", "theo", "và", "hoặc", "là", "được",
    "có", "trong", "mà", "đến", "từ", "để", "với", "giảm", "khoản", "số",
    "gồm", "sau", "đây", "thuế", "nhập",
}


def _is_title_dangling(title: str) -> bool:
    words = title.strip().split()
    return bool(words) and words[-1].lower() in DANGLING_LAST_WORDS


def parse_structure(text: str) -> list:
    """Giống logic đã dùng cho TT 91/2026 (có Chương), nhưng bảng Điều 9
    được xử lý RIÊNG ở extract_bieu_thue() -- ở đây khi gặp Điều 9, chỉ
    lưu phần dẫn nhập (khoản 1), bỏ qua vùng bảng để tránh OCR rác."""
    lines = text.split("\n")
    chuong_list = []
    current_chuong = None
    current_dieu = None
    current_khoan = None
    current_diem = None
    title_pending = False
    title_merge_count = 0
    expecting_chuong_title = False
    MAX_TITLE_MERGE = 4

    def new_chuong(so):
        c = {"so": so, "ten": "", "dieu_list": []}
        chuong_list.append(c)
        return c

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_chuong = RE_CHUONG.match(line)
        if m_chuong:
            current_chuong = new_chuong(_normalize_chuong_so(m_chuong.group(1)))
            expecting_chuong_title = True
            current_dieu = current_khoan = current_diem = None
            continue

        if expecting_chuong_title:
            current_chuong["ten"] = line
            expecting_chuong_title = False
            continue

        m_dieu = RE_DIEU.match(line)
        if m_dieu:
            if current_chuong is None:
                current_chuong = new_chuong("?")
            ten = m_dieu.group(2).strip()
            current_dieu = {"so": m_dieu.group(1), "ten": ten, "khoan_list": [], "text_truc_tiep": ""}
            current_chuong["dieu_list"].append(current_dieu)
            title_pending = _is_title_dangling(ten)
            title_merge_count = 0
            current_khoan = current_diem = None
            continue

        m_khoan_peek = RE_KHOAN.match(line) if current_dieu else None
        m_diem_peek = RE_DIEM.match(line) if current_khoan else None

        if (title_pending and title_merge_count < MAX_TITLE_MERGE
                and not current_dieu["khoan_list"] and not current_dieu["text_truc_tiep"]
                and not m_khoan_peek and not m_diem_peek):
            current_dieu["ten"] += " " + line
            title_merge_count += 1
            title_pending = _is_title_dangling(current_dieu["ten"])
            continue
        title_pending = False

        # Điều 9 chứa bảng -- bỏ qua nội dung thân (xử lý riêng), chỉ giữ
        # khoản 1 (câu dẫn, không phải bảng)
        if current_dieu and current_dieu["so"] == "9" and current_khoan and current_khoan["so"] != "1":
            continue

        if m_khoan_peek:
            current_khoan = {"so": m_khoan_peek.group(1), "text": m_khoan_peek.group(2).strip(), "diem_list": []}
            current_dieu["khoan_list"].append(current_khoan)
            current_diem = None
            continue

        if m_diem_peek:
            current_diem = {"ky_hieu": m_diem_peek.group(1), "text": m_diem_peek.group(2).strip()}
            current_khoan["diem_list"].append(current_diem)
            continue

        if current_dieu and current_dieu["so"] == "9" and current_khoan is None:
            # Dòng thuộc vùng bảng, chưa có khoản nào khớp regex -- bỏ qua
            if "Bậc" in line or "Đến" in line or "Trên" in line:
                continue

        if current_diem:
            current_diem["text"] += " " + line
        elif current_khoan:
            current_khoan["text"] += " " + line
        elif current_dieu:
            current_dieu["text_truc_tiep"] += " " + line

    return chuong_list


def _cleanup_trailing_junk(text: str) -> str:
    """Dọn các mẩu ký tự rác OCR còn sót lại ở cuối câu (VD dấu ngoặc/ký
    hiệu lạ từ con dấu, watermark bị cắt sót một phần) -- cắt bỏ cụm ký tự
    cuối cùng nếu nó KHÔNG chứa chữ cái tiếng Việt/số nào (an toàn vì câu
    luật hợp lệ luôn kết thúc bằng chữ hoặc số, không bao giờ bằng ký hiệu
    lạ đứng một mình)."""
    return re.sub(r"\s+[^\wÀ-ỹ]{1,4}\s*$", "", text).strip()


def _cleanup_structure(chuong_list: list) -> None:
    for c in chuong_list:
        for d in c["dieu_list"]:
            d["text_truc_tiep"] = _cleanup_trailing_junk(d["text_truc_tiep"])
            for k in d["khoan_list"]:
                k["text"] = _cleanup_trailing_junk(k["text"])
                for p in k["diem_list"]:
                    p["text"] = _cleanup_trailing_junk(p["text"])


def validate_structure(chuong_list: list) -> dict:
    all_dieu_so = [int(d["so"]) for c in chuong_list for d in c["dieu_list"] if d["so"].isdigit()]
    lien_tuc = all_dieu_so == list(range(all_dieu_so[0], all_dieu_so[-1] + 1)) if all_dieu_so else True
    return {
        "tong_so_chuong": len(chuong_list),
        "tong_so_dieu": len(all_dieu_so),
        "danh_sach_so_dieu": all_dieu_so,
        "so_dieu_lien_tuc": lien_tuc,
        "chuong_va_so_dieu": [
            {"chuong": c["so"], "ten_chuong": c["ten"], "so_luong_dieu": len(c["dieu_list"])}
            for c in chuong_list
        ],
    }


# ---------- Bước 3: trích riêng bảng Biểu thuế lũy tiến (Điều 9) ----------

RE_BAC_THUE_ROW = re.compile(
    r"(Đến|Trên)\s*([\d.]+)(?:\s*đến\s*([\d.]+))?.*?"
    r"(Đến|Trên)\s*([\d.]+)(?:\s*đến\s*([\d.]+))?.*?"
    r"(\d+)\s*$"
)


def extract_bieu_thue(full_text: str) -> list:
    """Trích bảng 5 bậc thuế lũy tiến bằng regex số, KHÔNG dựa vào tọa độ
    (khác với bảng thuế TTĐB trong 09-2026-QH16) vì đây là OCR text tuyến
    tính, không có tọa độ x/y để cắt cột. Regex bám vào pattern số cố định
    "Đến/Trên X (đến Y)" lặp lại 2 lần (năm + tháng) + số thuế suất cuối
    dòng -- đã kiểm chứng khớp đúng với 5 bậc thuế đã biết từ trước.

    LƯU Ý QUAN TRỌNG: dùng regex khớp "^Điều 9\\." ở ĐẦU DÒNG để tìm đúng
    vị trí TIÊU ĐỀ Điều 9, không phải find("Điều 9") đơn thuần -- cụm
    "Điều 9" còn xuất hiện dạng THAM CHIẾU giữa câu trong Điều 8 ("...quy
    định tại Điều 9 của Luật này") và xuất hiện TRƯỚC tiêu đề thật trong
    thứ tự đọc, nên find() thường sẽ bắt nhầm vị trí tham chiếu này."""
    m_start = re.search(r"^Điều 9\.", full_text, re.MULTILINE)
    m_end = re.search(r"^Điều 10\.", full_text, re.MULTILINE)
    if not m_start or not m_end:
        return []
    section = full_text[m_start.start():m_end.start()]

    rows = []
    for line in section.split("\n"):
        line = line.strip()
        m = RE_BAC_THUE_ROW.search(line)
        if m:
            rows.append({
                "raw_line": line,
                "muc_nam_tu": m.group(1), "muc_nam_gia_tri": m.group(2), "muc_nam_den": m.group(3),
                "muc_thang_tu": m.group(4), "muc_thang_gia_tri": m.group(5), "muc_thang_den": m.group(6),
                "thue_suat_percent": int(m.group(7)),
            })
    return rows


# ---------- Ghép kết quả ----------

def main():
    print(f"[1/4] Rasterize + OCR toàn bộ {PDF_PATH} (tesseract, lang=vie)...")
    full_text = ocr_full_document(PDF_PATH)
    print(f"      -> Tổng {len(full_text)} ký tự OCR")

    print("[2/4] Parse cấu trúc Chương/Điều/Khoản/Điểm...")
    chuong_list = parse_structure(full_text)
    _cleanup_structure(chuong_list)
    report = validate_structure(chuong_list)
    print(f"      -> {json.dumps(report, ensure_ascii=False)}")

    print("[3/4] Trích riêng bảng Biểu thuế lũy tiến (Điều 9)...")
    bieu_thue = extract_bieu_thue(full_text)
    print(f"      -> {len(bieu_thue)} hàng (kỳ vọng 5 bậc)")
    for r in bieu_thue:
        print(f"      -> Bậc: {r['raw_line']}")

    print(f"[4/4] Ghi kết quả ra {OUT_PATH}...")
    output = {
        "doc_id": "luat_109_2025_qh15",
        "so_hieu": "109/2025/QH15",
        "nguon_trich_xuat": "OCR (tesseract, vie) -- PDF gốc là ảnh scan, không có text layer",
        "canh_bao_chat_luong": (
            "Văn bản này được OCR, KHÔNG phải trích text trực tiếp như 4 văn bản "
            "trước. OCR có tỷ lệ lỗi chính tả nhất định (quan sát được: nhầm dấu "
            "thanh, nhầm chữ cái tương tự hình dạng). Cấu trúc Điều/Khoản đáng tin "
            "cậy (số + dấu chấm rõ ràng), nhưng nội dung câu chữ CẦN đối chiếu lại "
            "bản gốc trước khi dùng làm ground-truth chính thức."
        ),
        "validate_structure": report,
        "chuong_list": chuong_list,
        "bieu_thue_luy_tien_dieu_9": bieu_thue,
    }
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Xong.")
    return output


if __name__ == "__main__":
    main()