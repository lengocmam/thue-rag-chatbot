"""
extract_09_2026_qh16.py
Script RIÊNG, ĐỘC LẬP cho file 09-2026-QH16.pdf.

Lý do tách riêng thay vì dùng chung pipeline `legal_structure_parser.py`:
văn bản này có Điều 4 chứa BẢNG (Biểu thuế TTĐB xe điện), không phải văn
xuôi thuần -- regex Điều/Khoản/Điểm không xử lý được bảng.

Cách trích bảng: pdfplumber.extract_tables() KHÔNG nhận diện được bảng này
(PDF không có đường kẻ rõ), nên phải tự dựng bảng dựa vào TỌA ĐỘ CHỮ (x0):
  - Cột trái (mô tả xe): các từ có x0 < COL_SPLIT_X
  - Cột phải (thuế suất): các từ có x0 >= COL_SPLIT_X
  - Một HÀNG MỚI trong bảng bắt đầu khi cột trái có từ "Xe" ở x0 nằm trong
    khoảng ROW_START_X_RANGE (~131-134) -- đây là vị trí thụt đầu dòng của
    each dòng mở đầu 1 hàng; các dòng nối tiếp trong cùng hàng bắt đầu bằng
    dấu "-" ở x0 nhỏ hơn (~122).
  - Đã kiểm chứng thủ công: đúng cho cả 4 hàng dữ liệu của bảng này.

Giới hạn: các hằng số tọa độ (COL_SPLIT_X, ROW_START_X_RANGE) được rút ra
từ chính file PDF này -- nếu áp dụng cho PDF khác có layout khác, CẦN kiểm
tra lại bằng đoạn code debug ở cuối file (in tọa độ x0/top của từng từ).
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.pdf_parser import is_noise_line

PDF_PATH = "data/raw/09-2026-QH16.pdf"
OUT_PATH = "data/processed/luat_09_2026_qh16.noi_dung.json"

COL_SPLIT_X = 420          # x0 >= 420 -> cột thuế suất; < 420 -> cột mô tả
TABLE_TOP = 320             # top bắt đầu vùng bảng (ngay dưới dòng "* Xe có...")
TABLE_BOTTOM = 495          # top kết thúc vùng bảng (trước "Điều 5.")
TABLE_PAGE_INDEX = 1        # bảng nằm ở trang 2 (index 1) của PDF này


# ---------- Phần 1: Điều 1, 2, 3, 5 -- văn xuôi, parse bằng regex đơn giản ----------

def extract_text_pymupdf(pdf_path: str) -> str:
    """Dùng PyMuPDF thay vì pdfplumber để trích text các trang văn xuôi.
    LÝ DO: pdfplumber giải mã SAI cmap font ở một số đoạn của file này
    (đã kiểm chứng: đoạn mở đầu Điều 2 bị xáo trộn ký tự hoàn toàn khi
    dùng pdfplumber, trong khi PyMuPDF đọc đúng 100%). Đây là lỗi tương
    thích font của thư viện, không phải lỗi nội dung file PDF gốc."""
    # Watermark chữ ký điện tử ở góc trên văn bản -- PyMuPDF đôi khi chèn
    # lạc vào giữa dòng đọc chính, cần lọc riêng (khác với NOISE_PATTERNS
    # trong pdf_parser.py vốn chỉ xử lý rác do trình duyệt in PDF).
    SIGNATURE_WATERMARK_PREFIXES = ("Người ký:", "Email:", "Cơ quan:", "Thời gian ký:")

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
    return "\n".join(lines_out)


RE_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")

# Tái dùng heuristic xử lý tiêu đề Điều bị PDF ngắt dòng giữa chừng
# (xem giải thích chi tiết trong src/ingestion/legal_structure_parser.py)
DANGLING_LAST_WORDS = {
    "của", "về", "cho", "tại", "theo", "và", "hoặc", "là", "được",
    "có", "trong", "mà", "đến", "từ", "để", "với", "giảm", "khoản", "số", "Thuế",
}


def _is_title_dangling(title: str) -> bool:
    words = title.strip().split()
    if not words:
        return False
    return words[-1] in DANGLING_LAST_WORDS


def extract_prose_dieu(text: str) -> dict:
    """Tách các Điều dạng văn xuôi (1, 2, 3, 5) thành dict {so_dieu: {ten, noi_dung}}.
    Điều 4 (dạng bảng) sẽ bị bỏ qua ở đây, xử lý riêng ở phần 2."""
    lines = text.split("\n")
    result = {}
    current_so = None
    current_ten = None
    title_pending = False
    buffer = []

    def flush():
        if current_so is not None:
            result[current_so] = {
                "ten": current_ten.strip(),
                "noi_dung": " ".join(buffer).strip(),
            }

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = RE_DIEU.match(line)
        if m:
            flush()
            current_so = m.group(1)
            current_ten = m.group(2)
            title_pending = _is_title_dangling(current_ten)
            buffer = []
            continue
        if title_pending and not buffer:
            current_ten += " " + line
            title_pending = False
            continue
        if current_so == "4":
            # Đang trong vùng Điều 4 (bảng) -- bỏ qua ở nhánh văn xuôi này
            continue
        if current_so is not None:
            buffer.append(line)

    flush()
    result.pop("4", None)  # loại Điều 4 khỏi kết quả văn xuôi, xử lý riêng
    return result


# ---------- Phần 2: Điều 4 -- bảng, parse bằng tọa độ chữ ----------

RE_RATE = re.compile(r"T[ừùu]\s*(\d{2}/\d{2}/\d{4}):\s*(\d+)")


def extract_table_dieu4(pdf_path: str) -> list:
    """Trích bảng Biểu thuế TTĐB xe điện bằng cách CẮT RIÊNG 2 vùng cột
    (mô tả xe / thuế suất) rồi để pdfplumber.extract_text() tự gom dòng
    trong phạm vi hẹp của từng cột -- cách này ổn định hơn nhiều so với
    tự viết thuật toán chaining theo top, vì trong bảng này khoảng cách
    giữa các dòng thật (~10pt) gần bằng độ lệch baseline do dấu tiếng
    Việt (~4-5pt), khiến clustering thủ công dễ gộp nhầm dòng."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[TABLE_PAGE_INDEX]
        left_col = page.crop((0, TABLE_TOP, COL_SPLIT_X, TABLE_BOTTOM))
        right_col = page.crop((COL_SPLIT_X, TABLE_TOP, page.width, TABLE_BOTTOM))
        left_text = left_col.extract_text() or ""
        right_text = right_col.extract_text() or ""

    # ---- Cột trái: tách thành 4 hàng, mỗi hàng bắt đầu bằng "Xe" ----
    left_lines = [l for l in left_text.split("\n") if l.strip()]
    row_descriptions = []
    current = []
    for line in left_lines:
        line = line.strip()
        if line.startswith("*"):
            continue  # bỏ dòng phụ đề "* Xe có gắn động cơ..."
        if line.startswith("Xe"):
            if current:
                row_descriptions.append(" ".join(current))
            current = [line]
        elif line == "-":
            continue  # bỏ dấu gạch đầu dòng đứng riêng 1 dòng
        else:
            current.append(line.lstrip("- ").strip())
    if current:
        row_descriptions.append(" ".join(current))
    row_descriptions = [re.sub(r"\s+", " ", d).strip() for d in row_descriptions]

    # ---- Cột phải: mỗi 2 dòng "Từ dd/mm/yyyy: N" (+ dấu "-" xen giữa) = 1 hàng ----
    rate_matches = RE_RATE.findall(right_text)  # list[(ngay, suat)] theo thứ tự đọc
    rates_per_row = [rate_matches[i:i + 2] for i in range(0, len(rate_matches), 2)]

    parsed_rows = []
    for mo_ta, rates in zip(row_descriptions, rates_per_row):
        parsed_rows.append({
            "mo_ta_loai_xe": mo_ta,
            "muc_thue_suat": [
                {"tu_ngay": ngay, "thue_suat_percent": int(suat)}
                for ngay, suat in rates
            ],
        })

    return parsed_rows


# ---------- Ghép kết quả ----------

def main():
    print(f"[1/3] Trích văn xuôi (Điều 1, 2, 3, 5) từ {PDF_PATH} (PyMuPDF)...")
    raw_text = extract_text_pymupdf(PDF_PATH)
    dieu_van_xuoi = extract_prose_dieu(raw_text)
    for so, d in dieu_van_xuoi.items():
        print(f"      Điều {so}: {d['ten'][:70]}")

    print(f"\n[2/3] Trích bảng Điều 4 bằng tọa độ chữ...")
    bang_dieu4 = extract_table_dieu4(PDF_PATH)
    print(f"      -> {len(bang_dieu4)} hàng dữ liệu")
    for row in bang_dieu4:
        print(f"      - {row['mo_ta_loai_xe'][:60]}... | {row['muc_thue_suat']}")

    print(f"\n[3/3] Ghi kết quả ra {OUT_PATH}...")
    output = {
        "doc_id": "luat_09_2026_qh16",
        "so_hieu": "09/2026/QH16",
        "dieu_van_xuoi": dieu_van_xuoi,
        "dieu_4_bang_thue_suat": {
            "ten": "Sửa đổi, bổ sung quy định về xe có gắn động cơ dưới 24 chỗ "
                   "chạy bằng pin tại điểm g mục 4 phần I của Biểu thuế tiêu thụ "
                   "đặc biệt quy định tại khoản 1 Điều 8 của Luật Thuế tiêu thụ đặc biệt",
            "hang_du_lieu": bang_dieu4,
        },
    }
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Xong.")
    return output


if __name__ == "__main__":
    main()