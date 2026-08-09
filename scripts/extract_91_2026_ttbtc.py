"""
extract_91_2026_ttbtc.py
Script RIÊNG, ĐỘC LẬP cho file 91-2026-TT-BTC.pdf.

Đặc điểm khác biệt so với 2 văn bản trước (đòi hỏi logic riêng):
1. Có cấp CHƯƠNG (Chương I..V) phía trên Điều -- 2 văn bản trước không có.
2. Điểm lồng nhiều cấp: không chỉ "a)", "b)" mà còn "d.1)", "d.2.1)" (tối đa
   3 cấp trong văn bản này) -- regex Điểm cũ (1 ký tự) không bao phủ đủ.
3. Có 5 PHỤ LỤC dạng BIỂU MẪU (form) ở cuối, không phải văn xuôi Điều/Khoản:
   - Phụ lục I, II: quy tắc ký hiệu (có nội dung diễn giải thật, đáng trích)
   - Phụ lục III, IV, V: chỉ là danh mục mẫu biểu trống (tên + mã mẫu), hầu
     hết là các trường điền tay (...., ngày.../tháng.../năm...) -- ít giá
     trị nội dung để đưa vào RAG, nên script này CHỈ trích bảng danh mục
     (mẫu số + tên hồ sơ) làm tham chiếu, KHÔNG trích toàn bộ nội dung biểu
     mẫu trống.

Cách chạy: python scripts/extract_91_2026_ttbtc.py
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.pdf_parser import is_noise_line

PDF_PATH = "data/raw/91-2026-TT-BTC.pdf"
OUT_PATH = "data/processed/tt_91_2026_ttbtc.noi_dung.json"

SIGNATURE_WATERMARK_PREFIXES = ("Người ký:", "Email:", "Cơ quan:", "Thời gian ký:")
STOP_MARKER_MAIN_BODY = "Nơi nhận:"     # ranh giới cuối phần văn xuôi chính
PHU_LUC_I_MARKER = "PHỤ LỤC I"          # ranh giới bắt đầu Phụ lục


# ---------- Bước 1: trích text thô, tách phần chính / phụ lục ----------

def extract_raw_text(pdf_path: str) -> str:
    """Trích toàn bộ text bằng PyMuPDF (đã kiểm chứng đọc đúng hơn pdfplumber
    với font của các văn bản Bộ Tài chính trong bộ dữ liệu này)."""
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


def split_main_body_and_phu_luc(full_text: str):
    idx_stop = full_text.find(STOP_MARKER_MAIN_BODY)
    idx_phuluc = full_text.find(PHU_LUC_I_MARKER)
    main_body = full_text[:idx_stop] if idx_stop != -1 else full_text
    phu_luc_text = full_text[idx_phuluc:] if idx_phuluc != -1 else ""
    return main_body, phu_luc_text


# ---------- Bước 2: parse Chương / Điều / Khoản / Điểm (phần chính) ----------

RE_CHUONG = re.compile(r"^Chương\s+([IVXLCDM]+)\s*$")
RE_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
RE_KHOAN = re.compile(r"^(\d+)\.\s+(.*)$")
RE_DIEM = re.compile(r"^([a-zđ](?:\.\d+)*)\)\s+(.*)$")  # bao phủ a) .. d.2.1)

# Tái dùng heuristic tiêu đề Điều bị ngắt dòng (xem legal_structure_parser.py)
DANGLING_LAST_WORDS = {
    "của", "về", "cho", "tại", "theo", "và", "hoặc", "là", "được",
    "có", "trong", "mà", "đến", "từ", "để", "với", "giảm", "khoản", "số",
    "quản", "biện", "hướng",
}


def _is_title_dangling(title: str) -> bool:
    words = title.strip().split()
    return bool(words) and words[-1] in DANGLING_LAST_WORDS


def parse_main_body(main_text: str) -> list:
    """Trả về list[Chuong], mỗi Chuong chứa list[Dieu], mỗi Dieu chứa
    list[Khoan], mỗi Khoan chứa list[Diem] (Diem có thể lồng nhiều cấp
    nhưng được lưu PHẲNG theo ky_hieu, VD 'd.2.1' -- đủ dùng cho chunking,
    không cần dựng cây lồng thật sự)."""
    lines = main_text.split("\n")

    chuong_list = []
    current_chuong = None
    current_dieu = None
    current_khoan = None
    current_diem = None
    title_pending = False
    expecting_chuong_title = False

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
            current_chuong = new_chuong(m_chuong.group(1))
            expecting_chuong_title = True
            current_dieu = current_khoan = current_diem = None
            continue

        if expecting_chuong_title:
            # Dòng ngay sau "Chương X" là tiêu đề chương (in hoa)
            current_chuong["ten"] = line
            expecting_chuong_title = False
            continue

        m_dieu = RE_DIEU.match(line)
        if m_dieu:
            if current_chuong is None:
                current_chuong = new_chuong("?")  # phòng hờ văn bản thiếu Chương
            ten = m_dieu.group(2).strip()
            current_dieu = {"so": m_dieu.group(1), "ten": ten, "khoan_list": [], "text_truc_tiep": ""}
            current_chuong["dieu_list"].append(current_dieu)
            title_pending = _is_title_dangling(ten)
            current_khoan = current_diem = None
            continue

        if title_pending and current_dieu and not current_dieu["khoan_list"] and not current_dieu["text_truc_tiep"]:
            current_dieu["ten"] += " " + line
            title_pending = False
            continue

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

        # Dòng nối tiếp (do PDF ngắt dòng) -- nối vào cấp gần nhất đang mở
        if current_diem:
            current_diem["text"] += " " + line
        elif current_khoan:
            current_khoan["text"] += " " + line
        elif current_dieu:
            current_dieu["text_truc_tiep"] += " " + line

    return chuong_list


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


# ---------- Bước 3: Phụ lục I & II (có nội dung diễn giải thật) ----------

def extract_phu_luc_I_II(phu_luc_text: str) -> dict:
    """Cắt riêng đoạn Phụ lục I (ký hiệu mẫu hóa đơn) và Phụ lục II (ký hiệu
    mẫu chứng từ điện tử) -- giữ nguyên văn xuôi vì đây là quy tắc thật sự
    cần cho RAG (VD: giải mã ký hiệu hóa đơn "1C26TAA" nghĩa là gì)."""
    idx_I = phu_luc_text.find("PHỤ LỤC I")
    idx_II = phu_luc_text.find("PHỤ LỤC II")
    idx_III = phu_luc_text.find("PHỤ LỤC III")

    phu_luc_I = phu_luc_text[idx_I:idx_II].strip() if idx_II != -1 else ""
    phu_luc_II = phu_luc_text[idx_II:idx_III].strip() if idx_III != -1 else ""

    return {"phu_luc_I_raw": phu_luc_I, "phu_luc_II_raw": phu_luc_II}


# ---------- Bước 4: danh mục mẫu biểu Phụ lục III/IV/V (chỉ bảng tham chiếu) ----------

RE_MAU_SO_ALONE = re.compile(r"^\d{2}/[A-ZĐ][A-ZĐ0-9\-]*$")
RE_MAU_SO_ALT_ALONE = re.compile(r"^[A-Z]{2,6}\d{1,3}$")           # VD: "CTT50"
RE_MAU_SO_PREFIX = re.compile(r"^(\d{2}/[A-ZĐ][A-ZĐ0-9\-]*|[A-Z]{2,6}\d{1,3})\s+(\S.*)$")
INDEX_TABLE_HEADERS = ("Tên hồ sơ, mẫu biểu", "Tên loại hóa đơn")


def _extract_one_index_table(lines: list, start_idx: int) -> tuple:
    """Parse 1 bảng danh mục bắt đầu từ start_idx (dòng header cột 2),
    dừng khi gặp dòng 'Mẫu số:' (bắt đầu nội dung form thật).
    Trả về (entries, chỉ_số_dòng_dừng).

    Xử lý 2 kiểu dòng khác nhau tùy hàng trong bảng:
    - Mã mẫu và tên TÁCH RIÊNG 2 dòng (phổ biến nhất)
    - Mã mẫu và tên NẰM CHUNG 1 dòng (VD "01/TB-BSTT-NNT Thông báo...")
      -- xảy ra khi tên hàng trước đó đủ ngắn để không tràn dòng, làm
      PDF gộp luôn mã mẫu kế tiếp vào cùng dòng còn trống."""
    entries = []
    current_ma_mau = None
    current_ten_parts = []

    def flush():
        if current_ma_mau and current_ten_parts:
            entries.append({
                "ma_mau": current_ma_mau,
                "ten_mau_bieu": " ".join(current_ten_parts).strip(),
            })

    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Mẫu số:") or line.startswith("(Kèm theo"):
            flush()
            break
        if line and line != "Mẫu số" and line not in INDEX_TABLE_HEADERS:
            if RE_MAU_SO_ALONE.match(line) or RE_MAU_SO_ALT_ALONE.match(line):
                flush()
                current_ma_mau = line
                current_ten_parts = []
            else:
                m_prefix = RE_MAU_SO_PREFIX.match(line)
                if m_prefix:
                    flush()
                    current_ma_mau = m_prefix.group(1)
                    current_ten_parts = [m_prefix.group(2)]
                elif current_ma_mau:
                    current_ten_parts.append(line)
        i += 1
    return entries, i


def extract_mau_bieu_index(phu_luc_text: str) -> list:
    """Trích danh mục 'Mẫu số -> Tên hồ sơ, mẫu biểu' cho CẢ 3 bảng danh
    mục trong văn bản (đầu Phụ lục III, đầu Phụ lục IV, đầu Phụ lục V).
    Mỗi Phụ lục có 1 bảng riêng, cần quét lần lượt từng bảng thay vì dừng
    lại sau bảng đầu tiên."""
    lines = phu_luc_text.split("\n")
    all_entries = []
    seen = set()
    i = 0
    while i < len(lines):
        if lines[i].strip() in INDEX_TABLE_HEADERS:
            table_entries, i = _extract_one_index_table(lines, i + 1)
            for e in table_entries:
                if e["ma_mau"] not in seen:
                    seen.add(e["ma_mau"])
                    all_entries.append(e)
        else:
            i += 1
    return all_entries


# ---------- Ghép kết quả ----------

def main():
    print(f"[1/5] Trích text thô từ {PDF_PATH} (PyMuPDF)...")
    full_text = extract_raw_text(PDF_PATH)
    print(f"      -> {len(full_text)} ký tự tổng")

    print("[2/5] Tách phần văn xuôi chính (Chương I-V) và Phụ lục...")
    main_body, phu_luc_text = split_main_body_and_phu_luc(full_text)
    print(f"      -> văn xuôi chính: {len(main_body)} ký tự, phụ lục: {len(phu_luc_text)} ký tự")

    print("[3/5] Parse cấu trúc Chương/Điều/Khoản/Điểm...")
    chuong_list = parse_main_body(main_body)
    report = validate_structure(chuong_list)
    print(f"      -> {json.dumps(report, ensure_ascii=False)}")

    print("[4/5] Trích Phụ lục I, II (nội dung diễn giải) + danh mục mẫu biểu III-V...")
    phu_luc_I_II = extract_phu_luc_I_II(phu_luc_text)
    mau_bieu_index = extract_mau_bieu_index(phu_luc_text)
    print(f"      -> Phụ lục I: {len(phu_luc_I_II['phu_luc_I_raw'])} ký tự")
    print(f"      -> Phụ lục II: {len(phu_luc_I_II['phu_luc_II_raw'])} ký tự")
    print(f"      -> {len(mau_bieu_index)} mẫu biểu trong danh mục III/IV/V")

    print(f"[5/5] Ghi kết quả ra {OUT_PATH}...")
    output = {
        "doc_id": "tt_91_2026_ttbtc",
        "so_hieu": "91/2026/TT-BTC",
        "validate_structure": report,
        "chuong_list": chuong_list,
        "phu_luc_I_II": phu_luc_I_II,
        "mau_bieu_index_III_IV_V": mau_bieu_index,
        "ghi_chu": (
            "Phụ lục III, IV, V chỉ được trích ở dạng danh mục (mã mẫu + tên); "
            "nội dung form trống chi tiết KHÔNG được trích vì không có giá trị "
            "nội dung cho RAG. Nếu cần, xử lý bổ sung riêng."
        ),
    }
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Xong.")
    return output


if __name__ == "__main__":
    main()