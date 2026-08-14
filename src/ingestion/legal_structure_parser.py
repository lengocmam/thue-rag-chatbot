"""
legal_structure_parser.py
Tách văn bản pháp luật (Luật/Nghị định/Thông tư) thành cây phân cấp:
    Văn bản
      └── Điều
            └── Khoản
                  └── Điểm

Nguyên tắc parse: chỉ nhận một dòng là "tiêu đề mới" (Điều/Khoản/Điểm) khi
pattern khớp NGAY TỪ ĐẦU DÒNG (không có khoảng trắng/text đứng trước).
Lý do: các tham chiếu chéo kiểu "...quy định tại khoản 4 Điều 10 của Luật..."
luôn nằm GIỮA câu (không ở đầu dòng), nên sẽ không bị nhận nhầm là tiêu đề
mới -- đã kiểm tra thủ công trên văn bản mẫu Thông tư 87/2026/TT-BTC.

Giới hạn đã biết: nếu một dòng bị ngắt dòng đúng ngay trước một tham chiếu
"Điều X." + hoa chữ cái đầu (hiếm khi xảy ra trong thực tế vì câu tham chiếu
luật thường viết "Điều X của Luật/Nghị định..." chứ không có dấu chấm ngay
sau số Điều), parser có thể nhận nhầm. Nên luôn kiểm tra lại output
(xem hàm `validate_structure`) trước khi dùng làm dữ liệu chính thức.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---- Regex patterns cho từng cấp ----

RE_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
RE_KHOAN = re.compile(r"^(\d+)\.\s+(.*)$")
RE_DIEM = re.compile(r"^([a-zđ])\)\s+(.*)$")

# Đánh dấu điểm dừng: sau dòng này thường là phần chữ ký, nơi nhận... không
# còn là nội dung điều khoản, cần cắt bỏ khỏi cây cấu trúc.
STOP_MARKERS = ["Nơi nhận:", "KT. BỘ TRƯỞNG", "TM. CHÍNH PHỦ"]

# Heuristic xử lý tiêu đề Điều bị PDF ngắt dòng giữa chừng, VD:
#   "Điều 3. Mức thu nhập làm căn cứ ... được áp dụng giảm
#    trừ gia cảnh"
# Nếu từ cuối cùng của tiêu đề (dòng 1) là một trong các từ "lửng" dưới đây
# (giới từ, liên từ, hoặc kết thúc bằng số tham chiếu), coi dòng kế tiếp là
# PHẦN TIẾP THEO của tiêu đề, không phải nội dung thân Điều.
# Giới hạn đã biết: đây là heuristic dựa trên từ điển nhỏ, không phải NLP
# thật sự -- đã kiểm chứng đúng trên 6/6 Điều của TT 87/2026/TT-BTC, nhưng
# cần rà soát thủ công khi áp dụng cho văn bản khác (xem báo cáo QA ở
# scripts/build_all.py).
DANGLING_LAST_WORDS = {
    "của", "về", "cho", "tại", "theo", "và", "hoặc", "là", "được",
    "có", "trong", "mà", "đến", "từ", "để", "với", "giảm", "khoản", "số",
}
RE_TRAILING_DIGIT = re.compile(r"\d+$")


def _is_title_dangling(title: str) -> bool:
    words = title.strip().split()
    if not words:
        return False
    last = words[-1].lower()
    return last in DANGLING_LAST_WORDS or bool(RE_TRAILING_DIGIT.search(last))


@dataclass
class Diem:
    ky_hieu: str          # "a", "b", "c"...
    text: str = ""


@dataclass
class Khoan:
    so: str                # "1", "2"...
    text: str = ""         # phần text KHÔNG thuộc điểm con nào (mở đầu khoản)
    diem_list: list = field(default_factory=list)  # list[Diem]


@dataclass
class Dieu:
    so: str                 # "1", "2"...
    ten: str                # tiêu đề Điều, VD "Phạm vi điều chỉnh"
    khoan_list: list = field(default_factory=list)  # list[Khoan]
    text_truc_tiep: str = ""  # text ngay dưới tên Điều nhưng KHÔNG thuộc khoản nào
    _title_pending: bool = field(default=False, repr=False, compare=False)


def _cut_before_signature_block(lines: list) -> list:
    """Cắt bỏ phần chữ ký/nơi nhận ở cuối văn bản."""
    for i, line in enumerate(lines):
        if any(marker in line for marker in STOP_MARKERS):
            return lines[:i]
    return lines


def parse_legal_structure(text: str) -> list:
    """
    Parse text thô thành list[Dieu].
    Trả về list rỗng nếu không tìm thấy Điều nào (VD: văn bản không có
    cấu trúc Điều/Khoản chuẩn -- cần fallback sang fixed-size chunking).
    """
    lines = text.split("\n")
    lines = _cut_before_signature_block(lines)

    dieu_list = []
    current_dieu = None
    current_khoan = None
    current_diem = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        m_dieu = RE_DIEU.match(line)
        m_khoan = RE_KHOAN.match(line) if current_dieu else None
        m_diem = RE_DIEM.match(line) if current_khoan else None

        if m_dieu:
            ten = m_dieu.group(2).strip()
            current_dieu = Dieu(so=m_dieu.group(1), ten=ten)
            current_dieu._title_pending = _is_title_dangling(ten)
            dieu_list.append(current_dieu)
            current_khoan = None
            current_diem = None
            continue

        # Dòng nối tiếp tiêu đề Điều bị ngắt dòng (chỉ áp dụng 1 lần ngay
        # sau khi Điều vừa được tạo, và chưa có khoản nào bắt đầu)
        if (
            current_dieu is not None
            and current_dieu._title_pending
            and current_khoan is None
            and not m_khoan
        ):
            current_dieu.ten += " " + line
            current_dieu._title_pending = False
            continue

        if m_khoan:
            current_khoan = Khoan(so=m_khoan.group(1), text=m_khoan.group(2).strip())
            current_dieu.khoan_list.append(current_khoan)
            current_diem = None
            continue

        if m_diem:
            current_diem = Diem(ky_hieu=m_diem.group(1), text=m_diem.group(2).strip())
            current_khoan.diem_list.append(current_diem)
            continue

        # Dòng tiếp nối (do PDF ngắt dòng giữa câu) -- nối vào cấp gần nhất
        if current_diem:
            current_diem.text += " " + line
        elif current_khoan:
            current_khoan.text += " " + line
        elif current_dieu:
            current_dieu.text_truc_tiep += " " + line
        # else: text trước Điều 1 (phần "Căn cứ...") -- bỏ qua ở bước này,
        # xử lý riêng nếu cần lưu phần mở đầu.

    return dieu_list


def validate_structure(dieu_list: list) -> dict:
    """Kiểm tra nhanh: số Điều có liên tục không, Điều nào không có khoản
    nào (có thể là lỗi parse hoặc đúng là Điều chỉ có 1 đoạn văn không chia
    khoản, như Điều 2 trong TT 87/2026)."""
    so_dieu = [int(d.so) for d in dieu_list]
    lien_tuc = so_dieu == list(range(so_dieu[0], so_dieu[-1] + 1)) if so_dieu else True
    dieu_khong_co_khoan = [d.so for d in dieu_list if not d.khoan_list]
    return {
        "tong_so_dieu": len(dieu_list),
        "danh_sach_so_dieu": so_dieu,
        "so_dieu_lien_tuc": lien_tuc,
        "dieu_khong_co_khoan": dieu_khong_co_khoan,
    }


def to_dict_list(dieu_list: list) -> list:
    return [asdict(d) for d in dieu_list]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pdf_parser import extract_text_from_pdf

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/87-2026TTBTC.pdf"
    text = extract_text_from_pdf(path)
    dieu_list = parse_legal_structure(text)

    print("=== Kết quả validate ===")
    print(json.dumps(validate_structure(dieu_list), ensure_ascii=False, indent=2))

    print("\n=== Cấu trúc parse được ===")
    for d in dieu_list:
        print(f"Điều {d.so}. {d.ten}  ({len(d.khoan_list)} khoản)")
        for k in d.khoan_list:
            print(f"  - Khoản {k.so}: {k.text[:60]}...  ({len(k.diem_list)} điểm)")