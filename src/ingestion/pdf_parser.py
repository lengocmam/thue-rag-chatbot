"""
pdf_parser.py
Trích xuất text thô từ file PDF văn bản pháp luật.

Ghi chú: nhiều file PDF luật được người dùng "In ra PDF" từ trình duyệt
(thuvienphapluat.vn, vbpl.vn...) nên thường có rác header/footer dạng:
  - "21:26 8/8/26 about:blank"   (timestamp + tên tab trình duyệt)
  - "about:blank 1/5"            (footer số trang do trình duyệt chèn)
Các dòng này KHÔNG phải nội dung văn bản, cần loại bỏ trước khi parse
cấu trúc Điều/Khoản, nếu không sẽ làm gãy logic ghép dòng.
"""

import re
from pathlib import Path

import pdfplumber

# Các pattern rác thường gặp khi PDF được xuất từ trình duyệt
NOISE_PATTERNS = [
    re.compile(r"^\d{1,2}:\d{2}\s+\d{1,2}/\d{1,2}/\d{2,4}\s+about:blank$"),
    re.compile(r"^about:blank(\s+\d+/\d+)?$"),
    # PyMuPDF (khác pdfplumber) tách timestamp/về:blank/số trang thành CÁC
    # DÒNG RIÊNG thay vì gộp chung 1 dòng -- cần thêm pattern cho từng phần:
    re.compile(r"^\d{1,2}:\d{2}\s+\d{1,2}/\d{1,2}/\d{2,4}$"),  # "21:28 8/8/26"
    re.compile(r"^\d{1,3}/\d{1,3}$"),  # "41/73" (số trang / tổng số trang)
]


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in NOISE_PATTERNS)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Trích xuất toàn bộ text từ PDF, đã lọc rác header/footer trình duyệt."""
    pdf_path = Path(pdf_path)
    lines_out = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            for line in page_text.split("\n"):
                if is_noise_line(line):
                    continue
                lines_out.append(line)

    return "\n".join(lines_out)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/87-2026TTBTC.pdf"
    text = extract_text_from_pdf(path)
    print(text[:1500])
    print(f"\n--- Tổng số ký tự: {len(text)} ---")