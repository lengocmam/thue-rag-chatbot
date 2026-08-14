"""
generate_ground_truth.py
Sinh bộ câu hỏi ground-truth (câu thật + câu bẫy) cho đánh giá retrieval.

NGUYÊN TẮC: mọi chunk_id tham chiếu trong file này được VALIDATE tự động
khi chạy script (assert chunk_id có thật trong chunks_strategy_B_khoan_context.jsonl)
-- nếu gõ sai/đánh máy nhầm chunk_id, script sẽ báo lỗi ngay thay vì âm
thầm tạo ra ground-truth SAI (rất nguy hiểm vì eval sẽ đo sai mà không biết).

Cách chạy:
    python scripts/generate_ground_truth.py

Output:
    data/eval/ground_truth.jsonl   -- câu hỏi thật, có đáp án + chunk đúng
    data/eval/trap_questions.jsonl -- câu hỏi bẫy (lỗi thời/không trả lời được/mâu thuẫn)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_PATH_B = PROJECT_ROOT / "data" / "processed" / "chunks_strategy_B_khoan_context.jsonl"
CHUNKS_PATH_A = PROJECT_ROOT / "data" / "processed" / "chunks_strategy_A_dieu.jsonl"
OUT_GT = PROJECT_ROOT / "data" / "eval" / "ground_truth.jsonl"
OUT_TRAP = PROJECT_ROOT / "data" / "eval" / "trap_questions.jsonl"

VALID_CHUNK_IDS = set()
CHUNK_LOOKUP = {}  # chunk_id -> {"dieu": ..., "khoan": ..., "doc_id": ...}


def load_valid_chunk_ids():
    """Nạp chunk_id hợp lệ từ CẢ 2 chiến lược A và B -- một số nội dung
    (VD bảng số liệu) chỉ tồn tại dưới dạng chunk riêng trong chiến lược A
    (xem chunk_bang_rieng() trong chunk_remaining_docs.py), không có trong
    B. Ground-truth cần tham chiếu được cả 2 nguồn.

    Đồng thời dựng CHUNK_LOOKUP để tự động suy ra (dieu, khoan, doc_id) từ
    chunk_id -- dùng cho bước eval sau này khi cần đối chiếu 1 câu hỏi
    XUYÊN SUỐT cả 3 chiến lược chunking (A/B/C có chunk_id khác nhau hoàn
    toàn cho cùng 1 nội dung), tránh phải gõ tay dễ sai."""
    for path in (CHUNKS_PATH_A, CHUNKS_PATH_B):
        with open(path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                VALID_CHUNK_IDS.add(c["chunk_id"])
                CHUNK_LOOKUP[c["chunk_id"]] = {
                    "dieu": c.get("dieu"), "khoan": c.get("khoan"), "doc_id": c.get("doc_id"),
                }


def gt(id_, question, category, doc_so_hieu, chunk_ids, expected_answer, difficulty="trung_binh"):
    for cid in chunk_ids:
        assert cid in VALID_CHUNK_IDS, f"[{id_}] chunk_id KHÔNG TỒN TẠI: {cid}"
    # Suy ra (doc_id, dieu, khoan) từ chunk_id thật -- dùng để đối chiếu
    # xuyên chiến lược lúc eval (xem eval/resolve_relevant.py)
    relevant_dieu_khoan = [
        {"doc_id": CHUNK_LOOKUP[cid]["doc_id"], "dieu": CHUNK_LOOKUP[cid]["dieu"], "khoan": CHUNK_LOOKUP[cid]["khoan"]}
        for cid in chunk_ids
    ]
    return {
        "id": id_, "question": question, "category": category,
        "doc_so_hieu": doc_so_hieu, "relevant_chunk_ids": chunk_ids,
        "relevant_dieu_khoan": relevant_dieu_khoan,
        "expected_answer": expected_answer, "difficulty": difficulty, "is_trap": False,
    }


def trap(id_, question, trap_type, why_trap, correct_behavior,
         chunk_ids_lien_quan=None, cau_tra_loi_sai_thuong_gap=None):
    chunk_ids_lien_quan = chunk_ids_lien_quan or []
    for cid in chunk_ids_lien_quan:
        assert cid in VALID_CHUNK_IDS, f"[{id_}] chunk_id KHÔNG TỒN TẠI: {cid}"
    return {
        "id": id_, "question": question, "trap_type": trap_type,
        "why_trap": why_trap, "correct_behavior": correct_behavior,
        "relevant_chunk_ids": chunk_ids_lien_quan,
        "cau_tra_loi_sai_thuong_gap": cau_tra_loi_sai_thuong_gap,
        "is_trap": True,
    }


def build_ground_truth() -> list:
    Q = []

    # ===== NHÓM 1: Ngưỡng doanh thu / miễn thuế hộ kinh doanh =====
    Q.append(gt("GT001",
        "Ngưỡng doanh thu bao nhiêu thì hộ kinh doanh, cá nhân kinh doanh không phải nộp thuế thu nhập cá nhân?",
        "nguong_doanh_thu", "141/2026/NĐ-CP",
        ["nd_141_2026_ndcp_dieu1_khoan1"],
        "01 tỷ đồng/năm trở xuống (theo Nghị định 141/2026/NĐ-CP, thay thế mức 500 triệu đồng cũ trong Luật 109/2025/QH15).",
        "kho"))
    Q.append(gt("GT002",
        "Hộ kinh doanh có doanh thu năm bao nhiêu thì phải áp dụng hóa đơn điện tử có mã của cơ quan thuế?",
        "nguong_doanh_thu", "141/2026/NĐ-CP",
        ["nd_141_2026_ndcp_dieu1_khoan2"],
        "Doanh thu năm trên 01 tỷ đồng thì phải áp dụng hóa đơn điện tử có mã của cơ quan thuế, hóa đơn điện tử khởi tạo từ máy tính tiền có kết nối dữ liệu với cơ quan thuế.",
        "trung_binh"))
    Q.append(gt("GT003",
        "Doanh nghiệp có tổng doanh thu năm bao nhiêu thì được miễn thuế thu nhập doanh nghiệp theo Nghị định 141/2026/NĐ-CP?",
        "nguong_doanh_thu", "141/2026/NĐ-CP",
        ["nd_141_2026_ndcp_dieu2_full"] if "nd_141_2026_ndcp_dieu2_full" in VALID_CHUNK_IDS else ["nd_141_2026_ndcp_dieu2_khoan1"],
        "Doanh nghiệp có tổng doanh thu năm từ 01 tỷ đồng trở xuống được miễn thuế thu nhập doanh nghiệp (khoản 15 Điều 4 NĐ 320/2025/NĐ-CP, bổ sung bởi NĐ 141/2026/NĐ-CP).",
        "kho"))

    # ===== NHÓM 2: Giảm trừ gia cảnh =====
    Q.append(gt("GT004",
        "Mức giảm trừ gia cảnh cho bản thân người nộp thuế là bao nhiêu mỗi tháng?",
        "giam_tru_gia_canh", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu10_khoan1"],
        "15,5 triệu đồng/tháng (186 triệu đồng/năm).", "de"))
    Q.append(gt("GT005",
        "Mức giảm trừ gia cảnh cho mỗi người phụ thuộc là bao nhiêu?",
        "giam_tru_gia_canh", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu10_khoan1"],
        "6,2 triệu đồng/tháng cho mỗi người phụ thuộc.", "de"))
    Q.append(gt("GT006",
        "Một người phụ thuộc có được tính giảm trừ gia cảnh cho 2 người nộp thuế cùng lúc không?",
        "giam_tru_gia_canh", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu10_khoan3"],
        "Không. Mỗi người phụ thuộc chỉ được tính giảm trừ một lần vào một người nộp thuế.", "trung_binh"))
    Q.append(gt("GT007",
        "Con đang học đại học có được tính là người phụ thuộc để giảm trừ gia cảnh không?",
        "giam_tru_gia_canh", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu10_khoan4"],
        "Có, nếu là con thành niên đang học đại học, cao đẳng, trung học chuyên nghiệp hoặc học nghề và không có thu nhập hoặc thu nhập không vượt mức do Bộ trưởng Bộ Tài chính quy định.", "trung_binh"))
    Q.append(gt("GT008",
        "Mức thu nhập tối đa để người phụ thuộc vẫn được tính giảm trừ gia cảnh là bao nhiêu?",
        "giam_tru_gia_canh", "87/2026/TT-BTC",
        ["tt_87_2026_ttbtc_dieu3_khoan1"],
        "Thu nhập bình quân tháng trong năm từ tất cả các nguồn không vượt quá 03 triệu đồng.", "trung_binh"))

    # ===== NHÓM 3: Hồ sơ xác định người phụ thuộc (TT 87/2026) =====
    Q.append(gt("GT009",
        "Hồ sơ xác định con đẻ là người phụ thuộc cần những giấy tờ gì?",
        "ho_so_thu_tuc", "87/2026/TT-BTC",
        ["tt_87_2026_ttbtc_dieu4_khoan1"],
        "Bản chụp Giấy khai sinh của con hoặc Quyết định nhận cha/mẹ/con, và bản chụp thẻ Căn cước của con nếu đã được cấp.", "trung_binh"))
    Q.append(gt("GT010",
        "Hồ sơ xác định vợ hoặc chồng là người phụ thuộc cần giấy tờ gì?",
        "ho_so_thu_tuc", "87/2026/TT-BTC",
        ["tt_87_2026_ttbtc_dieu4_khoan2"],
        "Bản chụp thẻ Căn cước và bản chụp Giấy chứng nhận kết hôn hoặc giấy tờ khác chứng minh quan hệ vợ chồng do cơ quan có thẩm quyền cấp.", "de"))
    Q.append(gt("GT011",
        "Người phụ thuộc là cha dượng/mẹ kế cần hồ sơ gì để chứng minh?",
        "ho_so_thu_tuc", "87/2026/TT-BTC",
        ["tt_87_2026_ttbtc_dieu4_khoan3"],
        "Bản chụp thẻ Căn cước, Giấy khai sinh của người nộp thuế, và Giấy chứng nhận kết hôn hoặc giấy tờ chứng minh quan hệ vợ chồng giữa cha dượng/mẹ kế với mẹ đẻ/cha đẻ của người nộp thuế.", "kho"))
    Q.append(gt("GT012",
        "Người khuyết tật từ đủ 18 tuổi có được tính là người phụ thuộc không, cần thêm giấy tờ gì?",
        "ho_so_thu_tuc", "87/2026/TT-BTC",
        ["tt_87_2026_ttbtc_dieu4_khoan5"],
        "Có, nếu có tỷ lệ suy giảm khả năng lao động từ 81% trở lên, cần thêm giấy tờ chứng minh tỷ lệ suy giảm khả năng lao động theo quy định.", "trung_binh"))

    # ===== NHÓM 4: Biểu thuế lũy tiến (Điều 9 Luật 109/2025) =====
    Q.append(gt("GT013",
        "Thu nhập từ tiền lương, tiền công đến 10 triệu đồng/tháng chịu thuế suất bao nhiêu?",
        "bieu_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_dieu9_bangthue"],
        "5% (bậc 1 của biểu thuế lũy tiến từng phần).", "de"))
    Q.append(gt("GT014",
        "Thu nhập từ tiền lương, tiền công trên 100 triệu đồng/tháng chịu thuế suất cao nhất là bao nhiêu?",
        "bieu_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_dieu9_bangthue"],
        "35% (bậc 5, bậc cao nhất của biểu thuế lũy tiến từng phần).", "de"))
    Q.append(gt("GT015",
        "Biểu thuế lũy tiến từng phần có bao nhiêu bậc?",
        "bieu_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_dieu9_bangthue"],
        "5 bậc.", "de"))

    # ===== NHÓM 5: Thuế suất các loại thu nhập khác (cá nhân cư trú) =====
    Q.append(gt("GT016",
        "Thuế suất đối với thu nhập từ chuyển nhượng chứng khoán của cá nhân cư trú là bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu13_khoan2"],
        "0,1% trên giá chuyển nhượng theo từng lần chuyển nhượng.", "de"))
    Q.append(gt("GT017",
        "Thuế suất đối với thu nhập từ chuyển nhượng bất động sản của cá nhân cư trú là bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu14_khoan1"],
        "2% trên giá chuyển nhượng.", "de"))
    Q.append(gt("GT018",
        "Thuế suất đối với thu nhập từ trúng thưởng của cá nhân cư trú là bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu15_khoan1"],
        "10% trên thu nhập tính thuế (phần giá trị giải thưởng vượt trên 20 triệu đồng).", "trung_binh"))
    Q.append(gt("GT019",
        "Thuế suất đối với thu nhập từ đầu tư vốn (tiền lãi cho vay, cổ tức...) của cá nhân cư trú là bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu12_khoan1"],
        "5%.", "de"))
    Q.append(gt("GT020",
        "Thuế suất đối với thu nhập từ tiền bản quyền là bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu16_khoan1"],
        "5% (áp dụng cho phần thu nhập vượt trên 20 triệu đồng theo từng hợp đồng).", "trung_binh"))

    # ===== NHÓM 6: Thuế cá nhân kinh doanh (Điều 7) =====
    Q.append(gt("GT021",
        "Cá nhân kinh doanh có doanh thu năm trên mức miễn thuế đến 3 tỷ đồng chịu thuế suất bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu7_khoan2"],
        "15% (theo Điều 7 khoản 2 điểm b Luật 109/2025/QH15, tính trên thu nhập tính thuế = doanh thu trừ chi phí).", "kho"))
    Q.append(gt("GT022",
        "Cá nhân kinh doanh có doanh thu năm trên 50 tỷ đồng chịu thuế suất bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu7_khoan2"],
        "20% (bậc cao nhất theo Điều 7 khoản 2 điểm d).", "trung_binh"))
    Q.append(gt("GT023",
        "Cá nhân cho thuê bất động sản (không phải lưu trú) nộp thuế thu nhập cá nhân theo thuế suất bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongII_dieu7_khoan4"],
        "5% trên phần doanh thu vượt mức không chịu thuế.", "trung_binh"))

    # ===== NHÓM 7: Thu nhập miễn thuế (Điều 4 -- 22 khoản) =====
    Q.append(gt("GT024",
        "Thu nhập từ chuyển nhượng bất động sản giữa vợ và chồng có bị đánh thuế thu nhập cá nhân không?",
        "mien_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu4_khoan1"],
        "Không, được miễn thuế (thuộc trường hợp chuyển nhượng bất động sản giữa các thành viên trong gia đình trực hệ theo khoản 1 Điều 4).", "de"))
    Q.append(gt("GT025",
        "Tiền lương hưu do Quỹ bảo hiểm xã hội chi trả có phải chịu thuế thu nhập cá nhân không?",
        "mien_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu4_khoan9"],
        "Không, được miễn thuế.", "de"))
    Q.append(gt("GT026",
        "Thu nhập từ kiều hối có bị đánh thuế thu nhập cá nhân không?",
        "mien_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu4_khoan7"],
        "Không, thu nhập từ kiều hối được miễn thuế.", "de"))
    Q.append(gt("GT027",
        "Học bổng từ ngân sách nhà nước có phải chịu thuế thu nhập cá nhân không?",
        "mien_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu4_khoan10"],
        "Không, học bổng nhận từ ngân sách nhà nước được miễn thuế.", "de"))
    Q.append(gt("GT028",
        "Tiền lương làm thêm giờ, làm việc ban đêm có được miễn thuế thu nhập cá nhân không?",
        "mien_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu4_khoan8"],
        "Có, phần tiền lương trả thêm cho làm đêm/làm thêm giờ (so với lương làm việc bình thường) được miễn thuế.", "trung_binh"))
    Q.append(gt("GT029",
        "Thu nhập từ nhận thừa kế, quà tặng là bất động sản duy nhất tại Việt Nam có bị đánh thuế không?",
        "mien_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu4_khoan2"],
        "Không bị đánh thuế nếu là nhà ở/đất ở duy nhất của cá nhân tại Việt Nam.", "trung_binh"))
    Q.append(gt("GT030",
        "Cá nhân là chủ doanh nghiệp tư nhân đã nộp thuế thu nhập doanh nghiệp thì có phải nộp thêm thuế thu nhập cá nhân trên phần thu nhập đó không?",
        "mien_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu4_khoan21"],
        "Không, thu nhập sau khi đã nộp thuế thu nhập doanh nghiệp của chủ doanh nghiệp tư nhân/chủ công ty TNHH một thành viên được miễn thuế TNCN.", "kho"))

    # ===== NHÓM 8: Đối tượng cư trú/không cư trú =====
    Q.append(gt("GT031",
        "Cá nhân có mặt tại Việt Nam bao nhiêu ngày thì được coi là cá nhân cư trú?",
        "doi_tuong_ap_dung", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu2_khoan2"],
        "Từ 183 ngày trở lên tính trong 1 năm dương lịch hoặc 12 tháng liên tục kể từ ngày đầu tiên có mặt tại Việt Nam.", "de"))
    Q.append(gt("GT032",
        "Cá nhân không cư trú chịu thuế trên thu nhập từ tiền lương, tiền công với thuế suất bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongIII_dieu21_full"] if "luat_109_2025_qh15_chuongIII_dieu21_full" in VALID_CHUNK_IDS else ["luat_109_2025_qh15_chuongIII_dieu21_khoan1"],
        "20% trên tổng thu nhập, không phân biệt nơi trả thu nhập.", "trung_binh"))
    Q.append(gt("GT033",
        "Cá nhân không cư trú chuyển nhượng bất động sản tại Việt Nam chịu thuế suất bao nhiêu?",
        "thue_suat", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongIII_dieu24_khoan1"],
        "2% trên giá chuyển nhượng (không được trừ chi phí, khác với cá nhân cư trú vẫn là 2% nhưng cách tính đơn giản hơn).", "trung_binh"))

    # ===== NHÓM 9: Hóa đơn điện tử (TT 91/2026) =====
    Q.append(gt("GT034",
        "Ký hiệu 'C' trong ký hiệu hóa đơn điện tử có nghĩa là gì?",
        "hoa_don_dien_tu", "91/2026/TT-BTC",
        ["tt_91_2026_ttbtc_chuongII_dieu4_khoan1"] if "tt_91_2026_ttbtc_chuongII_dieu4_khoan1" in VALID_CHUNK_IDS else [],
        "Thể hiện hóa đơn điện tử có mã của cơ quan thuế (ký tự 'K' thể hiện không có mã).", "trung_binh"))
    Q.append(gt("GT035",
        "Các trường hợp nào bị ngừng sử dụng hóa đơn điện tử?",
        "hoa_don_dien_tu", "91/2026/TT-BTC",
        ["tt_91_2026_ttbtc_chuongII_dieu8_khoan1"],
        "Nhiều trường hợp gồm: chấm dứt hiệu lực mã số thuế, không hoạt động tại địa chỉ đăng ký, tạm ngừng kinh doanh, có quyết định cưỡng chế nợ thuế, sử dụng hóa đơn bán hàng cấm/hàng giả, lập hóa đơn khống để chiếm đoạt tiền, v.v.", "trung_binh"))
    Q.append(gt("GT036",
        "Tổ chức cung cấp dịch vụ nhận, truyền, lưu trữ dữ liệu hóa đơn điện tử cần ký quỹ tối thiểu bao nhiêu?",
        "hoa_don_dien_tu", "91/2026/TT-BTC",
        ["tt_91_2026_ttbtc_chuongII_dieu12_khoan2"],
        "Không dưới 05 tỷ đồng (ký quỹ tại ngân hàng hoặc có giấy bảo lãnh ngân hàng).", "kho"))
    Q.append(gt("GT037",
        "Mức kinh phí tối đa cho chương trình 'hóa đơn may mắn' là bao nhiêu mỗi năm?",
        "hoa_don_dien_tu", "91/2026/TT-BTC",
        ["tt_91_2026_ttbtc_chuongII_dieu13_khoan3"],
        "Không quá 150 tỷ đồng/năm.", "kho"))

    # ===== NHÓM 10: Hiệu lực thi hành =====
    Q.append(gt("GT038",
        "Luật Thuế thu nhập cá nhân 109/2025/QH15 có hiệu lực từ ngày nào?",
        "hieu_luc", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongIV_dieu29_khoan1"],
        "Từ ngày 01/7/2026, trừ các quy định về thu nhập từ kinh doanh/tiền lương/tiền công áp dụng từ kỳ tính thuế năm 2026.", "de"))
    Q.append(gt("GT039",
        "Thông tư 91/2026/TT-BTC có hiệu lực từ ngày nào và thay thế thông tư nào?",
        "hieu_luc", "91/2026/TT-BTC",
        ["tt_91_2026_ttbtc_chuongV_dieu25_khoan1", "tt_91_2026_ttbtc_chuongV_dieu25_khoan2"],
        "Có hiệu lực từ 01/7/2026, thay thế Thông tư 32/2025/TT-BTC.", "trung_binh"))
    Q.append(gt("GT040",
        "Nghị định 141/2026/NĐ-CP có hiệu lực thi hành từ ngày nào?",
        "hieu_luc", "141/2026/NĐ-CP",
        ["nd_141_2026_ndcp_dieu3_full"],
        "Từ ngày 01/01/2026 -- lưu ý đây là mốc HỒI TỐ, trước cả ngày Nghị định được ban hành (29/4/2026).", "trung_binh"))

    # ===== NHÓM 11: Thu nhập chịu thuế (Điều 3) =====
    Q.append(gt("GT041",
        "Thu nhập từ nhận thừa kế, quà tặng là chứng khoán có phải chịu thuế thu nhập cá nhân không?",
        "thu_nhap_chiu_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu3_khoan9"],
        "Có, đây thuộc nhóm thu nhập chịu thuế theo khoản 9 Điều 3 (thừa kế/quà tặng là chứng khoán, phần vốn, bất động sản, tài sản phải đăng ký sở hữu/sử dụng).", "de"))
    Q.append(gt("GT042",
        "Thu nhập từ chuyển nhượng tên miền quốc gia Việt Nam '.vn' có phải chịu thuế thu nhập cá nhân không?",
        "thu_nhap_chiu_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu3_khoan10"],
        "Có, thuộc nhóm 'thu nhập khác' chịu thuế theo điểm a khoản 10 Điều 3.", "kho"))
    Q.append(gt("GT043",
        "Thu nhập từ chuyển nhượng tín chỉ các-bon có phải chịu thuế thu nhập cá nhân không?",
        "thu_nhap_chiu_thue", "109/2025/QH15",
        ["luat_109_2025_qh15_chuongI_dieu3_khoan10"],
        "Có, thuộc nhóm 'thu nhập khác' chịu thuế theo điểm b khoản 10 Điều 3 (thu nhập từ chuyển nhượng kết quả giảm phát thải khí nhà kính, tín chỉ các-bon).", "kho"))

    # ===== NHÓM 12: Đối tượng áp dụng / thủ tục hóa đơn điện tử (TT 91) =====
    Q.append(gt("GT044",
        "Đối tượng áp dụng của Thông tư 91/2026/TT-BTC là ai?",
        "doi_tuong_ap_dung", "91/2026/TT-BTC",
        ["tt_91_2026_ttbtc_chuongI_dieu2_full"],
        "Tổ chức, cá nhân quy định tại Điều 2 Nghị định số 254/2026/NĐ-CP.", "de"))
    Q.append(gt("GT045",
        "Người nộp thuế đăng ký địa chỉ trụ sở tại căn hộ chung cư có bị coi là rủi ro cao khi đăng ký hóa đơn điện tử không?",
        "hoa_don_dien_tu", "91/2026/TT-BTC",
        ["tt_91_2026_ttbtc_chuongII_dieu7_khoan3"],
        "Có, trừ trường hợp là căn hộ/phần diện tích được phép sử dụng cho mục đích kinh doanh theo quy định, và trừ trường hợp là cá nhân kinh doanh.", "kho"))
    Q.append(gt("GT046",
        "Sau khi đăng ký sử dụng chứng từ điện tử, cơ quan thuế phải phản hồi trong bao lâu?",
        "hoa_don_dien_tu", "91/2026/TT-BTC",
        ["tt_91_2026_ttbtc_chuongIII_dieu17_khoan2"],
        "Trong thời gian 01 ngày làm việc kể từ ngày nhận được đăng ký.", "de"))

    return Q


def build_trap_questions() -> list:
    T = []

    T.append(trap("TRAP001",
        "Ngưỡng doanh thu để hộ kinh doanh không phải nộp thuế thu nhập cá nhân là bao nhiêu?",
        "outdated_source",
        "Luật 109/2025/QH15 gốc ghi '500 triệu đồng' -- đây là con số ĐÃ LỖI THỜI. Luật 09/2026/QH16 đổi thành 'mức quy định của Chính phủ' (bỏ số), rồi Nghị định 141/2026/NĐ-CP quy định cụ thể là '01 tỷ đồng'. Nếu chatbot trả lời '500 triệu đồng' dựa trên Luật gốc, đó là hallucination do dùng nguồn lỗi thời.",
        "Câu trả lời ĐÚNG là '01 tỷ đồng/năm' (theo NĐ 141/2026/NĐ-CP). Hệ thống RAG tốt phải retrieval được văn bản sửa đổi mới nhất, không chỉ văn bản Luật gốc có vẻ liên quan nhất về mặt từ khóa.",
        ["luat_109_2025_qh15_chuongII_dieu7_khoan1", "nd_141_2026_ndcp_dieu1_khoan1"],
        "500 triệu đồng"))

    T.append(trap("TRAP002",
        "Luật sửa đổi 4 luật thuế (09/2026/QH16) có hiệu lực từ ngày Quốc hội thông qua đúng không?",
        "cross_doc_conflict",
        "CHỈ ĐÚNG MỘT PHẦN. Luật có hiệu lực chung từ ngày thông qua (24/4/2026), NHƯNG riêng Điều 1, 2, 3 (nội dung sửa đổi thuế TNCN, GTGT, TNDN) lại có hiệu lực HỒI TỐ từ 01/01/2026 -- TRƯỚC cả ngày thông qua hơn 3 tháng. Câu hỏi khiến người trả lời dễ áp dụng quy tắc chung mà bỏ sót ngoại lệ.",
        "Cần trả lời rõ: Điều 4 (thuế TTĐB xe điện) hiệu lực từ 24/4/2026 theo quy tắc chung; nhưng Điều 1, 2, 3 hiệu lực từ 01/01/2026 theo khoản 2 Điều 5.",
        ["luat_09_2026_qh16_dieu5_full"] if "luat_09_2026_qh16_dieu5_full" in VALID_CHUNK_IDS else [],
        "Có, toàn bộ luật có hiệu lực từ ngày thông qua"))

    T.append(trap("TRAP003",
        "Mức doanh thu cụ thể để hộ kinh doanh được miễn thuế giá trị gia tăng theo Luật 09/2026/QH16 là bao nhiêu?",
        "unanswerable_no_number",
        "Luật 09/2026/QH16 Điều 2 (sửa khoản 25 Điều 5 Luật Thuế GTGT) KHÔNG tự nêu con số cụ thể -- chỉ ghi 'Chính phủ quy định mức doanh thu năm...'. Nếu chatbot bịa ra một con số cụ thể khi trích từ chunk này, đó là hallucination rõ ràng.",
        "Câu trả lời đúng: Luật không tự nêu con số, giao Chính phủ quy định. Cần retrieval thêm văn bản Nghị định hướng dẫn (hiện chưa có trong bộ dữ liệu) mới trả lời được con số cụ thể -- nếu không có, hệ thống nên thành thật nói 'không tìm thấy con số cụ thể trong dữ liệu hiện có', không bịa ra 500 triệu hay 1 tỷ.",
        ["luat_09_2026_qh16_dieu2_full"] if "luat_09_2026_qh16_dieu2_full" in VALID_CHUNK_IDS else [],
        None))

    T.append(trap("TRAP004",
        "Thuế suất chuyển nhượng bất động sản của cá nhân không cư trú có khác cá nhân cư trú không?",
        "entity_confusion",
        "Cả 2 đối tượng đều chịu thuế suất 2%, NHƯNG cách tính khác nhau: cá nhân cư trú tính trên giá chuyển nhượng (Điều 14), cá nhân không cư trú cũng tính trên giá chuyển nhượng (Điều 24) -- tưởng như giống hệt nhau, dễ khiến người hỏi/trả lời nhầm là không có khác biệt gì, trong khi thực ra nhiều sắc thuế KHÁC (thu nhập kinh doanh, tiền lương...) lại có công thức tính rất khác nhau giữa 2 nhóm đối tượng này.",
        "Cần nêu rõ: với RIÊNG chuyển nhượng bất động sản, thuế suất giống nhau (2%) cho cả 2 nhóm, nhưng đây là ngoại lệ so với các loại thu nhập khác (như thu nhập kinh doanh, tiền lương) nơi 2 nhóm có công thức tính khác biệt rõ rệt.",
        ["luat_109_2025_qh15_chuongII_dieu14_khoan1", "luat_109_2025_qh15_chuongIII_dieu24_khoan1"],
        None))

    T.append(trap("TRAP005",
        "Nếu có quy định khác nhau về ưu đãi thuế TNCN giữa Luật 109/2025/QH15 và Luật Thủ đô thì áp dụng theo luật nào?",
        "out_of_corpus",
        "Câu hỏi này đề cập đến Luật Thủ đô -- văn bản KHÔNG có trong bộ dữ liệu hiện tại của dự án. Đây là câu hỏi 'ngoài phạm vi' để kiểm tra hệ thống có thành thật nhận là không đủ dữ liệu hay không, thay vì bịa ra nội dung của Luật Thủ đô.",
        "Luật 109/2025/QH15 Điều 29 khoản 4 CÓ đề cập nguyên tắc (ưu tiên Luật Thủ đô/nghị quyết Quốc hội, trừ khi Luật TNCN ưu đãi hơn thì người nộp thuế được chọn), nhưng KHÔNG có nội dung chi tiết của Luật Thủ đô để so sánh cụ thể. Hệ thống nên trả lời được NGUYÊN TẮC ưu tiên (có trong dữ liệu) nhưng thành thật nói không có đủ thông tin để so sánh chi tiết từng trường hợp cụ thể.",
        ["luat_109_2025_qh15_chuongIV_dieu29_khoan4"] if "luat_109_2025_qh15_chuongIV_dieu29_khoan4" in VALID_CHUNK_IDS else [],
        None))

    T.append(trap("TRAP006",
        "Thông tư 91/2026/TT-BTC có áp dụng cho việc kê khai thuế thu nhập cá nhân từ tiền lương không?",
        "scope_confusion",
        "Thông tư 91/2026/TT-BTC quy định về HÓA ĐƠN ĐIỆN TỬ, CHỨNG TỪ ĐIỆN TỬ (hướng dẫn Luật Quản lý thuế), KHÔNG phải văn bản quy định cách tính/kê khai thuế TNCN từ tiền lương (thuộc phạm vi Luật 109/2025/QH15 và TT 87/2026/TT-BTC). Câu hỏi dễ khiến hệ thống nhầm lẫn phạm vi điều chỉnh giữa các văn bản có vẻ liên quan.",
        "Cần trả lời rõ: TT 91/2026 không điều chỉnh nội dung này; hướng người hỏi sang Luật 109/2025/QH15 (Điều 8, 9) và TT 87/2026/TT-BTC.",
        ["tt_91_2026_ttbtc_chuongI_dieu1_khoan1"] if "tt_91_2026_ttbtc_chuongI_dieu1_khoan1" in VALID_CHUNK_IDS else [],
        None))

    return T


def append_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    load_valid_chunk_ids()
    print(f"Đã nạp {len(VALID_CHUNK_IDS)} chunk_id hợp lệ để đối chiếu (gộp cả A + B).")
    gt_list = build_ground_truth()
    trap_list = build_trap_questions()

    append_jsonl(OUT_GT, gt_list)
    append_jsonl(OUT_TRAP, trap_list)

    print(f"\nĐã sinh {len(gt_list)} câu hỏi thật -> {OUT_GT}")
    print(f"Đã sinh {len(trap_list)} câu hỏi bẫy -> {OUT_TRAP}")
    print("\nPhân bố theo category (câu thật):")
    from collections import Counter
    cat_count = Counter(q["category"] for q in gt_list)
    for cat, n in cat_count.most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()