"""Receipts on reused paper: boundaries, background totals, and genuine documents."""
import cv2
import numpy as np
import pytest
import json
from pathlib import Path

from docscan.foreground import receipt_foreground
from docscan.ocr import Line


GOLDEN = Path(__file__).parent / "golden"


def uploaded_page():
    from docscan.imaging import auto_levels
    img = auto_levels(cv2.imread(str(GOLDEN / "receipts-on-used-paper.jpg")))
    rows = json.loads((GOLDEN / "receipts-on-used-paper.ocr.json").read_text())
    return img, [Line(i, r["text"], r["score"], np.array(r["quad"], np.float32)) for i, r in enumerate(rows)]


def test_actual_scan_with_invisible_edges_excludes_backing_fragments():
    from docscan.foreground import _inside
    img, lines = uploaded_page()
    result = receipt_foreground(img, lines)
    assert result is not None and len(result.regions) == 3
    for line in lines:
        if "1,432,432" in line.text or "0012506" in line.text:
            assert not any(_inside(q, line) for q in result.regions)
            assert np.all(result.image[int(line.y0):int(line.y1), int(line.x0):int(line.x1)] == 255)


def test_column_fallback_requires_a_complete_fuel_body_in_every_column():
    from docscan.foreground import _fuel_columns
    _, lines = uploaded_page()
    for line in lines:
        if line.text.startswith("Waktu") and "28/09" in line.text:
            line.text = "Unreadable"
    assert _fuel_columns(lines) == []


def test_column_fallback_preserves_a_separate_readable_invoice():
    img, lines = uploaded_page()
    for i, text in enumerate(["INVOICE", "SHOP CONTOH", "02/10/2026", "Paper 10000", "Tax 1000", "TOTAL 11000"]):
        y = 80 + 30 * i
        lines.append(Line(1000+i, text, .98, np.array([[1350,y],[1490,y],[1490,y+20],[1350,y+20]], np.float32)))
    assert receipt_foreground(img, lines) is None


def test_subsidy_footer_is_attached_even_without_visible_paper_edges():
    from docscan.layout import segment_page
    _, lines = uploaded_page()
    segments = segment_page(lines)
    fuel = [s for s in segments if s.first("volume")]
    assert len(fuel) == 3
    assert all(any("CASH" in t for t in s.texts) for s in fuel)
    assert all("Subsidi" in s.full_text for s in fuel)
    assert not any("Unaudited" in s.full_text for s in segments if s not in fuel)


@pytest.mark.slow
def test_actual_uploaded_scan_has_three_complete_receipts_only():
    from docscan.pipeline import DocScanner
    p = GOLDEN / "receipts-on-used-paper.jpg"
    result = DocScanner().scan([(p.name, p.read_bytes())], include_raw_text=True)
    assert len(result.receipts) == 3
    assert sorted((r.transaction_date, r.amount) for r in result.receipts) == [
        ("2026-09-28", 42000), ("2026-09-29", 40000), ("2026-10-02", 40000)]
    assert sorted(r.fields["volumeLiters"].value for r in result.receipts) == [4, 4, 4.2]
    assert result.summary.total_amount == 122000
    assert result.pages[0].background_lines_ignored > 0
    for receipt in result.receipts:
        text = " ".join(receipt.raw_text)
        assert "CASH" in text and "Subsidi" in text
        assert "1,432,432" not in text and "0012506" not in text and "Serpong" not in text


def paper_page(amounts=(42000, 40000, 40000), background=185, invoice=False, border=180):
    """Controlled scan based on the supplied three-receipt/backing-invoice layout."""
    img = np.full((1100, 1500, 3), background, np.uint8)
    lines = []

    def write(text, x, y, size=.65, color=30):
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (color,) * 3, 1, cv2.LINE_AA)
        lines.append(Line(len(lines), text, .98, np.array([[x,y-h],[x+w,y-h],[x+w,y],[x,y]], np.float32)))

    # A backing invoice has a visible total, but its body is covered by receipts.
    cv2.rectangle(img, (15, 15), (1485, 1085), (100,) * 3, 1)
    write("OLD INVOICE 12/09/2026", 20, 45)
    write("Final TOTAL Rp 1,432,432.40", 980, 1060, .8)
    for y in (700, 800, 900, 1000):
        cv2.line(img, (20, y), (1480, y), (120,) * 3, 1)
        write("Qty", 460, y - 5)
        write("1432432.40", 910, y - 5, .4)
    for i, amount in enumerate(amounts):
        x = 50 + 480 * i
        cv2.rectangle(img, (x, 75), (x+400, 1010), (255,) * 3, -1)
        cv2.rectangle(img, (x, 75), (x+400, 1010), (border,) * 3, 1)
        text = ["PERTAMINA", f"SPBU 34.123.0{i+1}", "Jl. Bintaro Raya No. 45",
                f"No. Trans: 30988{i}", "Waktu: 02/10/2026 18:22:03", "Nama Produk: PERTALITE",
                "Harga/Liter: Rp 10.000", f"Volume: (L) {amount/10000:.3f}",
                f"Total Harga: Rp {amount:,}", f"CASH Rp {amount:,}", "Terima kasih"]
        if invoice and i == len(amounts) - 1:
            text = ["INVOICE", "PT INTERNET CONTOH", "Statement date 01/10/2026",
                    "Customer ID 10025007", "INTERNET 300.000", "VAT 33.000", "TOTAL Rp 333.000"]
        for j, t in enumerate(text):
            write(t, x+20, 145 + j*35)
    return img, lines


@pytest.mark.parametrize("background", [185, 244])
def test_only_receipt_sheets_survive_on_dark_or_white_backing(background):
    img, lines = paper_page(background=background)
    result = receipt_foreground(img, lines)
    assert result is not None and len(result.regions) == 3
    assert result.ignored_lines > 0
    assert np.all(result.image[1040:1070, 980:1450] == 255)  # backing total
    assert np.array_equal(result.image[100:980, 80:430], img[100:980, 80:430])
    assert np.any(img[1040:1070, 980:1450] < 255)  # caller's original was not changed


def test_two_receipts_matching_the_sideways_example():
    img, lines = paper_page((200000, 200000))
    result = receipt_foreground(img, lines)
    assert result is not None and len(result.regions) == 2


def test_faint_receipt_edges_on_nearly_white_paper():
    img, lines = paper_page(background=244, border=244)
    result = receipt_foreground(img, lines)
    assert result is not None and len(result.regions) == 3


def test_receipt_sheets_touching_scan_frame_keep_their_contents():
    img, lines = paper_page()
    cropped = img[75:]
    translated = [Line(l.idx, l.text, l.score, l.quad - np.array([0,75],np.float32))
                  for l in lines if l.y0 >= 75]
    result = receipt_foreground(cropped, translated)
    assert result is not None and len(result.regions) == 3
    assert np.array_equal(result.image[30:850, 80:430], cropped[30:850, 80:430])


def test_unbounded_third_receipt_prevents_partial_page_masking():
    img, lines = paper_page((200000, 200000))
    text = "SPBU third receipt with no visible border"
    lines.append(Line(999, text, .9, np.array([[1030,450],[1460,450],[1460,475],[1030,475]],np.float32)))
    assert receipt_foreground(img, lines) is None


def test_separate_complete_invoice_in_a_mixed_page_is_preserved():
    img, lines = paper_page(invoice=True)
    result = receipt_foreground(img, lines)
    assert result is not None and len(result.regions) == 3
    assert np.array_equal(result.image[100:980, 1040:1390], img[100:980, 1040:1390])


def test_single_document_never_activates_multi_receipt_filter():
    img, lines = paper_page((42000,))
    assert receipt_foreground(img, lines) is None


def test_rectangles_and_prices_without_transaction_evidence_are_not_receipts():
    img, lines = paper_page()
    for ln in lines:
        ln.text = "Item price 1432432.40"
    assert receipt_foreground(img, lines) is None


def test_multiple_bordered_invoice_sections_do_not_trigger_receipt_masking():
    img, lines = paper_page()
    for ln in lines:
        ln.text = ln.text.replace("PERTAMINA", "INVOICE").replace("SPBU", "Customer")
    assert receipt_foreground(img, lines) is None


@pytest.mark.slow
@pytest.mark.parametrize("rotation,amounts", [(0, (42000,40000,40000)), (90, (200000,200000))])
def test_real_ocr_excludes_backing_total_from_receipts_and_summary(rotation, amounts):
    from docscan.pipeline import DocScanner
    from docscan.imaging import rotate
    img, _ = paper_page(amounts)
    img = rotate(img, rotation)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    result = DocScanner().scan([("receipts-on-used-paper.png", encoded.tobytes())], include_raw_text=True)
    assert sorted(r.amount for r in result.receipts) == sorted(amounts)
    assert result.summary.total_amount == sum(amounts)
    assert result.pages[0].background_lines_ignored > 0
    assert result.warnings
    assert not any("1,432,432" in t or "OLD INVOICE" in t for r in result.receipts for t in (r.raw_text or []))
