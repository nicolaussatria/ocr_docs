"""Mobile top-up regressions based on a manually transcribed m-Commerce layout.

The text/boxes below simulate OCR output, not recognition of the attached photo.
Phone/serial identifiers in the fixtures are synthetic.
"""
import io
from datetime import date

import numpy as np
import pytest
from PIL import Image

from docscan.batch import fingerprint, run_batch_checks
from docscan.config import Settings
from docscan.layout import Segment, build_units, header_kind, match_label, segment_page
from docscan.normalize import parse_printed_idr
from docscan.ocr import Line
from docscan.pipeline import DocScanner
from docscan.templates import Ctx, classify
from docscan.templates.mobile import MobileTopup


def receipt_text(*, second=False):
    return [
        "m-Commerce:", "BERHASIL",
        "30/09/2026 11:37:16" if second else "01/10/2026 06:23:35",
        "TELKOMSEL PULSA", "081234567802" if second else "081234567801",
        "NOMINAL: Rp. 75,000.00" if second else "NOMINAL: Rp. 200,000.00",
        "BIAYA ADMIN: Rp. 2,000.00",
        "JUMLAH BAYAR: Rp. 77,000.00" if second else "JUMLAH BAYAR: Rp. 202,000.00",
        "Keterangan: Pulsa 75,000.00" if second else "Keterangan: Pulsa 200,000.00",
        "SN=00000000000000000002" if second else "SN=00000000000000000001",
        "CEK PULSA www.telkomsel.com", "PT. TELEKOMUNIKASI SELULER",
        "GD TELKOM LANDMARK TOWER 1", "Lt1-20, JL.Jend.Gatot Subroto",
        "Kav 52-53, Jakarta 12710", "NPWP : 01-781.327.8-093.000",
        "Pulsa Anda Bertambah Otomatis", "Call Center TSEL di 188",
        "Biaya Termasuk PPN (Bila ada)", "PT. BANK CENTRAL ASIA TBK.",
        "MENARA BCA - JAKARTA PUSAT", "NPWP : 0013084496091000",
    ]


def lines_for(texts, *, x=100, y=50, slope=0, footer_gap=22):
    lines = []
    for i, text in enumerate(texts):
        if text.startswith("Biaya Termasuk"):
            y += footer_gap
        width = max(60, len(text) * 8)
        q = np.array([[x, y], [x + width, y + slope * width],
                      [x + width, y + 16 + slope * width], [x, y + 16]], np.float32)
        lines.append(Line(i, text, 0.98, q))
        y += 22
    return lines


def draft(texts=None, *, ctx=None):
    seg = Segment(sorted(build_units(lines_for(texts if texts is not None else receipt_text())), key=lambda u: u.y0))
    return MobileTopup().parse(seg, ctx or Ctx(reread=lambda _: [], refine=False))


class FixtureOcr:
    version = "fixture OCR (transcribed text)"

    def __init__(self, lines):
        self.lines = lines

    def detect_orientation(self, image):
        return 0

    def read(self, image):
        return self.lines


def scan(lines):
    image = io.BytesIO()
    Image.new("RGB", (1200, 1400), "white").save(image, format="PNG")
    return DocScanner(Settings(refine_enabled=False), FixtureOcr(lines)).scan(
        [("mobile-topups.png", image.getvalue())], include_raw_text=True, today=date(2026, 10, 12))


@pytest.mark.parametrize("text,value", [
    ("Rp. 200,000.00", 200000), ("Rp. 2.000,00", 2000), ("75,000.00", 75000),
    ("202000.00", 202000), ("77.000", 77000), ("0.00", 0), ("35.00", 35),
    ("Rp 1,234,567.00", 1234567), ("IDR 1.234.567,00", 1234567),
    ("20,00.00", None), ("Rp. 2,000.50", None), ("-2,000.00", None),
    ("Rp. 20?000.00", None), ("202000 1000", None), ("unreadable", None),
])
def test_printed_amounts_are_strict_and_do_not_guess_thousands(text, value):
    assert parse_printed_idr(text) == value


def test_mobile_labels_and_anchor():
    for text, key in [("NOMINAL: Rp. 200,000.00", "topupNominal"), ("BIAYA ADMIN: Rp. 2,000.00", "adminFee"),
                      ("JUMLAH BAYAR: Rp. 202,000.00", "paymentTotal"), ("SN=000012345678", "topupSerial")]:
        assert match_label(text).key == key
    assert header_kind("m-Commerce:") == "anchor"


@pytest.mark.parametrize("slope", [0, 0.025, -0.04])
def test_two_topups_on_one_page_are_separate_expenses(slope):
    result = scan(lines_for(receipt_text(), slope=slope) + lines_for(receipt_text(second=True), y=610, slope=slope))
    assert result.summary.receipt_count == result.summary.verified_count == 2
    assert result.summary.total_amount == result.summary.by_category["pulsa"].amount == 279000
    first, second = result.receipts
    assert (first.doc_type, first.category, first.amount, second.amount) == ("mobile_topup", "pulsa", 202000, 77000)
    assert (first.transaction_at, second.transaction_at) == ("2026-10-01T06:23:35", "2026-09-30T11:37:16")
    assert first.fields["phoneNumber"].value == "081234567801"
    assert second.fields["phoneNumber"].value == "081234567802"
    assert first.fields["transactionNo"].value == "00000000000000000001"
    assert first.merchant_name == second.merchant_name == "TELKOMSEL"
    assert first.fields["nominal"].value == 200000 and first.fields["adminFee"].value == 2000
    assert all(r.fields["amountPaid"].bbox and all(c.status == "pass" for c in r.checks) for r in result.receipts)
    assert all(sum("BANK CENTRAL" in t for t in r.raw_text) == 1 for r in result.receipts)


def test_repeated_header_splits_even_without_whitespace():
    lines = lines_for(receipt_text() + receipt_text(second=True), footer_gap=0)
    segments = segment_page(lines)
    assert len(segments) == 2
    assert all(classify(s)[0].doc_type == "mobile_topup" for s in segments)


def test_detached_bank_footers_are_not_extra_receipts():
    result = scan(lines_for(receipt_text(), footer_gap=70) +
                  lines_for(receipt_text(second=True), y=680, footer_gap=70))
    assert result.summary.receipt_count == result.summary.verified_count == 2
    assert result.summary.total_amount == 279000
    assert all(any("BANK CENTRAL" in t for t in r.raw_text) for r in result.receipts)


def test_distant_or_adjacent_bank_footer_is_not_attached():
    main = lines_for(receipt_text()[:18])
    footer = lines_for(receipt_text()[18:], x=650, y=460, footer_gap=0)
    segments = segment_page(main + footer)
    assert len(segments) == 2
    assert not any("BANK CENTRAL" in t for t in segments[0].texts)


def test_two_receipts_in_columns_do_not_share_amounts():
    result = scan(lines_for(receipt_text()) + lines_for(receipt_text(second=True), x=650))
    assert [r.amount for r in result.receipts] == [202000, 77000]
    assert result.summary.receipt_count == 2


def test_separate_label_and_value_boxes_are_paired():
    lines = lines_for(receipt_text())
    nominal = lines[5]
    y = nominal.y0
    lines[5:6] = [
        Line(100, "NOMINAL:", .98, np.array([[100, y], [180, y], [180, y+16], [100, y+16]], np.float32)),
        Line(101, "Rp. 200,000.00", .98, np.array([[200, y], [330, y], [330, y+16], [200, y+16]], np.float32)),
    ]
    (r,) = scan(lines).receipts
    assert r.status == "verified" and r.amount == 202000 and r.fields["nominal"].value == 200000


def test_mismatched_arithmetic_is_never_verified():
    texts = receipt_text()
    texts[7] = "JUMLAH BAYAR: Rp. 282,000.00"
    (r,) = scan(lines_for(texts)).receipts
    assert r.status == "needs_review" and r.amount == 282000
    assert any(c.code == "NOMINAL_PLUS_ADMIN_EQ_PAID" and c.status == "fail" for c in r.checks)


def test_missing_paid_total_is_derived_but_requires_review():
    texts = [t for t in receipt_text() if not t.startswith("JUMLAH BAYAR")]
    (r,) = scan(lines_for(texts)).receipts
    assert r.amount == 202000 and r.status == "needs_review"
    assert r.fields["amountPaid"].source == "derived"
    assert r.fields["amountPaid"].confidence < .85


@pytest.mark.parametrize("omit", ["BIAYA ADMIN", "NOMINAL", "081234", "BERHASIL"])
def test_missing_required_evidence_requires_review(omit):
    texts = [t for t in receipt_text() if not t.startswith(omit)]
    (r,) = scan(lines_for(texts)).receipts
    assert r.doc_type == "mobile_topup" and r.status == "needs_review"


@pytest.mark.parametrize("status", ["GAGAL", "TIDAK BERHASIL", "PENDING", "DALAM PROSES"])
def test_unsuccessful_topups_do_not_count_as_expenses(status):
    texts = receipt_text()
    texts[1] = status
    result = scan(lines_for(texts))
    (r,) = result.receipts
    assert r.status == "needs_review" and r.amount is None
    assert r.fields["amountPaid"].value == 202000
    assert result.summary.total_amount == 0


def test_zero_admin_fee_is_valid():
    texts = receipt_text()
    texts[6:8] = ["BIAYA ADMIN: Rp. 0.00", "JUMLAH BAYAR: Rp. 200,000.00"]
    (r,) = scan(lines_for(texts)).receipts
    assert r.amount == 200000 and r.fields["adminFee"].value == 0 and r.status == "verified"


def test_unreadable_amount_can_be_refined_with_multiple_readings():
    texts = receipt_text()
    texts[5] = "NOMINAL: Rp. ???"
    ctx = Ctx(reread=lambda _: [("NOMINAL: Rp. 200,000.00", .97), ("NOMINAL: Rp. 200,000.00", .96)])
    d = draft(texts, ctx=ctx)
    assert d.val("nominal") == 200000 and d.fields["nominal"].source == "ocr_refined"
    assert d.fields["nominal"].confidence < .85
    ctx = Ctx(reread=lambda _: [("NOMINAL: Rp. 200,000.00", .97)])
    assert draft(texts, ctx=ctx).val("nominal") is None


def test_phone_and_serial_are_not_confused_with_tax_ids():
    d = draft([t for t in receipt_text() if not t.startswith(("081234", "SN="))])
    assert d.val("phoneNumber") is None and d.val("transactionNo") is None


def test_duplicate_fingerprints_distinguish_phone_numbers():
    texts = [t for t in receipt_text() if not t.startswith("SN=")]
    a, b = draft(texts), draft([t.replace("081234567801", "081234567802") for t in texts])
    assert fingerprint(a) != fingerprint(b)
    same = draft(texts)
    a.rid, b.rid, same.rid = "r1", "r2", "r3"
    checks = run_batch_checks([a, b, same], date(2026, 10, 12))
    assert len(checks) == 1 and checks[0].code == "NOT_DUPLICATE"
    assert not any(c.code == "NOT_DUPLICATE" for c in b.checks)


def test_unrelated_mcommerce_payment_is_not_classified_as_pulsa():
    seg = Segment(build_units(lines_for([t.replace("TELKOMSEL PULSA", "PLN PRABAYAR") for t in receipt_text()])))
    assert MobileTopup().score(seg) == 0


def test_api_returns_typed_mobile_fields_and_pulsa_summary(monkeypatch):
    from fastapi.testclient import TestClient
    from docscan.api import _state, app

    monkeypatch.delenv("DOCSCAN_API_KEY", raising=False)
    lines = lines_for(receipt_text()) + lines_for(receipt_text(second=True), y=610)
    monkeypatch.setitem(_state, "scanner", DocScanner(Settings(refine_enabled=False), FixtureOcr(lines)))
    image = io.BytesIO()
    Image.new("RGB", (1200, 1400), "white").save(image, format="PNG")
    # Without a lifespan context, use the injected scanner rather than loading models.
    client = TestClient(app)
    try:
        response = client.post("/v1/scan?today=2026-10-12", files={"files": ("topups.png", image.getvalue(), "image/png")})
        assert response.status_code == 200
        body = response.json()
        assert body["schemaVersion"] == "1.0"
        assert body["summary"]["byCategory"]["pulsa"] == {"count": 2, "amount": 279000}
        first, second = body["receipts"]
        assert first["docType"] == second["docType"] == "mobile_topup"
        assert first["fields"]["phoneNumber"]["value"] == "081234567801"
        assert first["fields"]["amountPaid"]["value"] == 202000
        assert second["fields"]["amountPaid"]["value"] == 77000
        assert "rawText" not in first
    finally:
        client.close()


@pytest.mark.slow
def test_real_ocr_on_generated_topup_page():
    """Exercise the installed OCR models on a generated, mildly skewed page.

    This is a synthetic smoke test, not an accuracy claim for camera photos.
    """
    import cv2
    from PIL import ImageDraw, ImageFont

    image = Image.new("RGB", (960, 1280), (225, 225, 220))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    for second, start in ((False, 130), (True, 710)):
        y = start
        for text in receipt_text(second=second):
            if text.startswith("Biaya Termasuk"):
                y += 20
            draw.text((270, y), text, font=font, fill=(55, 55, 50))
            y += 22
    # Uneven illumination and a small perspective tilt exercise photo OCR.
    pixels = np.asarray(image).astype(np.float32)
    pixels *= np.linspace(.72, 1.0, 960)[None, :, None]
    source = np.float32([[0, 0], [959, 0], [959, 1279], [0, 1279]])
    target = np.float32([[20, 10], [939, 30], [949, 1269], [10, 1249]])
    pixels = cv2.warpPerspective(pixels.astype(np.uint8), cv2.getPerspectiveTransform(source, target), (960, 1280),
                                 borderValue=(220, 220, 220))
    data = io.BytesIO()
    Image.fromarray(pixels).save(data, format="JPEG", quality=85)
    result = DocScanner().scan([("synthetic-topups.jpg", data.getvalue())], today=date(2026, 10, 12))
    assert result.summary.receipt_count == 2
    assert [r.doc_type for r in result.receipts] == ["mobile_topup", "mobile_topup"]
    assert [r.amount for r in result.receipts] == [202000, 77000]
    assert [r.fields["phoneNumber"].value for r in result.receipts] == ["081234567801", "081234567802"]
    assert [r.transaction_date for r in result.receipts] == ["2026-10-01", "2026-09-30"]
    assert result.summary.total_amount == 279000
