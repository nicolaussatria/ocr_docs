"""Handwritten forms: calendar validation, ambiguity, field provenance, and decimal safety."""
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from docscan.imaging import handwritten_date_variants, zone_variants
from docscan.layout import Segment, Unit, match_label
from docscan.templates.base import Ctx, Draft
from docscan.templates.handwriting import choose_volume, handwritten_dates, handwritten_volume, set_handwritten_date
from docscan.templates.other import CashVoucher, Kwitansi
from test_bon_kontan import line


@pytest.mark.slow
def test_original_resolution_signing_date_is_second_not_thirteenth():
    """Actual date crops from the 4369 x 6074 upload, previously misread as 13/10.

    The full-page, smaller-copy regression lives in test_golden.py. These crops
    retain the original resolution and signature strokes without a 9 MB fixture.
    """
    from docscan.config import Settings
    from docscan.ocr import OcrEngine
    ocr = OcrEngine(Settings())
    readings = []
    for name in ("kwitansi-date-line.png", "kwitansi-date-digits.png"):
        crop = cv2.imread(str(Path(__file__).parent / "golden" / "handwriting" / name))
        readings += [(text, score, "ocr_refined", [])
                     for text, score in ocr.recognize(handwritten_date_variants(crop))]
    d = Draft("kwitansi", "other", Segment([]))
    set_handwritten_date(d, readings)
    assert d.val("transactionDate") == "2026-10-02"
    assert d.val("transactionAt") == "2026-10-02"
    assert "2/10/2026" in d.fields["transactionDate"].raw
    assert d.fields["transactionDate"].confidence <= .6


def test_thirteenth_is_not_replaced_with_second_by_a_date_rule():
    d = Draft("kwitansi", "other", Segment([]))
    set_handwritten_date(d, [("Jakarta, 13/10/2026", .9, "ocr_refined", [])] * 2)
    assert d.val("transactionDate") == "2026-10-13"


def test_date_preprocessing_caps_large_scan_dimensions():
    large = np.full((600, 3000, 3), 240, dtype=np.uint8)
    large[100:500, 400:500] = 60
    variants = handwritten_date_variants(large)
    assert len(variants) <= 12
    assert all(v.shape[0] == 96 and v.shape[1] <= 480 for v in variants)
    assert handwritten_date_variants(np.empty((0, 0, 3), np.uint8)) == []


@pytest.mark.parametrize("raw,expected", [
    ("Jakarta, 2/10/2026", date(2026, 10, 2)),
    ("Tgl 4.07 20..26", date(2026, 7, 4)),
    ("4.0720.26", date(2026, 7, 4)),
    ("4.07...20...2.6", date(2026, 7, 4)),
    ("29/02/2024", date(2024, 2, 29)),
    ("2 Oktober 2026", date(2026, 10, 2)),
    ("2026-10-02", date(2026, 10, 2)),
    ("31/02/2026", None), ("4/13/2026", None), ("4.07", None),
    ("4.07 20..", None), ("4.07.26", None), ("1310/2026", None),
    ("Rp 600.000", None), ("SPBU 34.13702", None),
])
def test_dates_require_complete_calendar_evidence(raw, expected):
    assert handwritten_dates(raw) == ({expected} if expected else set())


def test_date_disagreement_preserves_alternatives_instead_of_selecting_one():
    d = Draft("kwitansi", "other", Segment([]))
    set_handwritten_date(d, [(t, .9, "ocr_refined", []) for t in
                            ["2/10/2026", "2/10/2026", "13/10/2026"]])
    assert d.val("transactionDate") is None and d.val("transactionAt") is None
    assert d.val("transactionDateCandidates") == ["2026-10-02", "2026-10-13"]
    assert any("competing readings" in r for r in d.review)


def test_refined_date_needs_agreement_and_never_invents_time():
    d = Draft("fuel_cash_voucher", "fuel", Segment([]))
    readings = [("4.07 20..26", .9, "ocr_refined", [])]
    set_handwritten_date(d, readings)
    assert d.val("transactionDate") is None
    set_handwritten_date(d, readings * 2)
    assert d.val("transactionDate") == "2026-07-04"
    assert d.fields["transactionDate"].confidence <= .6
    assert d.val("transactionTime") is None
    assert d.val("transactionAt") == "2026-07-04"


def test_compact_date_only_supports_an_explicit_reading():
    d = Draft("kwitansi", "other", Segment([]))
    compact = [("210/2026", .95, "ocr_refined", [])] * 2
    set_handwritten_date(d, compact)
    assert d.val("transactionDate") is None
    set_handwritten_date(d, [("Jakarta, 2/10/2026", .8, "ocr_refined", [])] + compact)
    assert d.val("transactionDate") == "2026-10-02"
    assert d.fields["transactionDate"].raw == "Jakarta, 2/10/2026"


def test_decimal_lost_in_cleanup_can_support_two_explicit_readings():
    reads = [(t, s, "ocr_refined", []) for t, s in [("3.5", .82), ("3,5", .81), ("35", .95), ("35", .9)]]
    v = choose_volume(reads)
    assert v.value == 3.5 and v.raw == "3.5" and v.confidence <= .82
    assert choose_volume(reads[2:]).value == 35
    assert choose_volume(reads[1:]) is None


@pytest.mark.parametrize("text,expected", [
    ("3,5", 3.5), ("3.5 L", 3.5), ("3.500 Liter", 3.5), ("35", 35),
    (".3.5...", 3.5), ("0.25", .25), ("Rp 35.000", None), ("3.5.1", None),
    ("....", None), ("-3.5", None), ("0", None), ("35000", None),
])
def test_volume_preserves_decimal_semantics(text, expected):
    assert handwritten_volume(text) == expected


def unit(text, y):
    ln = line(text, 20, 500, y)
    return Unit([ln], label=match_label(text))


def test_kwitansi_uses_signing_date_not_billing_period():
    seg = Segment([unit("Sudah terima dari: PT Contoh", 30),
                   unit("Untuk pembayaran: tagihan 01/09/2026", 80),
                   unit("Jakarta, 2/10/2026", 180), unit("Jumlah: Rp 600.000", 230)])
    d = Kwitansi().parse(seg, Ctx(lambda lines: [], refine=False))
    assert d.val("transactionDate") == "2026-10-02"
    assert d.val("amount") == 600000
    assert d.val("volumeLiters") is None


def test_inline_volume_and_amount_parse_without_image_refinement():
    seg = Segment([unit("BON KONTAN", 30), unit("3,5 Liter Pertamax", 80),
                   Unit([line("Rp 45.500", 530, 650, 70)]), unit("Jumlah: Rp 45.500", 150)])
    d = CashVoucher().parse(seg, Ctx(lambda lines: [], refine=False))
    assert d.val("volumeLiters") == 3.5
    assert d.val("pricePerLiter") == 13000
    assert d.fields["volumeLiters"].source == "ocr"


def test_multiple_fuel_rows_do_not_assign_total_to_one_volume():
    seg = Segment([unit("BON KONTAN", 30), unit("3,5 Liter Pertamax", 80),
                   Unit([line("Rp 45.500", 530, 650, 70)]), unit("2 Liter Pertalite", 130),
                   Unit([line("Rp 20.000", 530, 650, 120)]), unit("Jumlah: Rp 65.500", 200)])
    d = CashVoucher().parse(seg, Ctx(lambda lines: [], refine=False))
    assert d.val("amount") == 65500 and len(d.val("items")) == 2
    assert d.val("volumeLiters") is None and d.val("pricePerLiter") is None


def test_handwriting_variants_are_bounded_and_keep_the_original():
    crop = np.full((40, 100, 3), 200, dtype=np.uint8)
    crop[10:30, 30:40] = 50
    original = crop.copy()
    base = zone_variants(crop, 30)
    variants = zone_variants(crop, 30, handwriting=True)
    assert 2 <= len(variants) <= 4
    assert np.array_equal(base[0], variants[0]) and np.array_equal(crop, original)
    assert all(v.shape == variants[0].shape for v in variants)
