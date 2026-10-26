"""What makes a receipt `verified`: a value only legible after image clean-up needs another figure to agree."""
import numpy as np

from docscan.config import Settings
from docscan.imaging import Page
from docscan.layout import Segment, Unit
from docscan.ocr import Line
from docscan.pipeline import DocScanner
from docscan.templates.base import FV, PASS, SKIP, Check, Draft


def draft(total_source: str, checks: list[Check]) -> Draft:
    q = np.array([[10, 10], [200, 10], [200, 40], [10, 40]], np.float32)
    seg = Segment([Unit([Line(0, "Total Harga: Rp. 319.000", 0.95, q)])])
    d = Draft("fuel_retail", "fuel", seg, amount_key="totalPrinted", checks=checks)
    d.set("totalPrinted", FV(319000, 0.9, total_source))
    d.set("transactionDate", FV("2026-10-02", 0.97))
    img = np.full((100, 300, 3), 255, np.uint8)
    d.page_meta = {"page": Page(img, "page.jpg", 1), "rotation": 0, "upright": img}
    return d


def finalize(d: Draft):
    return DocScanner(Settings(), engine=object())._finalize(d, include_images=False, include_raw=False)


def test_a_refined_total_nothing_confirms_goes_to_review():
    r = finalize(draft("ocr_refined", [Check("VOLUME_X_PRICE_EQ_TOTAL", SKIP, "missing volumeLiters", ["totalPrinted"])]))
    assert r.status == "needs_review"
    assert any("only legible after image clean-up" in x for x in r.review_reasons)


def test_a_refined_total_the_arithmetic_confirms_is_verified():
    r = finalize(draft("ocr_refined", [Check("VOLUME_X_PRICE_EQ_TOTAL", PASS, "ok", ["volumeLiters", "totalPrinted"])]))
    assert r.status == "verified" and r.amount == 319000


def test_a_plainly_read_total_is_unaffected():
    assert finalize(draft("ocr", [])).status == "verified"
