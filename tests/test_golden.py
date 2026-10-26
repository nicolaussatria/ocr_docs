"""End-to-end accuracy on the two sample pages (runs the real OCR models, ~20 s)."""
import json
from datetime import date
from pathlib import Path

import pytest

from docscan.pipeline import DocScanner
from eval_golden import evaluate

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def result():
    expected = json.loads((GOLDEN / "expected.json").read_text())
    files = sorted({e["file"] for e in expected})
    res = DocScanner().scan([(f, (GOLDEN / f).read_bytes()) for f in files], today=date(2026, 10, 12))
    return json.loads(res.model_dump_json()), expected


@pytest.mark.slow
def test_no_silent_errors_and_high_field_accuracy(result):
    out, expected = result
    s = evaluate(out, expected)
    assert s["matched_receipts"] == s["expected_receipts"] == 17
    assert s["extra_receipts"] == 0
    assert s["silent_errors"] == 0, s["details"]
    assert s["field_accuracy"] >= 0.95


@pytest.mark.slow
def test_handwritten_overrides_are_flagged(result):
    out, _ = result
    retail = [r for r in out["receipts"] if r["docType"] == "fuel_retail"]
    assert {r["amount"] for r in retail} == {300000, 319000}
    assert all(r["status"] == "needs_review" and r["claimedAmount"] for r in retail)


@pytest.mark.slow
def test_amounts_under_marker_and_handwriting_are_reported(result):
    out, _ = result
    by_type = lambda t: [r for r in out["receipts"] if r["docType"] == t]
    tolls = by_type("toll")
    assert sorted(r["amount"] for r in tolls) == [11000, 11000, 75500]
    halim = next(r for r in tolls if r["fields"].get("balanceAfter", {}).get("value") == 48501)
    assert halim["status"] == "verified" and halim["fields"]["fare"]["source"] == "ocr_refined"   # chain-confirmed
    assert all(r["status"] == "needs_review" for r in tolls if r is not halim)
    for doc_type, amount in (("kwitansi", 600000), ("fuel_cash_voucher", 35000)):
        (r,) = by_type(doc_type)
        assert r["amount"] == amount and r["status"] == "needs_review"


@pytest.mark.slow
def test_handwritten_date_and_volume_have_image_evidence(result):
    out, _ = result
    kwitansi = next(r for r in out["receipts"] if r["docType"] == "kwitansi")
    voucher = next(r for r in out["receipts"] if r["docType"] == "fuel_cash_voucher")
    signing_date = kwitansi["fields"]["transactionDate"]
    assert signing_date["value"] == "2026-10-02"
    assert signing_date["source"] == "ocr_refined" and signing_date["confidence"] <= .6
    volume = voucher["fields"]["volumeLiters"]
    assert volume["value"] == 3.5 and volume["source"] == "ocr_refined"
    assert volume["raw"] == "3.5" and volume["bbox"]
    price = voucher["fields"]["pricePerLiter"]
    assert price["value"] == 10000 and price["source"] == "derived"
    # The day/month still has competing readings. Preserve alternatives for the
    # reviewer rather than silently selecting the most frequent OCR variant.
    assert "transactionDate" not in voucher["fields"]
    assert "2026-07-04" in voucher["fields"]["transactionDateCandidates"]["value"]
