"""Written volume takes precedence over assumed prices; derived prices cannot validate their own inputs."""
import numpy as np

from docscan.layout import Segment, Unit
from docscan.ocr import Line
from docscan.templates.base import FV, Ctx, Draft
from docscan.templates.other import CashVoucher


def line(text, x0, x1, cy, h=30):
    q = np.array([[x0, cy - h / 2], [x1, cy - h / 2], [x1, cy + h / 2], [x0, cy + h / 2]], np.float32)
    return Line(0, text, 0.9, q)


def liters(readings, product="PERTALITE", amount=35000):
    row = line("..Liter Pertallte", 40, 390, 480)
    seg = Segment([Unit([row]), Unit([line("JUMLAH Rp.", 300, 560, 600)])])
    d = Draft("fuel_cash_voucher", "fuel", seg, amount_key="amount", handwritten=True)
    d.set("amount", FV(amount, 0.7, "ocr_refined"))
    answers = iter([list(readings)])                            # the first cut-out shows the writing, the wider one not
    ctx = Ctx(reread=lambda lines: [], refine=True, read_zone=lambda quad, **kw: next(answers, []))
    CashVoucher._liters(d, seg, [{"product": product, "amount": amount, "conf": 0.7, "row": row}], 0.0, ctx)
    return d


def test_written_liters_have_provenance_and_price_is_derived():
    d = liters([("3,5", 0.85)])
    assert d.val("volumeLiters") == 3.5 and d.val("pricePerLiter") == 10000
    assert d.fields["volumeLiters"].source == "ocr_refined"
    assert d.fields["volumeLiters"].raw == "3,5" and d.fields["volumeLiters"].lines
    assert d.fields["pricePerLiter"].source == "derived"
    assert d.fields["amount"].confidence == 0.7 and not d.checks


def test_one_shaky_reading_confirms_nothing():
    d = liters([("35", 0.49)])
    assert d.val("volumeLiters") == 3.5 and d.fields["amount"].confidence == 0.7 and not d.checks


def test_liters_partly_read_are_worked_out_and_flagged():
    d = liters([("3", 0.87), ("33", 0.51)])
    assert d.val("volumeLiters") == 3.5 and d.fields["volumeLiters"].confidence == 0.65
    assert any("assumed Rp 10,000" in r for r in d.review)
    assert d.fields["pricePerLiter"].source == "inferred"


def test_written_liters_that_disagree_are_preserved():
    d = liters([("4", 0.9), ("4", 0.8)])
    assert d.val("volumeLiters") == 4 and d.val("pricePerLiter") == 8750
    assert d.fields["amount"].confidence == 0.7
    assert [c.code for c in d.checks] == ["REFERENCE_PRICE_MISMATCH"]


def test_written_liters_do_not_require_a_reference_price():
    d = liters([("3,5", 0.9)], product="PERTAMAX", amount=45500)
    assert d.val("volumeLiters") == 3.5 and d.val("pricePerLiter") == 13000
    assert liters([], product="PERTAMAX").val("volumeLiters") is None


def test_decimal_point_is_not_invented_to_match_price():
    d = liters([("35", 0.9), ("35", 0.9)])
    assert d.val("volumeLiters") == 35
    assert d.val("pricePerLiter") == 1000
    assert d.checks[0].status == "warn"
