"""General receipt template and the layout rules behind it (no OCR models needed)."""
import numpy as np
import pytest

from docscan.layout import Segment, Unit, segment_page
from docscan.normalize import find_datetime, parse_amounts
from docscan.ocr import Line
from docscan.templates import classify
from docscan.templates.base import Ctx
from docscan.templates.receipt import Receipt, _label, build_rows, money_in, role_of

CTX = Ctx(reread=lambda lines: [], refine=False)


def line(text, x0, x1, y, h=30, score=0.99):
    return Line(0, text, score, np.array([[x0, y], [x1, y], [x1, y + h], [x0, y + h]], np.float32))


def receipt(*rows, x_amount=420, pitch=40):
    """One Line per cell: ('Total', '45,500') -> a name at the left, an amount flush right."""
    lines = []
    for i, cells in enumerate(rows):
        for j, text in enumerate(cells):
            x0 = 60 if j == 0 else x_amount
            lines.append(line(text, x0, x0 + 14 * len(text), 100 + i * pitch))
    return Segment([Unit([ln]) for ln in lines])


def parse(*rows, **kw):
    d = Receipt().parse(receipt(*rows, **kw), CTX)
    return d, {c.code: c.status for c in d.checks}


# --------------------------------------------------------------------------- rows
@pytest.mark.parametrize("text,role", [
    ("Total", "total"), ("GRAND TOTAL", "total"), ("Total Bayar", "total"), ("Dine In Total", "total"), ("T01AL", "total"),
    ("GR0 TTAL", "total"), ("BALANCE DUE:", "total"), ("Sub Total", "subtotal"), ("NETI AMT", "subtotal"),
    ("PB1 10%", "tax"), ("Pajak Resto (10%)", "tax"), ("SVC CHG 6%", "service"), ("Diskon", "discount"),
    ("Pembulatan", "rounding"), ("Tunai", "cash"), ("Pay Cash", "cash"), ("Total Kembajian", "change"),
    ("BCA Card", "noncash"), ("Total Qty", "count"), ("ITEMS:13", "count"),
    # things somebody ordered, not money rows
    ("Cashew Nut Chicken", None), ("Bottle", None), ("Papaya Juice", None), ("Ovomaltine", None), ("BALANCE PUYO CARD", None),
])
def test_row_names(text, role):
    assert role_of(_label(text)) == role


@pytest.mark.parametrize("text,small,values", [
    ("45,500", False, [45500]), ("Rp. 45.500,00", False, [45500]), ("70000.00", False, [70000]), ("154 000", False, [154000]),
    ("170 272.000", False, [272000]), ("1x12,500", False, [12500]), ("PB1 10% 27,666", True, [27666]), ("34-", True, [-34]),
    ("727", False, []), ("727", True, [727]),
    ("000.800", True, []),                      # a total read upside down
    ("12/10/2026 14:33", True, []), ("081234567890", True, []), ("1234 5678 9012 3456", True, []),
])
def test_money_tokens(text, small, values):
    assert [v for v, *_ in money_in(text, small=small)] == values


def test_printed_cents_are_not_missing_zeros():
    assert parse_amounts("70000.00")[0].value == 70000 and not parse_amounts("70000.00")[0].approx
    assert parse_amounts("35.00")[0].approx                      # handwriting: still '35.000' with a zero lost
    assert find_datetime("2026-10-12 14:33:05")[0].isoformat() == "2026-10-12"


def test_rows_pair_names_with_amounts_and_split_side_by_side_pairs():
    seg = receipt(("Nasi Goreng", "25,000"), ("Total", "25,000"))
    seg.units += [Unit([line("Pay Cash 50.000", 60, 260, 180)]), Unit([line("Change:25,000", 300, 480, 182)])]
    rows = {r.role: r.amount for r in build_rows(seg)}
    assert rows == {None: 25000, "total": 25000, "cash": 50000, "change": 25000}


def test_amount_printed_on_the_line_below_its_name():
    seg = Segment([Unit([ln]) for ln in (line("Service Charge 5%", 60, 300, 100), line("7700", 420, 480, 140),
                                         line("Total", 60, 130, 180), line("177.870", 400, 500, 220))])
    assert {r.role: r.amount for r in build_rows(seg)} == {"service": 7700, "total": 177870}


# --------------------------------------------------------------------------- the total
def test_total_confirmed_by_the_other_rows():
    d, checks = parse(("Choco Devil", "63,636"), ("SubTotal:", "63,636"), ("Discount:", "-9,545"), ("PB1:", "5,409"),
                      ("Total:", "59,500"), ("cash:", "100,000"), ("Change:", "40,500"))
    assert d.val("total") == 59500 and d.fields["total"].confidence >= 0.99
    assert checks["SUBTOTAL_TAX_EQ_TOTAL"] == checks["CASH_MINUS_CHANGE_EQ_TOTAL"] == "pass"
    assert (d.val("subtotal"), d.val("discount"), d.val("tax"), d.val("cashPaid"), d.val("change")) == (63636, 9545, 5409, 100000, 40500)
    assert d.val("paymentMethod") == "CASH"


def test_total_nobody_could_read_needs_two_independent_sums():
    rows = [("Choco Devil", "63,636"), ("SubTotal:", "63,636"), ("Discount:", "-9,545"), ("PB1:", "5,409"),
            ("Total:", "000.800"), ("cash:", "100,000"), ("Change:", "40,500")]
    d, _ = parse(*rows)                                           # subtotal sum and cash - change agree
    assert d.val("total") == 59500 and d.fields["total"].source == "derived" and d.fields["total"].confidence >= 0.95
    d, checks = parse(*rows[4:])                                  # only cash - change is left: a suggestion
    assert d.val("total") == 59500 and d.fields["total"].confidence < 0.5
    assert checks["CASH_MINUS_CHANGE_EQ_TOTAL"] == "skip" and "cashPaid" not in d.fields


def test_printed_total_is_not_replaced_by_a_sum_with_a_row_missing():
    # the tax row was lost: subtotal + service = 276,660, but 304,326 is what is printed
    d, checks = parse(("SUBTTL", "261,000"), ("SVC CHG 6%", "15,660"), ("TOTAL", "304,326"))
    assert d.val("total") == 304326 and d.fields["total"].source == "ocr"
    # ... and it is exactly (subtotal + service) + 10% tax, which confirms it without the tax row
    assert checks["SUBTOTAL_PLUS_TAX_RATE_EQ_TOTAL"] == "pass" and checks["SUBTOTAL_TAX_EQ_TOTAL"] == "skip"
    assert "serviceCharge" not in d.fields                         # a row no closed sum used stays out
    # print and arithmetic disagree and nothing takes a side: the printed figure is kept as the best
    # guess, but it is not confident enough to be passed on as the amount
    d, checks = parse(("SUBTTL", "261,000"), ("SVC CHG 6%", "15,660"), ("TOTAL", "300,000"))
    assert d.val("total") == 300000 and d.fields["total"].confidence < 0.5 and checks["SUBTOTAL_TAX_EQ_TOTAL"] == "warn"
    # with the tax row readable only as a number, the sum closes and everything is confirmed
    d, checks = parse(("SUBTTL", "261,000"), ("SVC CHG 6%", "15,660"), ("", "27,666"), ("TOTAL", "304,326"))
    assert d.val("total") == 304326 and checks["SUBTOTAL_TAX_EQ_TOTAL"] == "pass" and d.val("serviceCharge") == 15660


def test_implausible_readings_of_a_total_are_dropped():
    d, _ = parse(("Nutella Cheese", "27,272"), ("Total:", "29,9"), ("TOTAL QTY", "3"))
    assert "total" not in d.fields                                 # 299 is a total that lost its digits
    d, _ = parse(("Kopi", "23.000"), ("Total", "23.000"), ("Total Qty.", "171"))
    assert d.val("total") == 23000


def test_arithmetic_alone_names_a_total_only_as_a_suggestion():
    d, _ = parse(("", "75,000"), ("", "10,000"), ("", "85,000"))    # a bare column that happens to add up
    assert d.fields["total"].confidence < 0.5 and any("arithmetic only" in r for r in d.review)


def test_merchant_name_needs_a_visible_header():
    d, _ = parse(("LEMONADE 16OZ",), ("Thai Tea", "18,000"), ("Total", "18,000"))
    assert d.val("merchantName") is None                           # cropped to the items: that is a drink
    d, _ = parse(("KOPI KENANGAN",), ("Jl. Sudirman No. 5",), ("12/10/2026 14:33",), ("Thai Tea", "18,000"), ("Total", "18,000"))
    assert d.val("merchantName") == "KOPI KENANGAN" and d.val("transactionDate") == "2026-10-12"


def test_items_are_listed_only_when_they_add_up():
    d, checks = parse(("Egg Tart", "13,000"), ("Pizza Toast", "16,000"), ("TOTAL", "29,000"))
    assert [i["amount"] for i in d.val("items")] == [13000, 16000] and checks["ITEMS_SUM_EQ_TOTAL"] == "pass"
    d, _ = parse(("Egg Tart", "13,000"), ("Pizza Toast", "16,000"), ("TOTAL", "45,500"))
    assert d.val("items") is None and d.val("total") == 45500


def test_general_template_ranks_below_the_specific_ones():
    seg = receipt(("Egg Tart", "13,000"), ("TOTAL", "13,000"))
    assert classify(seg)[0].doc_type == "receipt"
    prose = Segment([Unit([line("Terima kasih atas kunjungan Anda", 60, 500, 100)])])
    assert classify(prose)[0].doc_type == "unknown"


# --------------------------------------------------------------------------- layout
def test_detached_amount_column_rejoins_its_names():
    names = ["Kopi Susu", "Total", "Tunai", "Kembali"]
    amounts = ["23.000", "23.000", "50.000", "27.000"]
    lines = [line(t, 60, 60 + 14 * len(t), 100 + 40 * i) for i, t in enumerate(names)]
    lines += [line(t, 700, 800, 102 + 40 * i) for i, t in enumerate(amounts)]      # 20 text heights to the right
    (seg,) = segment_page(lines)
    assert len(seg.lines) == 8


def test_amount_column_does_not_join_a_complete_receipt_next_door():
    left = [line(t, x, x + 14 * len(t), 100 + 40 * i) for i, row in enumerate([("Kopi Susu", "23.000"), ("Total", "23.000")])
            for t, x in zip(row, (60, 300))]
    right = [line(t, 520, 620, 101 + 40 * i) for i, t in enumerate(["18.000", "18.000"])]   # another receipt's amounts
    assert len(segment_page(left + right)) == 2


# --------------------------------------------------------------------------- how sure a total is
def test_total_nothing_confirms_is_reported_but_goes_to_review():
    d, _ = parse(("Kopi Susu", "23.000"), ("Total", "23.000"), ("Terima kasih",))
    assert d.val("total") == 23000 and d.fields["total"].confidence >= 0.97    # the item confirms it
    d, _ = parse(("Kopi Susu",), ("Total", "93.716"))
    assert d.val("total") == 93716 and 0.5 <= d.fields["total"].confidence < 0.85
    assert any("not confirmed" in r for r in d.review)


def test_rounding_tolerance_needs_a_round_figure():
    # 98,180 + 9,818 = 107,998; '107,999' is a misread digit, not a cashier rounding
    d, checks = parse(("Sub Total", "98.180"), ("Pajak Resto", "9,818"), ("Grand Total", "107,999"))
    assert checks["SUBTOTAL_TAX_EQ_TOTAL"] == "warn" and d.fields["total"].confidence < 0.5
    # 55,834 rounded down to 55,800 at the till is
    d, checks = parse(("TOTAL:", "55.834"), ("Tunai:", "60,000"), ("Uang Kembali:", "4,200"))
    assert d.val("total") == 55834 and checks["CASH_MINUS_CHANGE_EQ_TOTAL"] == "pass"


def test_items_and_subtotal_are_one_line_of_evidence():
    # the extra charge is named in a way no rule knows: items and subtotal both say 273,000, the print says 300,300
    d, _ = parse(("Kue Cubit", "153,000"), ("Es Buah", "120,000"), ("SUBTOTAL", "273,000"), ("XQZ WVK", "27,300"),
                 ("TOTAL", "300,300"))
    assert d.val("total") == 300300 and d.fields["total"].source == "ocr"
    assert role_of(_label("BIAYA TAMBAHAN")) == "fee"                 # ... and this one is known: not a misspelt 'bayar'


def test_card_payment_that_differs_makes_the_total_a_suggestion():
    d, _ = parse(("Avocado Coffee",), ("Total", "28,000"), ("OVO", "92,000"))
    assert d.fields["total"].confidence < 0.5


def test_totals_block_printed_after_a_gap_is_the_end_of_the_receipt_above():
    top = [("Pecel Pincuk", "127.270"), ("Empal", "82.729"), ("SubTotal", "209.999"), ("Total", "209.999"), ("PB1 10%", "21.000")]
    low = [("TOTAL", "230.999"), ("CASH", "250.000"), ("CHANGE", "19.001")]
    lines = [line(t, x, x + 13 * len(t), 100 + 40 * i) for i, row in enumerate(top) for t, x in zip(row, (60, 400))]
    lines += [line(t, x, x + 13 * len(t), 380 + 40 * i) for i, row in enumerate(low) for t, x in zip(row, (60, 400))]   # blank band
    (seg,) = segment_page(lines)
    d = Receipt().parse(seg, CTX)
    assert d.val("total") == 230999 and d.val("cashPaid") == 250000        # one receipt, one amount
