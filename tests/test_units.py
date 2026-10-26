"""Fast unit tests (no OCR models needed)."""
from datetime import date, time

import httpx
import numpy as np
import pytest

from docscan.batch import run_batch_checks
from docscan.imaging import flatten_shading, ink_extent, remove_rules
from docscan.layout import Segment, Unit, build_units, header_kind, match_label
from docscan.normalize import (find_card_numbers, find_datetime, find_partial_date, glyph_amount, hand_amount,
                               parse_amounts, parse_duration_minutes, parse_volume, pattern_matches, station_code,
                               terbilang_to_int, terbilang_vote)
from docscan.ocr import Line
from docscan.templates.base import FV, Cand, Draft, Rel, apply_solution, settle, solve


# --------------------------------------------------------------------------- normalize
@pytest.mark.parametrize("text,expected", [
    ("Rp. 15,950", [15950]), ("54.032", [54032]), ("Rp 33,000,00", [33000]), (":20,032", [20032]),
    ("Rp. 300,000 108.000", [300000, 108000]), ("Rp.59501", [59501]), ("3O,OOO", [30000]),
])
def test_amounts(text, expected):
    assert [a.value for a in parse_amounts(text)] == expected


def test_amount_wildcards_and_approx():
    a = parse_amounts("CN Rp.485[1")[-1]
    assert a.value is None and a.pattern == "485?1" and pattern_matches(a.pattern, 48501)
    assert parse_amounts("CN:01452022 0164{0 Rp.485(1")[-1].pattern == "485?1"     # same digit, off a phone photo
    assert [x.value for x in parse_amounts("Telp.(021)22329586")] == [21, 22329586]  # a phone stays two numbers
    b = parse_amounts("35.00")[0]
    assert b.value == 35000 and b.approx


def test_dates():
    assert find_datetime("02/10/202622:49:04")[:2] == (date(2026, 10, 2), time(22, 49, 4))
    assert find_datetime("MASUK: 01 0ct 2026 08:10:45")[:2] == (date(2026, 10, 1), time(8, 10, 45))
    d, t, conf = find_datetime("28/09/2026 :36:4 14/0./01")
    assert d == date(2026, 9, 28) and t is None and conf > 0.9  # partial time is not invented
    assert find_datetime("24/01/01")[2] < 0.9                   # 2-digit year = weak evidence
    assert find_datetime("28/09/20261:36:414/0./01")[:2] == (date(2026, 9, 28), None)   # not 01:36:41
    # a 2-digit year whose space before the time was lost is not the year 1819
    assert find_datetime("11/02/1819:21")[:2] == (date(2018, 2, 11), time(19, 21))
    assert find_datetime("27/06/185:02:32 PM")[:2] == (date(2018, 6, 27), time(17, 2, 32))
    assert find_datetime("[Sales]2018-01-0522:55:25")[:2] == (date(2018, 1, 5), time(22, 55, 25))
    # numbers before the date must not use up its digits
    assert find_datetime("ORD #07 -REG #19- 10/03/2018 17:24:07")[:2] == (date(2018, 3, 10), time(17, 24, 7))
    assert find_datetime("TU21- 6/72")[0] is None              # mixed separators: a code, not a date
    d, _, conf = find_datetime("28 Mar 18 18:32:36")
    assert d == date(2018, 3, 28) and conf < 0.85                # 2-digit year: a person confirms it
    assert find_datetime("02/JAN/2017")[0] == date(2017, 1, 2)
    assert find_datetime("OCT 3, 2016")[0] == date(2016, 10, 3)
    assert find_datetime("1 SEPATU 25")[0] is None             # a fuzzy month needs a 4-digit year
    p = find_partial_date("02/10/202")
    assert (p.day, p.month, p.year) == (2, 10, None)
    p = find_partial_date("09/2026 08:50:22")
    assert (p.day, p.month, p.year) == (None, 9, 2026)


def test_misc_normalizers():
    assert terbilang_to_int("Enam Ratus Ribu Rupiah")[0] == 600_000
    assert terbilang_to_int("dua juta tiga ratus lima puluh ribu")[0] == 2_350_000
    assert terbilang_to_int("sebelas ribu lima ratus")[0] == 11_500
    assert parse_duration_minutes("Odays 7hours 29min") == 449
    assert parse_duration_minutes("3 hours 13 minutes") == 193
    assert parse_volume(": (l) 18.8") == 18.8 and parse_volume("3.30liter") == 3.3
    assert find_card_numbers("CN:01452022[I 0164{0 Rp.485[1") == ["01452022?10164?0"]
    assert station_code("3413703") == "34.137.03" and station_code("SPBU:34.13702") == "34.137.02"


# --------------------------------------------------------------------------- re-read amount zones
@pytest.mark.parametrize("text,expected", [
    ("Rp11000", 11000), ("Rpi1000", 11000), ("p11000", 11000), ("Ro1100D", 11000), ("_75500", 75500),
    ("Rp11.000", 11000),
    ("R1000", None),        # the 'p' is gone - a digit may have gone with it
    ("A75500", None), ("Roilo00", None),
])
def test_amount_through_marker(text, expected):
    assert glyph_amount(text)[0] == expected


@pytest.mark.parametrize("text,value,approx", [
    ("600.000,-", 600000, False), ("600.0001", 600000, False), ("Rp. 35.000", 35000, False), ("6oo.000", 600000, False),
    ("35.00", 35000, True), ("35.0.", 35000, True), ("3500", 3500, True),
    ("1.250.00", None, None), ("30.2", None, None), ("35", None, None),
])
def test_handwritten_amount(text, value, approx):
    a = hand_amount(text)
    assert (a.value, a.approx) == (value, approx) if value else a is None


def test_terbilang_needs_every_word():
    assert terbilang_to_int("Evaua Ratus Bibm")[1] < 1.0        # 'ratus' alone would read as 100
    readings = ["Enau Radus 3bm Rupnad", "Enam Radus lRibn Rupiad", "Enau Ratus libn Rupial", "Enam Radus lRibn Rupiad"]
    assert terbilang_vote(readings) == (600_000, "enam ratus ribu rupiah")
    assert terbilang_vote(["Enau Ratus Bibon Rya", "Enau Radus Bilon Ruyind", "Enau Ratus libon Rupind"])[0] is None
    assert terbilang_vote(readings[:1])[0] is None


def test_settle_zone_readings():
    zone = lambda v, conf, **kw: Cand(v, conf, "ocr_refined", **kw)
    # hidden from the first pass, three variants agree: reported, but below the review threshold
    c, share = settle([zone(11000, .92), zone(11000, .88), zone(11000, .87), zone(1000, .7)])
    assert (c.value, c.source) == (11000, "ocr_refined") and c.conf == 0.84 and share == 1.0
    # one variant alone is noise; so is any number of readings without a thousands separator
    assert settle([zone(30200, .7)]) is None
    assert settle([zone(3500, .8, weak=True), zone(3500, .8, weak=True)]) is None
    c, _ = settle([Cand(35000, .48, raw="35.000"), zone(3500, .8, raw="3500", weak=True), zone(3500, .8, weak=True)])
    assert c.value == 35000 and c.source == "ocr" and c.conf == .48
    # ... unless they are the digits of a reading that has one: then they back it up
    c, _ = settle([Cand(35000, .48, raw="35.00"), zone(3500, .6, raw="3500", weak=True), zone(3500, .5, raw="3500", weak=True)])
    assert c.value == 35000 and c.conf == .6
    # the first pass and the zones agree on a clean print: nothing is capped
    c, _ = settle([Cand(9500, .93), zone(9500, .97), zone(9500, .96)])
    assert c.source == "ocr" and c.conf == 0.97


def test_zone_cleanup():
    h = 40
    img = np.full((80, 600), 235, np.uint8)
    img[:, 150:400] = 120                               # marker band
    img[30:50:6, 160:390] = 40                          # hatching inside it
    img[20:60, 200:206] = img[20:60, 230:236] = 30      # two pen strokes
    img[20:60, 520:526] = 30                            # a signature further right
    flat = flatten_shading(img, h)
    assert flat[10, 300] > 200 and flat[40, 203] < 80    # band gone, stroke kept
    clean = remove_rules(img, h)
    assert clean[36, 320] > 100 and clean[36, 203] < 60  # hatch line gone, stroke through it kept
    x0, x1 = ink_extent(img, h, gap=1.0)
    assert x0 <= 200 and 236 <= x1 < 520                 # the block ends before the signature


# --------------------------------------------------------------------------- labels
@pytest.mark.parametrize("text,key", [
    ("Dibavar Konsumen", "paid"), ("Subsidi Pemer intah : 15,616", "subsidy"), ("No. Irans: 2669613", "transNo"),
    ("Waktu. 08/09/2026 22:27:34", "datetime"), ("Volum", "volume"), ("Biaya Parkir: Rp", "parkingFee"),
    ("TOTAL", "total"), ("CAR:D1099SBE", "plate"), ("Sudah terima darl", "receivedFrom"), ("Jumlah Rp.", "amountTotal"),
])
def test_label_match(text, key):
    hit = match_label(text)
    assert hit is not None and hit.key == key


def test_same_label_to_the_right_is_a_value_only_when_bare():
    def line(i, text, x0, x1):
        return Line(i, text, .95, np.array([[x0, 100], [x1, 100], [x1, 140], [x0, 140]], np.float32))
    # 'Operator' followed by the word OPERATOR: that is the cashier's name
    (u,) = build_units([line(0, "Operator", 0, 150), line(1, "OPERATOR", 260, 420)])
    assert u.key == "operator" and u.value_text == "OPERATOR"
    # 'Pulau/Pompa' whose own value was lost, next to the neighbouring receipt's 'Pulau/Pompa : 11'
    a, b = build_units([line(0, "Pulau/Pompa", 0, 200), line(1, "Pulau/Pompa : 11", 480, 760)])
    assert a.value_text == "" and b.value_text == "11" and len(a.lines) == len(b.lines) == 1


@pytest.mark.parametrize("text", ["Total Penjualan (Rp)", "Informasi Harga BBM (Rp/Liter)", ": PERTALITE",
                                  "Info Tol:1408)", "Harga produk Pertamax saat ini"])
def test_not_a_label(text):
    assert match_label(text) is None


def test_header_kinds():
    assert header_kind("PERTAMINA") == "anchor" and header_kind("3413703") == "anchor"
    assert header_kind("PERTAMAX") is None          # product name must not start a new receipt
    assert header_kind("JL. RAYA CIRACAS") == "header"


# --------------------------------------------------------------------------- solver
def _subsidized_rels():
    tol = 0.01
    return [
        Rel("R1", ("hns", "spl", "hj"), lambda v: abs(v["hns"] - v["spl"] - v["hj"]), True, "",
            {"hj": lambda v: v["hns"] - v["spl"]}),
        Rel("R2", ("ts", "st", "paid"), lambda v: abs(v["ts"] - v["st"] - v["paid"]) / 2, True, "",
            {"paid": lambda v: v["ts"] - v["st"]}),
        Rel("R3", ("vol", "hj", "paid"), lambda v: abs(v["vol"] * v["hj"] - v["paid"]) / (tol * v["hj"] + 2), False, ""),
    ]


def test_solver_drops_misread_digit_and_derives_it():
    # OCR misread Dibayar 34,000 as 84,000; two relations disagree with it
    cands = {"hns": [Cand(15892, .99)], "spl": [Cand(5892, .99)], "hj": [Cand(10000, .99)], "ts": [Cand(54032, .99)],
             "st": [Cand(20032, .99)], "paid": [Cand(84000, .95, raw="84,000")], "vol": [Cand(3.4, .97)]}
    rels = _subsidized_rels()
    d = Draft("t", "fuel", Segment([]))
    apply_solution(d, cands, solve(cands, rels), rels)
    assert d.val("paid") == 34000 and d.fields["paid"].source == "corrected"


def test_solver_prefers_consistent_candidate():
    cands = {"hns": [Cand(15892, .9)], "spl": [Cand(5892, .9)], "hj": [Cand(10000, .9), Cand(18000, .95)]}
    rels = _subsidized_rels()[:1]
    assert solve(cands, rels)["hj"].value == 10000


# --------------------------------------------------------------------------- batch
def _ev(rid, typ, at, amount=None, bal=None, pat=None, partial=None):
    d = Draft(typ, typ, Segment([]), amount_key="fare" if typ == "toll" else "fee")
    d.rid = rid
    d.fields["cardNumber"] = FV("0145202201016480", .95)
    if at:
        d.fields["transactionAt"] = FV(at, .95)
        d.fields["transactionDate"] = FV(at[:10], .95)
    if partial:
        d.fields["datePartial"] = FV(partial, .5)
    if amount is not None:
        d.fields[d.amount_key] = FV(amount, .9)
    if bal is not None:
        d.fields["balanceAfter"] = FV(bal, .9)
    if pat:
        d.fields["balanceAfter"] = FV(None, 0.0, raw=pat)
    return d


def test_card_chain_recovers_hidden_fare_and_unreadable_balance():
    b = _ev("B", "toll", None, amount=None, bal=59501, partial={"day": None, "month": 9, "year": 2026})
    a = _ev("A", "toll", "2026-09-28", amount=None, pat="485?1")       # fare under marker, digit unreadable
    p = _ev("P", "parking", "2026-09-29T17:51:11", amount=17000, bal=31501)
    run_batch_checks([b, a, p], date(2026, 10, 12))
    assert a.val("balanceAfter") == 48501 and a.fields["balanceAfter"].source == "inferred"
    assert a.val("fare") == 11000 and a.fields["fare"].source == "inferred"
    assert any(c.code == "CARD_BALANCE_CHAIN" and c.status == "pass" for c in p.checks)


def test_card_chain_confirms_completed_balance_with_second_link():
    b = _ev("B", "toll", "2026-09-27T08:50:22", amount=75500, bal=59501)
    a = _ev("A", "toll", "2026-09-28", amount=11000, pat="485?1")       # fare now read from under the marker
    p = _ev("P", "parking", "2026-09-29T17:51:11", amount=17000, bal=31501)
    run_batch_checks([b, a, p], date(2026, 10, 12))
    bal = a.fields["balanceAfter"]
    assert bal.value == 48501 and bal.confidence >= 0.97 and not a.review   # B->A completed it, A->P confirmed it
    assert a.fields["fare"].confidence >= 0.98


def test_card_chain_flags_odd_remainder_once():
    b = _ev("B", "toll", "2026-09-27T08:50:22", amount=75500, bal=59501)
    a = _ev("A", "toll", "2026-09-28T05:36:04", amount=11000, bal=48511)     # 48501 with a misread digit
    p = _ev("P", "parking", "2026-09-29T17:51:11", amount=17000, bal=31501)
    run_batch_checks([b, a, p], date(2026, 10, 12))
    warned = lambda d: any(c.code == "CARD_BALANCE_CHAIN" and c.status == "warn" for c in d.checks)
    assert warned(a) and not warned(p)                  # the same 10 on the next link is A's balance again
    top_up = _ev("T", "toll", "2026-09-30T10:00:00", amount=11000, bal=120501)
    run_batch_checks([p, top_up], date(2026, 10, 12))
    assert not warned(top_up)                           # +100,000 is a top-up, not a misread


def test_partial_year_and_future_date():
    x = _ev("X", "toll", "2026-09-30T10:00:00", amount=11000, bal=1000)
    y = _ev("Y", "toll", None, partial={"day": 2, "month": 10, "year": None})
    z = _ev("Z", "toll", "2027-01-01T10:00:00", amount=11000, bal=500)
    run_batch_checks([x, y, z], date(2026, 10, 12))
    assert y.val("transactionDate") == "2026-10-02" and y.fields["transactionDate"].source == "inferred"
    assert any(c.code == "DATE_NOT_IN_FUTURE" and c.status == "fail" for c in z.checks)


def test_duplicate_detection():
    def fuel(rid):
        d = Draft("fuel_subsidized", "fuel", Segment([]), amount_key="amountPaid")
        d.rid = rid
        for k, v in {"merchantId": "34.137.03", "transactionNo": "2739357", "transactionAt": "2026-09-24T21:38:53",
                     "transactionDate": "2026-09-24", "amountPaid": 34000}.items():
            d.fields[k] = FV(v, .99)
        return d
    a, b = fuel("a"), fuel("b")
    run_batch_checks([a, b], date(2026, 10, 12))
    assert any(c.code == "NOT_DUPLICATE" and c.status == "fail" for c in a.checks + b.checks)


# --------------------------------------------------------------------------- vlm merge
def test_vlm_fills_only_missing_and_never_overrides_verified():
    from docscan.config import Settings
    from docscan.templates.base import Check
    from docscan.vlm import VlmClient

    calls = {}

    def handler(request: httpx.Request):
        calls["auth"] = request.headers.get("authorization")
        content = '{"amount": "600.000", "transactionDate": "2026-10-02", "receivedFrom": "X"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    s = Settings(vlm_mode="needs_review", vlm_base_url="http://router/v1", vlm_api_key="k", vlm_model="m")
    client = VlmClient(s, transport=httpx.MockTransport(handler))
    data = client.extract(np.full((40, 40, 3), 255, np.uint8), "kwitansi")
    assert calls["auth"] == "Bearer k" and data["amount"] == "600.000"

    d = Draft("kwitansi", "other", Segment([]), amount_key="amount")
    d.fields["receivedFrom"] = FV("PT. Prambanan Kencana", .95)
    d.checks.append(Check("SOMETHING", "pass", "", ["transactionDate"]))  # protected field
    assert VlmClient.merge(d, data)
    assert d.val("amount") == 600000 and d.fields["amount"].source == "vlm"
    assert d.val("receivedFrom") == "PT. Prambanan Kencana"              # confident OCR kept
    assert "transactionDate" not in d.fields                              # protected by a passing check


def test_vlm_fills_the_optician_date_and_keeps_what_ocr_read():
    from docscan.vlm import VlmClient

    d = Draft("optical", "medical", Segment([]), amount_key="amount")
    d.fields["patientName"] = FV("Rita", 0.95)
    d.fields["amount"] = FV(350000, 0.6, "ocr_refined")
    data = {"transactionDate": "2026-10-02", "amount": "Rp 350.000,-", "patientName": "Rita",
            "serviceDescription": "Frame F8610, lensa Photochromic", "merchantName": "Optik Rizki eye+"}
    assert VlmClient.merge(d, data)
    assert d.val("transactionDate") == "2026-10-02" and d.fields["transactionDate"].source == "vlm"
    assert d.val("amount") == 350000 and d.val("patientName") == "Rita" and d.fields["patientName"].source == "ocr"
    assert d.val("serviceDescription") == "Frame F8610, lensa Photochromic"
    assert any("vision model" in r for r in d.review)              # filled by the model: still to confirm
