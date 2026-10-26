"""Invoices: a header table, labelled fields and one document spread over the page (no OCR models needed)."""
import numpy as np
import pytest

from docscan.layout import Segment, Unit, segment_page
from docscan.ocr import Line
from docscan.templates import assemble, classify
from docscan.templates.base import Ctx
from docscan.templates.receipt import Receipt, _label, key_of, labelled, role_of

CTX = Ctx(reread=lambda lines: [], refine=False)


def line(text, x0, x1, cy, h=20, score=0.99):
    y0 = cy - h / 2
    return Line(0, text, score, np.array([[x0, y0], [x1, y0], [x1, y0 + h], [x0, y0 + h]], np.float32))


# What the OCR returns for an internet provider's invoice (text | x0 | x1 | centre y | height). The logo
# is misread, the bold 'NPWP' caption is not detected at all, and one month comes back as '0ct'.
INVOICE = """
HYPERINEI|94|397|18|35
an|97|134|58|26
XLSMART company|170|405|58|33
www.hypernet.co.id|1100|1287|66|22
Kode Pelanggan/Customer ID|964|1249|141|20
INVOICE|17|219|148|37
10025007|1055|1154|178|23
Tanggal Cetak Tagihan/|26|268|222|23
Tanggal Jatuh Tempo/|296|515|222|23
No. Invoice/|543|681|220|24
Total Tagihan Bulan Ini/|719|982|222|23
Statement Date|71|226|259|20
Due Date|362|454|259|20
Invoice No.|553|674|258|23
Total Amount|787|917|258|17
01 October 2026|65|230|296|20
31 October 2026|325|490|296|20
9000053072|557|670|296|21
3.330.000|838|941|296|23
001.313.261.8-073.000|1024|1266|296|17
Kepada/To|14|119|334|22
Customer Service|953|1141|338|19
PT PRAMBANAN KENCANA|14|235|370|20
Tel: 021-29393939. Fax:|921|1171|374|20
Jl. Tanah Abang II No.36 Gambir Kota Jakarta Pusat, Daerah Khusus Ibukota|15|802|409|20
Jakarta , Jakarta Pusat 10160|16|327|446|20
Keterangan/ Description|117|373|562|20
Periode/ Period|523|694|562|23
Pemakaian/|748|870|562|23
Harga/Unit/|913|1044|563|24
Total (Rupiah)|1097|1261|562|23
Usage|780|843|596|23
Price/Unit(Rupiah)|882|1074|596|19
FION INTERNET DEDICATED IIX 50Mbps INT 50|16|460|638|20
01 0ct 2026 - 31 Oct 2026|484|739|638|17
1.0000|767|834|638|21
3.000.000|960|1056|638|20
3.000.000|1182|1279|639|20
Mbps Site Pergudangan Sirie (IP Public/30)|16|466|668|20
Sub Total|900|1009|700|23
3.000.000|1177|1281|700|20
Discount|911|1009|734|19
0|1263|1279|734|19
Subject to Tax|845|1011|766|23
3.000.000|1176|1279|766|23
VAT|967|1011|801|22
330.000|1198|1279|800|21
Pelunasan Bea Materai/Stamp Duty|643|1010|834|22
0|1263|1279|834|20
Delivery|910|1009|867|24
0|1263|1281|866|20
Tagihan Bulan Ini/Total Amount|670|1009|899|20
3.330.000|1177|1280|899|20
"""


def invoice_lines():
    rows = [r.split("|") for r in INVOICE.strip().splitlines()]
    return [line(t, float(x0), float(x1), float(cy), float(h)) for t, x0, x1, cy, h in rows]


def test_invoice_is_read_as_one_document():
    pieces = segment_page(invoice_lines())
    assert len(pieces) > 1                                         # letterhead, header table, lines ...
    docs = assemble(pieces)
    assert [classify(s)[0].doc_type for s in docs] == ["receipt"]
    d = Receipt().parse(docs[0], CTX)
    got = {k: d.val(k) for k in ("total", "subtotal", "discount", "tax", "taxBase", "transactionDate", "dueDate", "receiptNo",
                                 "customerName", "customerId", "npwp", "merchantName")}
    assert got == {"total": 3330000, "subtotal": 3000000, "discount": 0, "tax": 330000, "taxBase": 3000000,
                   "transactionDate": "2026-10-01",                # the statement date - not the due date, not the period
                   "dueDate": "2026-10-31", "receiptNo": "9000053072", "customerName": "PT PRAMBANAN KENCANA",
                   "customerId": "10025007", "npwp": "001.313.261.8-073.000",
                   "merchantName": "HYPERNET"}                     # from the web address; the logo reads 'HYPERINEI'
    assert d.val("items") == [{"name": "FION INTERNET DEDICATED IIX 50Mbps INT 50 Mbps Site Pergudangan Sirie (IP Public/30)",
                               "amount": 3000000}]
    assert "otherCharges" not in d.fields                          # stamp duty 0 and delivery 0 say nothing
    passed = {c.code for c in d.checks if c.status == "pass"}
    assert passed >= {"PRINTED_TWICE_EQ_TOTAL", "SUBTOTAL_TAX_EQ_TOTAL", "TAX_EQ_RATE_OF_BASE", "ITEMS_SUM_EQ_TOTAL"}
    assert d.fields["total"].confidence >= 0.99 and not d.review


@pytest.mark.parametrize("text,key", [
    ("Tanggal Cetak Tagihan/", "transactionDate"), ("Statement Date", "transactionDate"), ("Tanggal", "transactionDate"),
    ("Tanggal Jatuh Tempo/", "dueDate"), ("Due Date", "dueDate"), ("No. Invoice/", "receiptNo"), ("Invoice No.", "receiptNo"),
    ("Nomor Faktur", "receiptNo"), ("Kode Pelanggan/Customer ID", "customerId"), ("Kepada/To", "customerName"),
    ("Bill To", "customerName"), ("NPWP", "npwp"), ("Total Tagihan Bulan Ini/", "printedTotal"),
    ("Customer Service", None), ("Periode/ Period", None), ("Tel: 021-29393939. Fax:", None), ("Tomato Soup", None),
])
def test_header_field_names(text, key):
    hit = key_of(text)
    assert (hit[0] if hit else None) == key


def test_header_values_after_a_colon_beside_or_underneath():
    lines = [line("Invoice No: INV/2026/0042", 20, 330, 100),                       # after the colon
             line("Tanggal", 20, 110, 140), line("12/10/2026 14:33", 300, 480, 141),  # beside
             line("Kepada/To", 20, 120, 200), line("PT MAJU JAYA", 20, 180, 232),     # underneath
             line("Jatuh Tempo", 500, 640, 200), line("26 Oktober 2026", 500, 680, 232)]
    got = {k: v for k, (v, _) in labelled(lines).items()}
    assert got == {"receiptNo": "INV/2026/0042", "transactionDate": "2026-10-12", "customerName": "PT MAJU JAYA",
                   "dueDate": "2026-10-26"}


def test_due_date_is_never_taken_for_the_document_date():
    seg = Segment([Unit([ln]) for ln in (line("PT SUMBER AIR", 20, 220, 20), line("Jl. Melati No. 3", 20, 220, 50),
                                         line("Jatuh Tempo", 20, 160, 100), line("31/10/2026", 300, 420, 100),
                                         line("Periode", 20, 110, 130), line("01/10/2026 - 31/10/2026", 300, 560, 130),
                                         line("Total", 20, 90, 200), line("150.000", 300, 400, 200))])
    d = Receipt().parse(seg, CTX)
    assert d.val("total") == 150000 and d.val("dueDate") == "2026-10-31" and d.val("transactionDate") is None


def test_summary_rows_of_an_invoice():
    assert role_of(_label("Subject to Tax")) == "taxbase"            # not a misspelt 'total'
    assert role_of(_label("Pelunasan Bea Materai/Stamp Duty")) == "fee" and role_of(_label("Delivery")) == "fee"
    assert role_of(_label("Tagihan Bulan Ini/Total Amount")) == "total" and role_of(_label("VAT")) == "tax"


def _receipt(x, total, shop):
    rows = [(shop, None), ("Jl. Mawar No. 1", None), ("Kopi", total), ("Total", total)]
    out = []
    for i, (name, amount) in enumerate(rows):
        out.append(line(name, x, x + 13 * len(name), 100 + 34 * i))
        if amount:
            out.append(line(amount, x + 260, x + 340, 100 + 34 * i))
    return out


def test_pieces_are_joined_only_when_they_bring_no_amount_of_their_own():
    letterhead = [line("TOKO MAJU", 40, 200, 20), line("Jakarta Pusat 10160", 40, 300, 50), line("12/10/2026", 40, 180, 80)]
    body = [line("Kopi", 40, 100, 400), line("23.000", 300, 380, 400), line("Total", 40, 110, 440), line("23.000", 300, 380, 440)]
    pieces = [Segment([Unit([ln]) for ln in part]) for part in (letterhead, body)]
    (doc,) = assemble(pieces)                                        # a postcode is not a price
    assert len(doc.lines) == 7
    other = [line("Teh", 700, 760, 20), line("18.000", 900, 980, 20)]   # a second receipt nobody could classify
    pieces = [Segment([Unit([ln]) for ln in part]) for part in (other, body)]
    assert len(assemble(pieces)) == 2
    two = segment_page(_receipt(40, "23.000", "TOKO SATU") + _receipt(700, "18.000", "TOKO DUA"))
    assert len(assemble(two)) == 2                                   # two receipts on a page stay two
