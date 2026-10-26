"""E-toll slips off a phone photo of a claim page: read the way the recogniser reads them (no OCR models needed)."""
import numpy as np

from docscan.layout import segment_page
from docscan.ocr import Line
from docscan.templates import classify
from docscan.templates.base import Ctx

CTX = Ctx(reread=lambda lines: [], refine=False)


def page(rows: str) -> list[Line]:
    out = []
    for r in rows.strip().splitlines():
        text, x0, x1, cy, h, score = r.split("|")
        x0, x1, cy, h = float(x0), float(x1), float(cy), float(h)
        q = np.array([[x0, cy - h / 2], [x1, cy - h / 2], [x1, cy + h / 2], [x0, cy + h / 2]], np.float32)
        out.append(Line(0, text, float(score), q))
    return out


# text | x0 | x1 | centre y | height | score, as read off the photo. 'l' comes back as '1' ('e-To11', 'GO1-1'),
# a letter goes missing ('No se-i'), the slips overlap so dates and the class are cut ('09/2026', '-1 e-T').
PASTEUR = """
24/01/01|645|709|302|26|1.00
09/2026 08:50:22|433|551|305|28|0.99
150637-250015|612|711|324|27|1.00
:003832|437|497|327|28|0.97
Gerbang:30 [PASTEUR]|441|606|347|30|0.96
e-To11 BCA75500|444|581|375|43|0.77
2201016480 Rp.59501|428|620|405|31|1.00
"""
CEMPAKA = """
PTC|1308|1338|484|19|1.00
lsrga sissphala Purs do Tbk|1354|1500|489|21|0.74
1:(0219381 838)|1353|1472|503|23|0.86
CA PUTIH|1356|1424|544|28|0.88
02/10/202|1262|1337|605|28|0.94
:33:2€ 02/0:/02|1354|1487|610|30|0.88
No se-i :|1261|1337|627|29|0.87
38 DC6364/001062|1356|1486|632|28|0.85
GO1-1 eB-T|1260|1339|656|45|0.62
Roilt00|1419|1489|664|41|0.69
CN:C14520|1261|1337|687|29|0.94
J1648C Rp.107!01|1354|1479|692|30|0.83
"""
# a toll slip printed close to the left of a parking ticket: mostly numbers, but captioned
TOLL_BESIDE_PARKING = """
: 29 Sep 2026 14:37:30 - M20|871|1105|12|24|0.98
In|804|831|14|25|1.00
:29 Sep 2026 17:51:11 - F21|869|1106|32|27|0.99
Out|805|837|34|24|1.00
Duration: 3 hours 13 minutes|803|1042|52|27|1.00
28/09/2026:36:414/0/01|490|705|60|27|1.00
No seri :01t:43 (07452/261603|489|705|82|26|0.93
17.000|967|1028|93|29|1.00
Biaya Parkir: Rp|804|945|95|28|0.99
GOL-1 e-Tol BCA|489|614|110|43|0.95
31.501|974|1032|134|25|1.00
Flazz Balance: Rp|806|954|135|24|0.99
CN:01452022 0164{0 Rp.485(1|490|693|140|27|0.91
Flazz CardNo : 0145202201016480|805|1067|155|26|0.98
TERIMA KASIH|954|1060|214|24|0.97
ATAS KUNJUNGAN ANDA|924|1089|234|25|1.00
"""


def only(rows):
    (seg,) = segment_page(page(rows))
    return seg


def test_a_photographed_toll_slip_is_still_a_toll_slip():
    for rows in (PASTEUR, CEMPAKA):
        assert classify(only(rows))[0].doc_type == "toll"


def test_fare_and_gate_off_the_photo():
    seg = only(PASTEUR)
    d = classify(seg)[0].parse(seg, CTX)
    assert d.val("fare") == 75500 and d.val("gate") == "PASTEUR"
    assert d.val("transactionDate") is None                # '09/2026': the day is under the next slip
    assert any("partly unreadable" in r for r in d.review)


def test_a_toll_slip_is_not_taken_for_a_column_of_prices():
    segs = segment_page(page(TOLL_BESIDE_PARKING))
    kinds = sorted(classify(s)[0].doc_type for s in segs)
    assert kinds == ["parking", "toll"]
    toll = next(s for s in segs if classify(s)[0].doc_type == "toll")
    assert all(ln.x1 < 800 for ln in toll.lines)             # nothing of the parking ticket in it
