"""Medical receipts: what Claim Medical Karyawan needs from them (no OCR models needed)."""
import numpy as np

from docscan.describe import describe
from docscan.layout import segment_page
from docscan.ocr import Line
from docscan.templates import assemble, classify
from docscan.templates.base import Ctx

CTX = Ctx(reread=lambda lines: [], refine=False)


def page(rows: str) -> list[Line]:
    out = []
    for r in rows.strip().splitlines():
        text, x0, x1, cy = r.split("|")
        x0, x1, cy, h = float(x0), float(x1), float(cy), 22.0
        out.append(Line(0, text, 0.97, np.array([[x0, cy - h / 2], [x1, cy - h / 2], [x1, cy + h / 2], [x0, cy + h / 2]], np.float32)))
    return out


def read(rows: str):
    (seg,) = assemble(segment_page(page(rows)))
    tmpl = classify(seg)[0]
    return tmpl.parse(seg, CTX)


CLINIC = """
KLINIK PRATAMA SEHAT SENTOSA|40|480|30
Jl. Raya Bogor No. 12, Kramat Jati|40|420|60
Jakarta Timur 13510 Telp. 021-8090123|40|470|90
KWITANSI PEMBAYARAN|40|300|140
Tanggal : 02/09/2026|40|300|175
Nama Pasien : Andi Setiawan|40|380|205
Dokter : dr. Budi Santoso, Sp.PD|40|420|235
Konsultasi Dokter Spesialis|40|360|290
350.000|520|610|290
Obat Amoxicillin 500mg|40|330|320
250.000|520|610|320
Pemeriksaan Laboratorium|40|340|350
100.000|520|610|350
TOTAL|40|120|400
700.000|520|610|400
"""

PHARMACY = """
APOTEK KIMIA FARMA 312|40|360|30
Jl. Tebet Raya No. 5 Jakarta Selatan|40|460|60
19/09/2026 14:22|40|240|95
Paracetamol 500mg|40|280|140
12.000|420|500|140
OBH Combi Sirup|40|250|170
23.500|420|500|170
TOTAL|40|120|210
35.500|420|500|210
"""


def test_clinic_receipt_gives_everything_the_medical_form_asks_for():
    d = read(CLINIC)
    assert d.doc_type == "receipt" and d.category == "medical"
    assert d.val("total") == 700000 and d.val("transactionDate") == "2026-09-02"
    assert d.val("patientName") == "Andi Setiawan"
    assert d.val("doctorName") == "dr. Budi Santoso, Sp.PD"
    assert d.val("merchantAddress") == "Jl. Raya Bogor No. 12, Kramat Jati, Jakarta Timur 13510"
    assert d.val("serviceDescription") == "Konsultasi Dokter Spesialis, Obat Amoxicillin 500mg, Pemeriksaan Laboratorium"
    # Keterangan: hospital or clinic + address + doctor
    assert describe(d) == ("Klinik Pratama Sehat Sentosa - Jl. Raya Bogor No. 12, Kramat Jati, Jakarta Timur 13510"
                           " - dr. Budi Santoso, Sp.PD")


def test_pharmacy_receipt_without_patient_or_doctor():
    d = read(PHARMACY)
    assert d.category == "medical" and d.val("total") == 35500
    assert d.val("serviceDescription") == "Paracetamol 500mg, OBH Combi Sirup"
    assert d.val("patientName") is None and d.val("doctorName") is None        # not on the paper: not guessed
    assert describe(d) == "Apotek Kimia Farma 312 - Jl. Tebet Raya No. 5 Jakarta Selatan"


def test_a_kind_of_patient_is_not_a_name():
    d = read(CLINIC.replace("Nama Pasien : Andi Setiawan", "Pasien : Umum").replace("Dokter : dr. Budi Santoso, Sp.PD", "Dokter Umum"))
    assert d.val("patientName") is None and d.val("doctorName") is None


# --------------------------------------------------------------------------- optician's handwritten receipt
from docscan.templates.medical import OpticalForm, _lens, _price  # noqa: E402

# text | x0 | x1 | centre y, as read off a phone photo of an Optik receipt (handwriting: Rita, F8610, Photocromic, Rp 350.000,-)
OPTIK = """
OPTIK|220|331|221
Vetern. 02028.|674|937|229
Rizki|336|500|233
eye+|334|504|277
NO:256651|790|930|304
SOLUSI MATA KELUARGA.ANDA|225|502|310
Kantor : Jakarta - 0851 3854 6518|224|525|338
MENERIMA RESEP DOKTER|224|487|366
HP=0821-5109-6882|513|767|390
Rita|307|383|395
Nama|224|283|402
Palembang|293|462|426
Alamat|225|300|428
01981|298|390|450
Frame|224|290|456
Lensa|225|289|484
SISA|579|631|506
10.350.06P0|296|474|507
Harga|224|288|513
CYL|345|405|551
-2.25|241|314|582
NOTE:|226|297|685
"""
# what re-reading each dotted line gives (handwriting variants), by the caption it follows
REREADS = {"Lensa": [("Photoenmc", 0.72), ("Photocnonc", 0.83), ("Photocrne", 0.69)],
           "Harga": [("3500O", 0.7), ("3500O", 0.71), ("35000", 0.68), ("3500", 0.6)],
           "Frame": [("aGlog", 0.62), ("RGlO", 0.67)]}


def optik(rereads=REREADS):
    from docscan.layout import Segment, Unit
    lines = page(OPTIK)
    seg = Segment([Unit([ln]) for ln in lines])

    def read_zone(quad, **kw):             # which caption's line is being re-read: the band's lower edge tells
        bottom = float(quad[2][1])
        cap = min(("Lensa", 484), ("Harga", 513), ("Frame", 456), ("Nama", 402), key=lambda c: abs(bottom - 0.5 * 22 - c[1]))
        return list(rereads.get(cap[0], []))
    return classify(seg)[0], OpticalForm().parse(seg, Ctx(reread=lambda lines: [], refine=True, read_zone=read_zone))


def test_an_optician_receipt_is_a_medical_claim():
    tmpl, d = optik()
    assert isinstance(tmpl, OpticalForm) and d.category == "medical"
    assert d.val("patientName") == "Rita" and d.val("receiptNo") == "256651"
    assert d.val("merchantName") == "OPTIK RIZKI EYE+" and d.val("merchantAddress") == "Jakarta"
    assert d.val("serviceDescription") == "Kacamata: frame + lensa Photochromic"     # the frame code is not legible
    assert d.val("amount") == 350000 and d.fields["amount"].confidence == 0.6       # handwriting: confirm
    assert d.val("transactionDate") is None                                          # '02/10' written as a fraction
    assert describe(d) == "Optik Rizki Eye+ - Jakarta"


def test_without_readings_that_agree_the_price_is_only_a_guess():
    _, d = optik({"Harga": [("12", 0.7)]})
    assert d.val("amount") == 350000 and d.fields["amount"].confidence == 0.45


def test_handwritten_price_and_lens_readings():
    assert [_price(t) for t in ("10.350.06P0", "Rp 350.000,-", "350060", "3500O", "12.500.000", "1.234.567")] == \
        [350000, 350000, 350000, None, 12500000, None]
    assert [_lens(t) for t in ("Photocnonc", "Photoermc", "bluecut", "Phokon", "aGlog")] == \
        ["Photochromic", "Photochromic", "Bluecut", None, None]
