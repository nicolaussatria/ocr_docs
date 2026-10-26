"""What a receipt is, in words - the 'Keterangan' line (no OCR models needed)."""
import pytest

from docscan.describe import describe, kind_of
from docscan.layout import Segment
from docscan.templates.base import FV, Draft


def draft(doc_type, category, extra=None, **fields):
    d = Draft(doc_type, category, Segment([]))
    d.segment.extra.update(extra or {})
    for key, value in fields.items():
        d.fields[key] = value if isinstance(value, FV) else FV(value, 0.95)
    return d


@pytest.mark.parametrize("d,text", [
    (draft("fuel_subsidized", "fuel", product="PERTALITE", volumeLiters=3.2, merchantName="SPBU KRAMAT JATI NO.109"),
     "BBM Pertalite 3,2 L - SPBU Kramat Jati No.109"),
    (draft("fuel_retail", "fuel", product="PERTAMAX", volumeLiters=20.0, merchantName="SPBU CIPULARANG B"),
     "BBM Pertamax 20 L - SPBU Cipularang B"),
    (draft("fuel_cash_voucher", "fuel", product="PERTALITE", merchantName="SPBU 34.13702"),
     "BBM Pertalite - SPBU 34.13702 (bon kontan)"),
    (draft("toll", "toll", gate="PASTEUR", merchantName="JASA MARGA"), "Tol Pasteur"),
    (draft("toll", "toll", merchantName="JASA MARGA"), "Tol Jasa Marga"),
    (draft("toll", "toll"), "Tol"),
    (draft("parking", "parking", merchantName="PLAZA SENAYAN"), "Parkir Plaza Senayan"),
    (draft("parking", "parking", merchantName="PK-LG-MANLESS"), "Parkir"),            # a gate terminal is not a place
    (draft("mobile_topup", "pulsa", merchantName="TELKOMSEL", phoneNumber="081388633017"), "Pulsa Telkomsel 081388633017"),
    (draft("kwitansi", "pulsa"), "Pulsa (kwitansi)"),
    (draft("kwitansi", "other"), "Kwitansi"),
    (draft("receipt", "internet", merchantName="HYPERNET", items=[{"name": "FION INTERNET DEDICATED 50Mbps", "amount": 3000000}]),
     "Internet Hypernet: FION INTERNET DEDICATED 50Mbps"),
    (draft("receipt", "meal", items=[{"name": "Egg Tart", "amount": 1}, {"name": "Pizza Toast", "amount": 1}]),
     "Makan/minum: Egg Tart +1 lainnya"),
    (draft("receipt", "other", extra={"item_names": ["Choco Devil", "CP 360 Club Card"]}), "Choco Devil +1 lainnya"),
    (draft("receipt", "other"), "Struk belanja"),
    (draft("unknown", "other"), None),
])
def test_description(d, text):
    assert describe(d) == text


def test_description_leaves_out_what_was_read_with_doubt():
    d = draft("fuel_subsidized", "fuel", product="PERTALITE", merchantName=FV("SPBU C1RAC4S", 0.4))
    assert describe(d) == "BBM Pertalite"


@pytest.mark.parametrize("text,kind", [
    ("FION INTERNET DEDICATED IIX 50Mbps", "internet"), ("Token Listrik PLN 100.000", "utilities"),
    ("Paket Data 10GB", "pulsa"), ("HOTEL SANTIKA room rate", "lodging"), ("Ganti oli mesin + tune up", "vehicle_service"),
    ("Apotek K24", "medical"), ("NASI GORENG / ES TEH", "meal"),
    ("Service charge 5%", None),               # a restaurant's service charge is not a car service
    ("Sub Total / Total / Cash", None),
])
def test_kind_of_a_general_receipt(text, kind):
    assert kind_of(text) == kind
