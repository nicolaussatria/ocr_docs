"""What a document is, in words: the line a claimant would write under 'Keterangan'.

Two things are worked out here from what the templates already extracted:

* `kind_of`  - what a receipt without a template was for (internet, a meal, a hotel ...),
               from the words on it. It becomes the receipt's `category`.
* `describe` - one short Indonesian line per receipt: what was bought and where
               ('BBM Pertalite 3,2 L - SPBU Ciracas', 'Parkir Plaza Senayan').

Neither is a checked value. A description is a starting point for a person to edit, so it
is built only from fields the receipt gave with some confidence and stays empty otherwise.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:       # the templates import this module; importing them back at run time would loop
    from .templates.base import Draft

# Hospital, clinic, doctor and pharmacy papers - on a receipt, or in what a kwitansi was paid for.
MEDICAL = (r"apotek|apotik|klinik|rumah sakit|\brs[ua]?\b|puskesmas|dokter|\bdrg?\.|farmasi|\bobat\b|berobat|pengobatan"
           r"|rawat (?:jalan|inap)|laboratorium|hospital|clinic|pharmacy|medical|\bmedis\b|dokter gigi|optik|optical|kacamata")

# category -> (label for the description, words that give it away). Order matters: the first hit
# wins, specific before general - an internet invoice may well mention a 'service'.
_KINDS = [
    ("internet", "Internet", r"internet|broadband|wi-?fi|mbps|fiber|fibre|dedicated|\bisp\b"),
    ("utilities", "Listrik/air", r"\bpln\b|token listrik|listrik|\bkwh\b|\bpdam\b"),
    ("pulsa", "Pulsa", r"pulsa|paket data|kuota|voucher data"),
    ("lodging", "Hotel", r"hotel|penginapan|homestay|guest ?house|check[- ]?in|check[- ]?out|room rate"),
    ("transport", "Transport", r"kereta|pesawat|airlines?|boarding pass|\btaxi\b|\btaksi\b|bluebird|\bgrab\b|gojek|shuttle"),
    ("vehicle_service", "Servis kendaraan", r"bengkel|servi[cs]e? kendaraan|ganti oli|oli mesin|tune ?up|spooring|balancing"
                                            r"|spare ?part|suku cadang|kampas rem|\baki\b|\bbusi\b"),
    ("medical", "Medical", MEDICAL),
    ("office", "ATK", r"\batk\b|alat tulis|foto ?copy|fotokopi|kertas hvs|stationery"),
    ("meal", "Makan/minum", r"resto|restaurant|\bcafe\b|\bkafe\b|coffee|\bkopi\b|bakery|\broti\b|\bnasi\b|\bayam\b|\bbebek\b|\bmie\b"
                           r"|bakso|\bsate\b|\bsoto\b|\bteh\b|\btea\b|juice|\bjus\b|makan|minum|\bfood\b|dine in"
                           r"|take ?away|burger|pizza|chicken|\brice\b"),
]
_LABEL = {cat: label for cat, label, _ in _KINDS}
_KEEP_UPPER = {"SPBU", "PT", "CV", "TBK", "BCA", "BNI", "BRI", "PLN", "KAI", "ATK", "PDAM", "OVO", "XL", "II", "III", "IV"}


def kind_of(text: str) -> str | None:
    """Category of a receipt nobody wrote a template for, or None when its words do not say."""
    low = text.lower()
    return next((cat for cat, _, words in _KINDS if re.search(words, low)), None)


def _name(text: str | None) -> str:
    """'SPBU KRAMAT JATI NO.109' -> 'SPBU Kramat Jati No.109' (shouting is hard to read in a table)."""
    if not text:
        return ""
    words = [w if w.upper() in _KEEP_UPPER or not w.isupper() else w.capitalize() for w in str(text).split()]
    return " ".join(words)


def _clip(text: str, limit: int = 110) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -:;,")
    return text if len(text) <= limit else text[:limit - 1].rstrip(" -:;,") + "…"


def describe(d: Draft) -> str | None:
    """One line saying what the receipt is. None when there is nothing worth saying."""
    def val(key: str, min_conf: float = 0.6):
        f = d.fields.get(key)
        return f.value if f is not None and f.value not in (None, "") and f.confidence >= min_conf else None

    shop = _name(val("merchantName"))
    if re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", shop):
        shop = ""               # 'PK-LG-MANLESS' is a gate terminal, not a place anyone would write down
    if d.category == "fuel":
        liters = val("volumeLiters")
        what = " ".join(p for p in ("BBM", _name(val("product")), f"{liters:g} L".replace(".", ",") if liters else "") if p)
        tail = " (bon kontan)" if d.doc_type == "fuel_cash_voucher" else ""
        return _clip(f"{what} - {shop}{tail}" if shop else what + tail)
    if d.doc_type == "toll":
        return _clip(f"Tol {_name(val('gate')) or shop}".strip())
    if d.doc_type == "parking":
        return _clip(f"Parkir {shop}".strip())
    if d.doc_type == "mobile_topup":
        return _clip(" ".join(p for p in ("Pulsa", shop, str(val("phoneNumber") or "")) if p))
    if d.category == "medical":
        # the claim form's Keterangan for medical: hospital or clinic, its address, the doctor
        parts = [p for p in (shop, _name(val("merchantAddress")), val("doctorName")) if p]
        if parts:
            return _clip(" - ".join(parts), 200)
        return "Medical (kwitansi)" if d.doc_type == "kwitansi" else "Medical"
    if d.doc_type == "kwitansi":
        return "Pulsa (kwitansi)" if d.category == "pulsa" else "Kwitansi"
    if d.doc_type != "receipt":
        return None
    label = _LABEL.get(d.category, "")
    # the lines as read are good enough to say what was bought, even where they did not add up
    names = [i["name"] for i in val("items") or []] or list(d.segment.extra.get("item_names", []))
    what = names[0] + (f" +{len(names) - 1} lainnya" if len(names) > 1 else "") if names else ""
    head = " ".join(p for p in (label, shop) if p)
    if head or what:
        return _clip(f"{head}: {what}" if head and what else head or what)
    # nothing readable about the purchase: at least say what kind of paper it is
    if re.search(r"(?i)\b(invoice|faktur|tagihan)\b", d.segment.full_text):
        return _clip(f"Invoice {val('receiptNo') or ''}")
    return "Struk belanja"
