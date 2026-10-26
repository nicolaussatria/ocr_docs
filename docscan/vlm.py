"""Optional vision-LLM fallback for receipts OCR cannot settle (handwriting, marker
highlights, torn edges). Talks to any OpenAI-compatible /chat/completions endpoint
(an LLM router, OpenRouter, vLLM, Ollama ...). Off by default.

Rules that keep it from *adding* errors:
  * only fills fields that are missing or below `REPLACE_BELOW` confidence
  * never overrides a value that passed an arithmetic / chain check
  * anything it fills is marked source='vlm' and the receipt stays needs_review
"""
from __future__ import annotations

import json
import re

import httpx
import numpy as np

from .config import Settings
from .imaging import to_jpeg_b64
from .templates.base import FV, PASS, Draft

REPLACE_BELOW = 0.7

PROMPT = """You read Indonesian expense receipts (SPBU fuel receipts, e-toll slips, parking tickets,
kwitansi, bon kontan, hospital / clinic / pharmacy / optician receipts). Return ONLY a JSON object with
these keys (null when unreadable, never guess):
{"transactionDate": "YYYY-MM-DD", "transactionTime": "HH:MM:SS", "amount": <integer IDR actually paid>,
 "handwrittenAmount": <integer IDR if a handwritten amount overrides a printed one, else null>,
 "amountInWords": "<terbilang text if present>", "merchantName": "...", "merchantAddress": "...",
 "receivedFrom": "...", "purpose": "...", "volumeLiters": <number or null>, "product": "...",
 "fare": <integer or null>, "balanceAfter": <integer or null>,
 "patientName": "<patient named on a medical receipt>", "doctorName": "<doctor, with title, e.g. dr. Budi, Sp.PD>",
 "serviceDescription": "<what was paid for: services, medicines, or glasses frame + lens>"}
Amounts: '75.500' and '75,500' both mean 75500; 'Rp 350.000,-' means 350000. Do not convert or round.
Dates are day first. A handwritten date may be written as 'Place, dd/mm 20yy' with the day above and the
month below a slash, next to a printed '20' followed by the handwritten year."""

# which VLM keys may fill which draft fields
FIELD_MAP = {
    "transactionDate": "transactionDate", "transactionTime": "transactionTime", "merchantName": "merchantName",
    "receivedFrom": "receivedFrom", "purpose": "purpose", "amountInWords": "amountInWords",
    "volumeLiters": "volumeLiters", "product": "product", "fare": "fare", "balanceAfter": "balanceAfter",
    "handwrittenAmount": "amountClaimed",
    "patientName": "patientName", "doctorName": "doctorName", "merchantAddress": "merchantAddress",
    "serviceDescription": "serviceDescription",
}


class VlmClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.s = settings
        headers = {"Authorization": f"Bearer {settings.vlm_api_key}"} if settings.vlm_api_key else {}
        self.http = httpx.Client(base_url=settings.vlm_base_url.rstrip("/"), headers=headers,
                                 timeout=settings.vlm_timeout_s, transport=transport)

    def extract(self, crop: np.ndarray, doc_type_hint: str) -> dict | None:
        b64 = to_jpeg_b64(crop, max_side=1400, quality=85)
        body = {
            "model": self.s.vlm_model,
            "temperature": 0,
            "max_tokens": 800,                  # some endpoints (Anthropic's) require a limit
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Document type guess: {doc_type_hint}. Return the JSON object."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        }
        r = self.http.post("/chat/completions", json=body)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        m = re.search(r"\{.*\}", content, re.S)
        return json.loads(m.group(0)) if m else None

    @staticmethod
    def merge(d: Draft, data: dict) -> bool:
        protected = {f for c in d.checks if c.status == PASS for f in c.fields}
        changed = False

        def put(key: str, value):
            nonlocal changed
            if value in (None, "", []) or key in protected:
                return
            cur = d.fields.get(key)
            if cur is not None and cur.value is not None and cur.confidence >= REPLACE_BELOW:
                if str(cur.value) != str(value):
                    d.review.append(f"Vision model read {key} as {value!r} (OCR: {cur.value!r})")
                return
            d.fields[key] = FV(value, 0.75, "vlm", "vision model")
            changed = True

        for src, dst in FIELD_MAP.items():
            put(dst, _norm(src, data.get(src)))
        if d.amount_key:
            put(d.amount_key, _norm("amount", data.get("amount")))
        if d.val("transactionDate") and "transactionAt" not in d.fields:
            t = d.val("transactionTime")
            d.fields["transactionAt"] = FV(f"{d.val('transactionDate')}T{t}" if t else d.val("transactionDate"),
                                           0.75, "vlm")
        if changed:
            d.review.append("Some fields were filled by the vision model - confirm against the image")
        return changed


def _norm(key: str, v):
    if v is None:
        return None
    if key in ("amount", "fare", "balanceAfter", "handwrittenAmount"):
        if isinstance(v, str):
            digits = re.sub(r"[^\d]", "", v.split(",00")[0])
            return int(digits) if digits else None
        return int(v) if isinstance(v, (int, float)) else None
    if key == "volumeLiters":
        try:
            return float(str(v).replace(",", "."))
        except ValueError:
            return None
    if key == "transactionDate":
        return v if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)) else None
    if key == "transactionTime":
        return v if re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", str(v)) else None
    return str(v).strip() or None
