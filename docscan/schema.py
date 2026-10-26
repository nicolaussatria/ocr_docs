"""Public JSON contract (camelCase keys – binds cleanly to Dart json_serializable and
ASP.NET System.Text.Json web defaults). Bump SCHEMA_VERSION on breaking changes."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

SCHEMA_VERSION = "1.0"

Status = Literal["verified", "needs_review"]
CheckStatus = Literal["pass", "fail", "warn", "skip"]
Source = Literal["ocr", "ocr_refined", "derived", "corrected", "inferred", "batch", "vlm"]


class _M(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, serialize_by_alias=True)


class FieldOut(_M):
    value: Any = Field(description="Normalised value (int for IDR amounts, ISO-8601 for dates/times)")
    confidence: float = Field(ge=0, le=1)
    source: Source = Field(description="ocr | ocr_refined (re-read with image variants) | derived (computed from "
                                       "other fields) | corrected (one-digit OCR fix proven by arithmetic) | "
                                       "inferred (batch context, e.g. card balance chain) | vlm")
    raw: str | None = Field(None, description="OCR text the value came from")
    bbox: list[int] | None = Field(None, description="[x0,y0,x1,y1] in the ORIGINAL image pixels")


class CheckOut(_M):
    code: str
    status: CheckStatus
    message: str
    fields: list[str] = []


class SourceRef(_M):
    file: str
    page: int


class ReceiptOut(_M):
    id: str
    source: SourceRef
    doc_type: str = Field(description="fuel_subsidized | fuel_retail | fuel_cash_voucher | toll | parking | optical | "
                                      "kwitansi | mobile_topup | receipt (any other receipt or invoice with "
                                      "labelled money rows) | unknown")
    category: str = Field(description="fuel | toll | parking | pulsa | vehicle_service | internet | utilities | "
                                      "lodging | transport | medical | office | meal | other - what the "
                                      "expense was for; the last eight are read off the words of a general receipt")
    status: Status
    confidence: float
    amount: int | None = Field(None, description="Evidence-backed transaction amount in IDR")
    claimed_amount: int | None = Field(None, description="Handwritten amount written over the printed one, if any")
    suggested_amount: int | None = Field(None, description="Best guess when `amount` is null because it could not be "
                                                           "read reliably. Never part of the totals: pre-fill it "
                                                           "for a person to confirm")
    currency: str = "IDR"
    transaction_date: str | None = None
    transaction_at: str | None = None
    merchant_name: str | None = None
    merchant_id: str | None = None
    handwritten: bool = False
    description: str | None = Field(None, description="One line saying what the receipt is ('BBM Pertalite 3,2 L - "
                                                      "SPBU Ciracas'): a starting point for a remark / "
                                                      "'Keterangan' column, not a checked value")
    fingerprint: str | None = Field(None, description="Stable hash for duplicate detection across uploads")
    bbox: list[int] = Field(description="Receipt region in the ORIGINAL image pixels")
    fields: dict[str, FieldOut] = {}
    checks: list[CheckOut] = []
    review_reasons: list[str] = []
    image: str | None = Field(None, description="Base64 JPEG crop (upright) when includeImages=true")
    raw_text: list[str] | None = None


class PageOut(_M):
    file: str
    page: int
    width: int
    height: int
    rotation: int = Field(description="Clockwise degrees applied to make the page upright")
    ocr_lines: int
    background_lines_ignored: int = Field(0, description="Text lines outside detected receipt sheets excluded before parsing")
    receipt_ids: list[str]
    elapsed_ms: int


class CategoryTotal(_M):
    count: int
    amount: int


class Summary(_M):
    receipt_count: int
    verified_count: int
    needs_review_count: int
    total_amount: int = Field(description="Sum of `amount` over receipts that have one")
    verified_amount: int
    by_category: dict[str, CategoryTotal]
    date_from: str | None = None
    date_to: str | None = None


class ScanResult(_M):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    engine: str
    processed_at: str
    elapsed_ms: int
    summary: Summary
    receipts: list[ReceiptOut]
    pages: list[PageOut]
    batch_checks: list[CheckOut] = []
    warnings: list[str] = []
