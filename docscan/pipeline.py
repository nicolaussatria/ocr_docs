"""Orchestration: files -> pages -> upright OCR -> receipt segments -> template
parsing (+ constraint-guided re-OCR) -> batch checks -> optional VLM -> JSON."""
from __future__ import annotations

import logging
import time as _time
import uuid
from datetime import date, datetime, timezone

import numpy as np

from .batch import fingerprint, run_batch_checks
from .config import Settings
from .describe import describe
from .foreground import receipt_foreground
from .imaging import (InputError, Page, auto_levels, bbox_to_original, crop_box, crop_quad, enhance_variants, flatten_page,
                      handwritten_date_variants, ink_extent, load_pages, rotate, to_jpeg_b64, zone_variants)
from .layout import Segment, segment_page
from .normalize import mask_card
from .ocr import Line, OcrEngine, merge_lines
from .schema import (CategoryTotal, CheckOut, FieldOut, PageOut, ReceiptOut, ScanResult, SourceRef, Summary)
from .templates import KEY_FIELDS, Ctx, Draft, assemble, classify
from .templates.base import FAIL, PASS, WARN

log = logging.getLogger(__name__)


class DocScanner:
    def __init__(self, settings: Settings | None = None, engine: OcrEngine | None = None):
        self.settings = settings or Settings.from_env()
        self.ocr = engine or OcrEngine(self.settings)
        self.vlm = None
        if self.settings.vlm_mode != "off" and self.settings.vlm_base_url:
            from .vlm import VlmClient
            self.vlm = VlmClient(self.settings)

    # ------------------------------------------------------------------ public
    def scan(self, files: list[tuple[str, bytes]], *, include_images: bool = False, include_raw_text: bool = False,
             today: date | None = None) -> ScanResult:
        t0 = _time.perf_counter()
        st = self.settings
        if len(files) > st.max_files:
            raise InputError(f"too many files ({len(files)} > {st.max_files})")
        warnings: list[str] = []
        pages: list[Page] = []
        for name, data in files:
            if len(data) > st.max_upload_mb * 1024 * 1024:
                raise InputError(f"{name}: larger than {st.max_upload_mb} MB")
            pages += load_pages(data, name, max_pixels=st.max_pixels, pdf_dpi=st.pdf_dpi,
                                max_pdf_pages=st.max_pdf_pages)
        drafts: list[Draft] = []
        page_out: list[PageOut] = []
        for p in pages:
            pd, po = self._scan_page(p, len(drafts))
            drafts += pd
            page_out.append(po)
            if po.background_lines_ignored:
                warnings.append(f"{p.source} p{p.page_no}: ignored {po.background_lines_ignored} text lines outside receipt sheets")
            if not pd:
                warnings.append(f"{p.source} p{p.page_no}: no receipt detected")
        today = today or datetime.now(timezone.utc).date()
        batch_checks = run_batch_checks(drafts, today, self.settings.date_outlier_days)
        receipts = [self._finalize(d, include_images, include_raw_text) for d in drafts]
        if self.vlm is not None:
            receipts = self._vlm_pass(drafts, receipts, include_images, include_raw_text, warnings)
        return ScanResult(
            request_id=uuid.uuid4().hex[:12],
            engine=self.ocr.version + (f" + vlm:{st.vlm_model}" if self.vlm else ""),
            processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            elapsed_ms=int((_time.perf_counter() - t0) * 1000),
            summary=_summary(receipts),
            receipts=receipts,
            pages=page_out,
            batch_checks=[CheckOut(code=c.code, status=c.status, message=c.message, fields=c.fields) for c in batch_checks],
            warnings=warnings,
        )

    # ------------------------------------------------------------------ per page
    def _scan_page(self, page: Page, offset: int) -> tuple[list[Draft], PageOut]:
        t0 = _time.perf_counter()
        img = auto_levels(page.image)       # a dark or faded photo would otherwise show the detector no text
        rot = self.ocr.detect_orientation(img)
        up = rotate(img, rot)
        lines = self.ocr.read(up)
        foreground = receipt_foreground(up, lines)
        if foreground is not None:
            up = foreground.image
            lines = self.ocr.read(up)
        cache: dict[int, list[tuple[str, float]]] = {}

        def reread(lns: list[Line]) -> list[tuple[str, float]]:
            out = []
            for ln in lns:
                if id(ln) not in cache:
                    crop = crop_quad(up, ln.quad, pad=0.12)
                    cache[id(ln)] = self.ocr.recognize(enhance_variants(crop))
                out += cache[id(ln)]
            return out

        zones: dict[tuple, list[tuple[str, float]]] = {}

        def read_zone(quad: np.ndarray, *, h: float, stretch: tuple[float, ...] = (1.0,), rules: bool = False,
                      tighten: float | None = None, handwriting: bool = False, date_line: bool = False) -> list[tuple[str, float]]:
            key = (tuple(np.round(quad).astype(int).ravel()), round(h, 1), stretch, rules, tighten, handwriting, date_line)
            if key not in zones:            # a second parse of the page asks for the same zones again
                zones[key] = _read_zone(quad, h=h, stretch=stretch, rules=rules, tighten=tighten,
                                        handwriting=handwriting, date_line=date_line)
            return list(zones[key])

        def _read_zone(quad: np.ndarray, *, h: float, stretch: tuple[float, ...], rules: bool,
                       tighten: float | None, handwriting: bool, date_line: bool) -> list[tuple[str, float]]:
            crop = crop_quad(up, quad)
            if tighten is not None:
                span = ink_extent(crop, h, gap=tighten)
                if span is None:
                    return []
                crop = crop[:, span[0]:span[1]]
            variants = (handwritten_date_variants(crop, stretch=stretch) if date_line else
                        zone_variants(crop, h, stretch=stretch, rules=rules, handwriting=handwriting))
            res = self.ocr.recognize(variants)
            log.debug("zone %s -> %s", np.round(quad).astype(int).tolist(), res)
            return res

        ctx = Ctx(reread=reread, refine=self.settings.refine_enabled, read_zone=read_zone)

        def parse(lines: list[Line]) -> list[Draft]:
            drafts = []
            for seg in assemble(segment_page(lines)):
                if _is_noise(seg):
                    continue
                tmpl, _ = classify(seg)
                try:
                    d = tmpl.parse(seg, ctx)
                except Exception:  # a template bug must not kill the whole batch
                    log.exception("template %s failed", tmpl.doc_type)
                    from .templates.other import Generic
                    d = Generic().parse(seg, ctx)
                    d.review.append(f"Parser error in template {tmpl.doc_type}")
                d.rid = f"r{offset + len(drafts) + 1}"
                d.page_meta = {"page": page, "rotation": rot, "upright": up}
                drafts.append(d)
            return drafts

        drafts = parse(lines)
        if self.settings.second_pass and _read_poorly(drafts, self.settings.review_threshold):
            # Shadows, a dim corner or coloured light cost words the cleaned-up copy still shows. Both
            # readings are merged, and the page keeps whichever parse gives more trusted evidence, so
            # a second look can only add.
            merged = merge_lines(lines, self.ocr.read(flatten_page(up, _text_height(lines, up))))
            again = parse(merged)
            if _evidence(again) > _evidence(drafts) + 0.1:
                log.debug("second pass kept: %.2f -> %.2f", _evidence(drafts), _evidence(again))
                drafts, lines = again, merged
        h, w = img.shape[:2]
        po = PageOut(file=page.source, page=page.page_no, width=w, height=h, rotation=rot, ocr_lines=len(lines),
                     background_lines_ignored=foreground.ignored_lines if foreground else 0,
                     receipt_ids=[d.rid for d in drafts], elapsed_ms=int((_time.perf_counter() - t0) * 1000))
        return drafts, po

    # ------------------------------------------------------------------ output
    def _finalize(self, d: Draft, include_images: bool, include_raw: bool) -> ReceiptOut:
        page: Page = d.page_meta["page"]
        rot = d.page_meta["rotation"]
        h, w = page.image.shape[:2]
        reasons = list(dict.fromkeys(d.review))
        conf_parts = []
        for k in KEY_FIELDS.get(d.doc_type, []):
            f = d.fields.get(k)
            if f is None or f.value is None:
                reasons.append(f"Missing {k}")
                conf_parts.append(0.0)
            else:
                conf_parts.append(f.confidence)
                if f.source in ("derived", "corrected", "inferred", "vlm") and f.confidence < 0.95:
                    reasons.append(f"{k} is {f.source} ({f.value}), not read directly - confirm")
                elif f.source == "ocr_refined" and (f.confidence < self.settings.review_threshold or not any(
                        c.status == PASS and k in c.fields for c in d.checks)):
                    # a value read off a cleaned-up crop is trusted only when another figure agrees with it
                    reasons.append(f"{k} ({f.value}) was only legible after image clean-up "
                                   f"(marker, hatching or handwriting over it) - confirm")
        for c in d.checks:
            if c.status in (FAIL, WARN):
                reasons.append(f"{c.code}: {c.message}")
        confidence = round(min(conf_parts) if conf_parts else 0.0, 3)
        if confidence < self.settings.review_threshold and not any(
                r.startswith("Missing") or r.endswith("- confirm") for r in reasons):
            reasons.append(f"Low confidence ({confidence:.2f})")
        reasons = list(dict.fromkeys(reasons))
        status = "verified" if not reasons else "needs_review"

        fields = {}
        for k, f in d.fields.items():
            if f.value is None:
                continue
            val = f.value
            if k == "cardNumber" and self.settings.mask_card_numbers:
                val = mask_card(val)
            bbox = None
            if f.lines:
                b = (min(l.x0 for l in f.lines), min(l.y0 for l in f.lines),
                     max(l.x1 for l in f.lines), max(l.y1 for l in f.lines))
                bbox = bbox_to_original(b, rot, w, h)
            raw = f.raw
            if k == "cardNumber" and self.settings.mask_card_numbers and raw:
                raw = None
            fields[k] = FieldOut(value=val, confidence=round(min(1.0, f.confidence), 3), source=f.source, raw=raw,
                                 bbox=bbox)
        amount = d.val(d.amount_key) if d.amount_key else None
        af = d.fields.get(d.amount_key) if d.amount_key else None
        suggested = None
        if af is not None and af.confidence < 0.5:
            # never feed an unreadable amount into totals; the best guess travels separately so a
            # reviewer confirms a pre-filled number instead of retyping it
            suggested = int(af.value) if isinstance(af.value, (int, float)) else None
            amount = None
            status = "needs_review"
            reasons.append(f"Amount not reliably readable (best guess {af.value:,}) - enter manually")
        seg_box = d.segment.bbox
        img_b64 = None
        if include_images:
            img_b64 = to_jpeg_b64(crop_box(d.page_meta["upright"], seg_box, margin=12))
        return ReceiptOut(
            id=d.rid, source=SourceRef(file=page.source, page=page.page_no), doc_type=d.doc_type,
            category=d.category, status=status, confidence=confidence,
            amount=int(amount) if isinstance(amount, (int, float)) else None,
            claimed_amount=d.val("amountClaimed"), suggested_amount=suggested, transaction_date=d.val("transactionDate"),
            transaction_at=d.val("transactionAt"), merchant_name=d.val("merchantName"), description=describe(d),
            merchant_id=d.val("merchantId"), handwritten=d.handwritten, fingerprint=fingerprint(d),
            bbox=bbox_to_original(seg_box, rot, w, h), fields=fields,
            checks=[CheckOut(code=c.code, status=c.status, message=c.message, fields=c.fields) for c in d.checks],
            review_reasons=reasons, image=img_b64,
            raw_text=d.segment.texts if include_raw else None,
        )

    def _vlm_pass(self, drafts, receipts, include_images, include_raw, warnings):
        budget = self.settings.vlm_max_receipts
        out = list(receipts)
        for i, (d, r) in enumerate(zip(drafts, receipts)):
            if budget <= 0:
                break
            if self.settings.vlm_mode == "needs_review" and r.status == "verified":
                continue
            crop = crop_box(d.page_meta["upright"], d.segment.bbox, margin=16)
            try:
                data = self.vlm.extract(crop, d.doc_type)
            except Exception as exc:  # network, quota, bad JSON ...
                warnings.append(f"{d.rid}: vision fallback failed ({exc.__class__.__name__})")
                continue
            budget -= 1
            if data and self.vlm.merge(d, data):
                out[i] = self._finalize(d, include_images, include_raw)
        return out


def _read_poorly(drafts: list[Draft], threshold: float) -> bool:
    """No receipt at all, or one whose amount or date is missing or not trusted - for a reason a
    better-lit copy can fix. Handwriting stays as hard to read, and a value that was only legible
    after its own clean-up (marker, hatching) has had that treatment already."""
    if not drafts:
        return True
    for d in drafts:
        if d.handwritten:
            continue
        for k in KEY_FIELDS.get(d.doc_type, []):
            f = d.fields.get(k)
            if f is None or f.value is None or (f.confidence < threshold and f.source != "ocr_refined"):
                return True
    return False


def _evidence(drafts: list[Draft]) -> float:
    """How much a page's parse can be trusted: confidence of every key field found, less failed checks."""
    score = 0.0
    for d in drafts:
        found = [d.fields.get(k) for k in KEY_FIELDS.get(d.doc_type, [])]
        score += sum(f.confidence for f in found if f is not None and f.value is not None)
        score -= 0.5 * sum(c.status == FAIL for c in d.checks)
    return score


def _text_height(lines: list[Line], img: np.ndarray) -> float:
    hs = sorted(ln.h for ln in lines if ln.w > 1.5 * ln.h)
    return hs[len(hs) // 2] if hs else max(10.0, max(img.shape[:2]) / 60)


def _is_noise(seg: Segment) -> bool:
    if any(u.label for u in seg.units):
        return False
    text = " ".join(seg.texts)
    return len(seg.lines) < 3 and len(text) < 40


def _summary(receipts: list[ReceiptOut]) -> Summary:
    by: dict[str, CategoryTotal] = {}
    total = verified_amt = 0
    dates = []
    for r in receipts:
        a = r.amount or 0
        total += a
        if r.status == "verified":
            verified_amt += a
        ct = by.setdefault(r.category, CategoryTotal(count=0, amount=0))
        ct.count += 1
        ct.amount += a
        if r.transaction_date:
            dates.append(r.transaction_date)
    v = sum(1 for r in receipts if r.status == "verified")
    return Summary(receipt_count=len(receipts), verified_count=v, needs_review_count=len(receipts) - v,
                   total_amount=total, verified_amount=verified_amt, by_category=by,
                   date_from=min(dates) if dates else None, date_to=max(dates) if dates else None)
