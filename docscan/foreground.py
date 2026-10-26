"""Find separate receipt sheets before layout can attach backing-paper text.

Use visible paper boundaries or complete, separate fuel-receipt columns. A plain
invoice or a page without convincing receipt evidence keeps the normal path.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import numpy as np

from .normalize import compact, find_dates
from .ocr import Line


def _fuel_columns(lines: list[Line]) -> list[np.ndarray]:
    """Recover side-by-side fuel sheets whose white edges disappear in a scan.

    Require a separate station heading, date, volume and payment in every column.
    Bare prices, invoice fragments and subsidy notices cannot seed a column.
    """
    from .layout import build_units

    anchors = [l for l in lines if compact(l.text).startswith("SPBU")]
    if len(anchors) < 2:
        return []
    anchors.sort(key=lambda l: l.cx)
    h = float(np.median([l.h for l in anchors]))
    if max(l.cy for l in anchors) - min(l.cy for l in anchors) > 8 * h:
        return []  # this fallback is only for receipts arranged beside each other
    units = build_units(lines)
    quads = []
    for anchor in anchors:
        body = [u for u in units if u.key in {"datetime", "pump", "volume", "price", "unitPrice", "paid", "totalPrice"}
                and u.y0 > anchor.y0
                and min(anchors, key=lambda a: abs(a.cx - (u.x0 + u.x1) / 2)) is anchor]
        keys = {u.key for u in body}
        if not {"datetime", "volume", "pump"} <= keys or not keys & {"paid", "totalPrice"}:
            return []
        x0 = min(anchor.x0, *(u.x0 for u in body))
        x1 = max(anchor.x1, *(u.x1 for u in body))
        y0 = min(anchor.y0 - 4 * h, *(u.y0 for u in body))
        bottom = max(u.y1 for u in body)
        # Wider footer text may extend beyond the right-aligned amount column.
        # Use the original body span for every comparison: never grow into a
        # neighbouring invoice through a chain of overlapping fragments.
        contained = [l for l in lines if y0 <= l.cy <= bottom + 16 * h
                     and min(x1, l.x1) - max(x0, l.x0) >= .7 * l.w]
        if not contained:
            return []
        x0 = min(x0, *(l.x0 for l in contained)) - .35 * h
        x1 = max(x1, *(l.x1 for l in contained)) + 1.2 * h
        y1 = max(bottom, *(l.y1 for l in contained)) + .35 * h
        quads.append(np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]], np.float32))
    if any(cv2.intersectConvexConvex(q, r)[0] > 0 for i, q in enumerate(quads) for r in quads[i+1:]):
        return []
    return quads


@dataclass
class Foreground:
    image: np.ndarray
    regions: list[np.ndarray]
    ignored_lines: int


def _inside(quad: np.ndarray, line: Line) -> bool:
    return cv2.pointPolygonTest(quad, (line.cx, line.cy), False) >= 0


def _receipt_identity(lines: list[Line]) -> bool:
    text = compact(" ".join(l.text for l in lines))
    return any(s in text for s in ("PERTAMINA", "SPBU", "JASAMARGA", "BONKONTAN", "KWITANSI", "KUITANSI", "MCOMMERCE"))


def _transaction(lines: list[Line]) -> bool:
    """Require a receipt identity and body; an exposed total is not a receipt."""
    if len(lines) < 6:
        return False
    text = "\n".join(l.text for l in lines)
    c = compact(text)
    identity = _receipt_identity(lines)
    fields = sum(bool(re.search(p, c)) for p in
                 (r"VOLUME|LITER|NOMINAL", r"TOTAL|JUMLAH|DIBAYAR|TARIF|BIAYA", r"WAKTU|TANGGAL|TGL|SHIFT|TRANS"))
    # Also retain an independently bordered ordinary receipt/invoice in a mixed page.
    dated = any(find_dates(l.text) for l in lines)
    labelled_total = bool(re.search(r"(?i)\b(total|jumlah|dibayar|tarif|biaya|grand\s*total|amount\s*due)\b", text))
    return labelled_total and ((identity and fields >= 2) or dated)


def receipt_foreground(image: np.ndarray, lines: list[Line]) -> Foreground | None:
    """Mask background when separate transaction sheets or fuel columns are found.

    Coordinates stay in the upright page frame. A containing rectangle is backing
    paper, not another receipt. OCR runs again on the mask so a detector box that
    originally crossed a paper edge cannot smuggle background digits into a field.
    """
    if len(lines) < 12:
        return None
    h, w = image.shape[:2]
    scale = min(1.0, 1600 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(smooth, 15, 55)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # A receipt may reach the scanner frame. Close its open edge at that frame,
    # allowing the visible sides to define a sheet without requiring a white margin.
    cv2.rectangle(edges, (0, 0), (edges.shape[1]-1, edges.shape[0]-1), 255, 1)
    masks = [edges]
    # On a dark photocopied backing sheet, the paper surfaces separate even when
    # their edges touch a page boundary. Closing fills letters, not gaps between sheets.
    for threshold in (220, 240, 250):
        bright = cv2.threshold(smooth, threshold, 255, cv2.THRESH_BINARY)[1]
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        masks.append(bright)
    candidates = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour) / scale**2
            if not 0.025 * h * w <= area <= 0.85 * h * w:
                continue
            quad = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            quad = quad.reshape(4, 2).astype(np.float32) / scale
            sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
            if min(sides) < 0.08 * min(h, w):
                continue
            contained = [l for l in lines if _inside(quad, l)]
            if _transaction(contained):
                candidates.append((area, quad, contained))
    # Prefer the outer paper edge among nearly identical detections, and discard
    # a large enclosing backing sheet when smaller receipt sheets are present.
    regions = []
    for area, quad, contained in sorted(candidates, key=lambda item: item[0]):
        overlap = [r for r in regions if sum(_inside(quad, l) for l in r[2]) >= 0.8 * len(r[2])]
        if overlap:
            if len(overlap) == 1 and area <= overlap[0][0] * 1.15:
                regions = [r for r in regions if r is not overlap[0]]
            else:
                continue
        regions.append((area, quad, contained))
    # Two bordered sections of one ordinary invoice must not trigger masking.
    if sum(_receipt_identity(contained) for _, _, contained in regions) < 2:
        quads = _fuel_columns(lines)
        if not quads:
            return None
        regions = [(cv2.contourArea(q), q, [l for l in lines if _inside(q, l)]) for q in quads]
    # Ambiguous intersecting sheets are left to the existing layout parser.
    for i, (_, q, _) in enumerate(regions):
        for area, other, _ in regions[i + 1:]:
            intersection, _ = cv2.intersectConvexConvex(q, other)
            if intersection > 0.05 * min(cv2.contourArea(q), area):
                return None
    quads = [q for _, q, _ in regions]
    outside = [l for l in lines if not any(_inside(q, l) for q in quads)]
    # A third receipt whose paper edge was missed is not backing paper. Likewise,
    # preserve an independently readable invoice separated from all selected sheets.
    if _receipt_identity(outside):
        return None
    if _transaction(outside):
        from .layout import segment_page
        # Scattered backing fragments can make the outside union cover the whole
        # page. Check individual groups too, so they cannot hide a real invoice.
        for group in [outside, *(s.lines for s in segment_page(outside))]:
            if not _transaction(group):
                continue
            x0, y0 = min(l.x0 for l in group), min(l.y0 for l in group)
            x1, y1 = max(l.x1 for l in group), max(l.y1 for l in group)
            box = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]], np.float32)
            if all(cv2.intersectConvexConvex(box, q)[0] < 0.05 * cv2.contourArea(q) for q in quads):
                return None
    ignored = len(outside)
    if not ignored:
        return None
    mask = np.zeros((h, w), np.uint8)
    for q in quads:
        cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 255)
    # Remove the whole rejected line, including strokes crossing a mask edge.
    # Otherwise a sliver of an invoice word can become a new OCR token on reread.
    for line in outside:
        q = line.quad
        centre = q.mean(axis=0)
        q = centre + (q - centre) * 1.08
        cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 0)
    cleaned = image.copy()
    cleaned[mask == 0] = 255
    return Foreground(cleaned, quads, ignored)
