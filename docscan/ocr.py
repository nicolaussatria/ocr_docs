"""Thin, thread-safe wrapper around RapidOCR (PP-OCR ONNX models, CPU)."""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Settings
from .imaging import crop_quad, rotate

log = logging.getLogger(__name__)


@dataclass(slots=True, eq=False)
class Line:
    idx: int
    text: str
    score: float
    quad: np.ndarray                 # (4,2) tl,tr,br,bl in the upright frame
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    h: float = 0.0                   # glyph height
    slope: float = 0.0               # baseline dy/dx
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        q = self.quad
        self.x0, self.y0 = float(q[:, 0].min()), float(q[:, 1].min())
        self.x1, self.y1 = float(q[:, 0].max()), float(q[:, 1].max())
        self.h = float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2) or (self.y1 - self.y0)
        dx = float(q[1][0] - q[0][0])
        self.slope = float(q[1][1] - q[0][1]) / dx if dx > 1 else 0.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Line({self.idx}, {self.text!r}, {self.score:.2f}, x={self.x0:.0f}-{self.x1:.0f}, y={self.y0:.0f}-{self.y1:.0f})"


class OcrEngine:
    """One engine per process; calls are serialised (onnxruntime already uses all cores)."""

    def __init__(self, settings: Settings):
        from rapidocr import RapidOCR

        self.settings = settings
        self._lock = threading.Lock()
        self._eng = RapidOCR(params={
            "Global.log_level": "error",
            "Global.max_side_len": settings.max_side_len,
            "Global.text_score": settings.text_score,
            "Rec.rec_batch_num": settings.rec_batch_num,
            "EngineConfig.onnxruntime.intra_op_num_threads": settings.intra_op_threads,
        })
        import rapidocr

        self.version = f"rapidocr {getattr(rapidocr, '__version__', '?')} / PP-OCRv6-small"

    # ------------------------------------------------------------------ public
    def detect_orientation(self, img: np.ndarray) -> int:
        """Return clockwise rotation (0/90/180/270) that makes the page upright.

        1) detector boxes tell horizontal vs vertical text (cheap, small image)
        2) recognising a sample of crops under both candidate rotations decides
           the direction: the right one scores markedly higher.
        """
        h, w = img.shape[:2]
        s = min(1.0, 960 / max(h, w))
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        with self._lock:
            det = self._eng(small, use_det=True, use_cls=False, use_rec=False)
        boxes = getattr(det, "boxes", None)
        if boxes is None or len(boxes) < 3:
            return 0
        sides = []
        for q in boxes:
            q = np.asarray(q, dtype=np.float32)
            bw = np.linalg.norm(q[1] - q[0])
            bh = np.linalg.norm(q[3] - q[0])
            if max(bw, bh) >= 2.2 * max(min(bw, bh), 1):
                sides.append((bw, bh, q))
        if not sides:
            return 0
        vertical = sum(1 for bw, bh, _ in sides if bh > bw) > len(sides) / 2
        cands = (90, 270) if vertical else (0, 180)
        # longest boxes carry the most evidence
        sample = sorted(sides, key=lambda t: -max(t[0], t[1]))[: self.settings.orientation_samples]
        crops = [crop_quad(small, q) for _, _, q in sample]
        best, best_score = cands[0], -1.0
        for deg in cands:
            res = self.recognize([rotate(c, deg) for c in crops])
            if not res:
                continue
            mean_conf = float(np.mean([sc for _, sc in res]))
            score = float(np.mean([sc * min(len(t), 12) for t, sc in res]))
            if mean_conf >= 0.9 and deg == cands[0]:
                return deg          # upside-down text never scores this high: skip the 2nd pass
            if score > best_score:
                best, best_score = deg, score
        return best

    def read(self, img: np.ndarray) -> list[Line]:
        with self._lock:
            r = self._eng(img, use_det=True, use_cls=True, use_rec=True)
        if r is None or getattr(r, "txts", None) is None:
            return []
        out = []
        for i, (q, t, s) in enumerate(zip(r.boxes, r.txts, r.scores)):
            t = (t or "").strip()
            if not t:
                continue
            out.append(Line(i, t, float(s), np.asarray(q, dtype=np.float32)))
        return out

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognition only (no detection) for already-cropped single text lines."""
        crops = [c for c in crops if c is not None and c.size and min(c.shape[:2]) >= 4]
        if not crops:
            return []
        with self._lock:
            try:
                from rapidocr.ch_ppocr_rec import TextRecInput

                rec = self._eng._load_rec_model()          # pinned rapidocr==3.10.x
                res = rec(TextRecInput(img=crops))
                return [(t or "", float(s)) for t, s in zip(res.txts, res.scores)]
            except Exception:  # pragma: no cover - fallback for API drift
                log.debug("batch rec failed, falling back to per-crop calls", exc_info=True)
                out = []
                for c in crops:
                    r = self._eng(c, use_det=False, use_cls=False, use_rec=True)
                    txt = (r.txts[0] if getattr(r, "txts", None) else "") or ""
                    sc = float(r.scores[0]) if getattr(r, "scores", None) else 0.0
                    out.append((txt, sc))
                return out


def merge_lines(first: list[Line], second: list[Line]) -> list[Line]:
    """Combine two readings of one page, the second taken from a cleaned-up copy.

    * a line both found keeps the reading the recogniser is surer of, in the first reading's place;
    * a line only the second found is added;
    * a second-reading line that holds first-reading lines whole and says more ('TAX' -> 'TAX 16000')
      takes their place;
    * any other second-reading line that straddles first-reading lines is dropped, so no text is
      read twice."""
    if not first or not second:
        return list(first or second)
    a = np.array([[ln.x0, ln.y0, ln.x1, ln.y1] for ln in first])
    b = np.array([[ln.x0, ln.y0, ln.x1, ln.y1] for ln in second])
    w = np.clip(np.minimum(b[:, None, 2], a[None, :, 2]) - np.maximum(b[:, None, 0], a[None, :, 0]), 0, None)
    h = np.clip(np.minimum(b[:, None, 3], a[None, :, 3]) - np.maximum(b[:, None, 1], a[None, :, 1]), 0, None)
    inter = w * h                                                     # second x first
    area = lambda m: np.maximum((m[:, 2] - m[:, 0]) * (m[:, 3] - m[:, 1]), 1e-6)   # noqa: E731
    area_a, area_b = area(a), area(b)
    iou = inter / (area_b[:, None] + area_a[None, :] - inter)
    squash = lambda t: "".join(t.split()).upper()                     # noqa: E731
    keep: list[Line | None] = list(first)
    extra = []
    for i, ln in enumerate(second):
        touched = np.flatnonzero(inter[i] > 0)
        if not len(touched):
            extra.append(ln)
            continue
        j = int(iou[i].argmax())
        if iou[i, j] >= 0.75 or (iou[i, j] >= 0.6 and len(touched) == 1):
            old = first[j]
            if keep[j] is old and ln.text != old.text and ln.score > old.score + 0.05:
                keep[j] = Line(old.idx, ln.text, ln.score, old.quad.copy())
        elif all(inter[i, k] >= 0.8 * area_a[k] and keep[k] is first[k] and squash(first[k].text) in squash(ln.text)
                 for k in touched):
            for k in touched:
                keep[k] = None
            extra.append(ln)
        elif inter[i].sum() < 0.25 * area_b[i]:
            extra.append(ln)
    return [ln for ln in keep if ln is not None] + extra


def local_slope(lines: list[Line], k: int = 12) -> np.ndarray:
    """Per-line baseline slope smoothed over its k nearest *long* neighbours.

    Short boxes ('6', '11') have unreliable angles; borrowing the median slope of
    nearby long lines makes same-row matching robust on skewed photos.
    """
    if not lines:
        return np.zeros(0)
    c = np.array([[ln.cx, ln.cy] for ln in lines])
    long_mask = np.array([ln.w >= 3 * max(ln.h, 1) for ln in lines])
    slopes = np.array([ln.slope for ln in lines])
    if long_mask.sum() < 3:
        return np.full(len(lines), float(np.median(slopes)) if len(slopes) else 0.0)
    lc, ls = c[long_mask], slopes[long_mask]
    out = np.empty(len(lines))
    for i in range(len(lines)):
        d = np.hypot(*(lc - c[i]).T)
        idx = np.argsort(d)[:k]
        out[i] = float(np.median(ls[idx]))
    # clamp absurd values (>10 degrees) – those are detector artefacts
    lim = math.tan(math.radians(10))
    return np.clip(out, -lim, lim)
