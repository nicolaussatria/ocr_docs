"""Image loading, rotation bookkeeping and crop/enhancement helpers."""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 120_000_000  # hard ceiling; Settings.max_pixels is the real gate

PDF_MAGIC = b"%PDF"


class InputError(ValueError):
    """Raised for unreadable / oversized / unsupported uploads (maps to HTTP 422)."""


@dataclass
class Page:
    image: np.ndarray        # BGR, original orientation (EXIF already applied)
    source: str              # filename
    page_no: int             # 1-based page inside the source file


def load_pages(data: bytes, filename: str, *, max_pixels: int, pdf_dpi: int, max_pdf_pages: int) -> list[Page]:
    if not data:
        raise InputError(f"{filename}: empty file")
    if data[:4] == PDF_MAGIC:
        return _load_pdf(data, filename, pdf_dpi, max_pdf_pages, max_pixels)
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)  # phone photos carry rotation in EXIF
        if im.width * im.height > max_pixels:
            scale = (max_pixels / (im.width * im.height)) ** 0.5
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)
        im = im.convert("RGB")
    except InputError:
        raise
    except Exception as exc:  # PIL raises many types
        raise InputError(f"{filename}: not a readable image ({exc.__class__.__name__})") from exc
    return [Page(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), filename, 1)]


def _load_pdf(data: bytes, filename: str, dpi: int, max_pages: int, max_pixels: int) -> list[Page]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover
        raise InputError("PDF support needs pypdfium2") from exc
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        raise InputError(f"{filename}: unreadable PDF") from exc
    pages = []
    try:
        if len(pdf) > max_pages:
            raise InputError(f"{filename}: {len(pdf)} pages exceeds limit of {max_pages}")
        for i in range(len(pdf)):
            page = pdf[i]
            w, h = page.get_size()
            scale = dpi / 72
            if w * h * scale * scale > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
            # pypdfium2 takes a float scale (documented); its unannotated `scale=1` default makes Pylance infer int
            pil = page.render(scale=scale).to_pil().convert("RGB")  # pyright: ignore[reportArgumentType]
            pages.append(Page(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), filename, i + 1))
    finally:
        pdf.close()
    return pages


# --------------------------------------------------------------------------- rotation
_CV_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(img: np.ndarray, deg_cw: int) -> np.ndarray:
    return img if deg_cw % 360 == 0 else cv2.rotate(img, _CV_ROT[deg_cw % 360])


def unrotate_points(pts: np.ndarray, deg_cw: int, orig_w: int, orig_h: int) -> np.ndarray:
    """Map points from the rotated (upright) frame back to the original image frame."""
    pts = np.asarray(pts, dtype=np.float32)
    x, y = pts[..., 0], pts[..., 1]
    d = deg_cw % 360
    if d == 0:
        return pts.copy()
    if d == 90:      # upright(x', y') = (H-1-y, x)
        return np.stack([y, orig_h - 1 - x], axis=-1)
    if d == 180:
        return np.stack([orig_w - 1 - x, orig_h - 1 - y], axis=-1)
    # 270: upright(x', y') = (y, W-1-x)
    return np.stack([orig_w - 1 - y, x], axis=-1)


def bbox_to_original(b: tuple[float, float, float, float], deg_cw: int, orig_w: int, orig_h: int) -> list[int]:
    x0, y0, x1, y1 = b
    pts = unrotate_points(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), deg_cw, orig_w, orig_h)
    return [int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())]


# --------------------------------------------------------------------------- page clean-up
def auto_levels(img: np.ndarray, min_range: float = 100) -> np.ndarray:
    """Stretch a dim or washed-out picture (evening light, faded thermal paper) to the full range.

    The text detector needs contrast: on a dark photo whose ink and paper are ~60 grey levels apart
    it finds no text at all and the receipt disappears. Ordinary photos span ~120-250 levels and
    come back unchanged - stretching a merely soft one (blur lowers the span too) costs more
    words than it wins."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lo, hi = np.percentile(gray[::4, ::4], (0.5, 99.5))
    if hi - lo >= min_range:
        return img
    lut = np.clip((np.arange(256, dtype=np.float32) - lo) * 255 / max(hi - lo, 16.0), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def flatten_page(img: np.ndarray, h: float) -> np.ndarray:
    """The whole page with uneven light divided out - a hand or phone shadow, a dim corner, a yellow
    lamp - for a second reading of a badly lit photo. `h` is the typical text height in pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.cvtColor(flatten_shading(gray, 1.35 * h), cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- crops
def crop_quad(img: np.ndarray, quad: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Perspective-crop a text quad (tl,tr,br,bl) without any auto-rotation."""
    q = np.asarray(quad, dtype=np.float32).copy()
    if pad:
        c = q.mean(axis=0)
        q = c + (q - c) * (1 + pad)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    w, h = max(w, 2), max(h, 2)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)


def crop_box(img: np.ndarray, box: tuple[float, float, float, float], margin: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return img[max(0, y0 - margin):min(h, y1 + margin), max(0, x0 - margin):min(w, x1 + margin)]


def enhance_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Alternative renderings of one text line for re-recognition.

    Thermal prints fade and reimbursement receipts are often highlighted with a
    grey/yellow marker over the amount; binarisation removes the marker tone,
    CLAHE restores faded strokes, upscaling helps thin dot-matrix digits.
    """
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = 2.0 if gray.shape[0] < 64 else 1.0
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(big)
    _, otsu = cv2.threshold(cv2.GaussianBlur(big, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # marker removal: anything lighter than the darkest 35% of pixels becomes paper
    cut = np.percentile(big, 35)
    marker = np.where(big > cut, 255, big).astype(np.uint8)
    out = [big, clahe, otsu, marker]
    return [cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in out]


# --------------------------------------------------------------------------- amount zones
def flatten_shading(gray: np.ndarray, h: float) -> np.ndarray:
    """Divide by the local background so a marker stroke over the text becomes paper.

    `h` is the text height in pixels: strokes are thinner than ~h/2, a highlighter
    band is not, so a closing of that size keeps the band and drops the glyphs."""
    k = max(5, int(0.45 * h)) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    flat = cv2.divide(gray, bg, scale=255)
    lo, hi = np.percentile(flat, 2), np.percentile(flat, 60)
    return np.clip((flat.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)


def remove_rules(gray: np.ndarray, h: float) -> np.ndarray:
    """Erase long thin horizontal rules (hatched amount boxes, form lines).

    Pen strokes that cross a rule are taller than the rule and survive."""
    inv = 255 - gray
    hor = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(0.7 * h)), 1)))
    tall = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(h / 8)))))
    return 255 - cv2.subtract(inv, cv2.subtract(hor, tall))


def ink_extent(crop: np.ndarray, h: float, gap: float = 1.0) -> tuple[int, int] | None:
    """Horizontal extent (x0, x1) of the first block of writing in a row crop.

    Used to cut a handwritten value out of the band right of its label: the block
    ends at the first ink-free stretch wider than `gap` text heights, so a signature
    or the next column further right is left out."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.size == 0:
        return None
    ink = remove_rules(flatten_shading(gray, h), h) < 110
    cols = ink.sum(axis=0) >= max(2, int(0.1 * h))
    runs, start = [], None
    for x, on in enumerate(np.append(cols, False)):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if runs and start - runs[-1][1] < gap * h:
                runs[-1][1] = x
            else:
                runs.append([start, x])
            start = None
    runs = [r for r in runs if r[1] - r[0] >= 0.8 * h]
    if not runs:
        return None
    pad = int(0.3 * h)
    return max(0, runs[0][0] - pad), min(gray.shape[1], runs[0][1] + pad)


def handwritten_date_variants(crop: np.ndarray, *, stretch: tuple[float, ...] = (0.7, 1.0)) -> list[np.ndarray]:
    """Date lines at recognition height and double height, independent of scan DPI.

    Normalize before smoothing/thresholding so a 3-pixel filter has the same
    meaning on phone photos and large scans. Avoid local shading division: it
    amplifies faint signature strokes crossing a date into additional digits.
    Keep the grayscale reading alongside two foreground thresholds.
    """
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    out = []
    # Keep already small crops at native resolution: resampling those again can
    # remove the separator. Large scans get bounded, comparable recognition sizes.
    heights = (gray.shape[0],) if gray.shape[0] <= 96 else (48, 96)
    for height in heights:
        scale = height / gray.shape[0]
        normalized = cv2.resize(gray, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        smooth = cv2.GaussianBlur(normalized, (3, 3), 0)
        variants = [normalized]
        variants += [cv2.threshold(smooth, int(t), 255, cv2.THRESH_BINARY)[1]
                     for t in np.unique(np.percentile(smooth, (10, 30)).astype(int))]
        for fx in stretch:
            up = 2 if height < 96 else 1
            out += [cv2.cvtColor(cv2.resize(v, None, fx=fx * up, fy=up, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
                    for v in variants]
    return out


def zone_variants(crop: np.ndarray, h: float, *, stretch: tuple[float, ...] = (1.0,),
                  rules: bool = False, handwriting: bool = False) -> list[np.ndarray]:
    """Renderings of an amount zone for re-recognition.

    `stretch` widens the crop: double-height thermal fonts (the fare line of a toll
    slip) are so narrow that the recogniser merges repeated digits ('11000' -> '100').
    `rules` adds variants with horizontal hatching removed (kwitansi amount boxes).
    `handwriting` suppresses faint form backgrounds at two local ink thresholds;
    the original is retained because thresholding can erase a decimal point."""
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    pen = []
    if handwriting:
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        for threshold in np.unique(np.percentile(smooth, (10, 30)).astype(int)):
            pen.append(cv2.threshold(smooth, int(threshold), 255, cv2.THRESH_BINARY)[1])
    up = 2.0 if gray.shape[0] < 96 else 1.0
    out = []
    for fx in stretch:
        big = cv2.resize(gray, None, fx=up * fx, fy=up, interpolation=cv2.INTER_CUBIC)
        hh = h * up
        out += [big, flatten_shading(big, hh)]
        out += [cv2.resize(v, None, fx=up * fx, fy=up, interpolation=cv2.INTER_CUBIC) for v in pen]
        if rules:
            clean = remove_rules(big, hh)
            out += [clean, flatten_shading(clean, hh)]
    return [cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in out]


def to_jpeg_b64(img: np.ndarray, max_side: int = 900, quality: int = 80) -> str:
    import base64
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""
