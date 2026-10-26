"""The golden pages again, photographed badly: does a change help or hurt on poor pictures?

    python tests/eval_bad_photos.py            # 13 versions x 2 pages, 4 processes
    python tests/eval_bad_photos.py -j 2 -v    # fewer processes; -v lists every wrong or missing value

Each page is damaged the way phone photos are (dark, shadow, faded, blur, motion blur, low
resolution, heavy JPEG, noise, tilt, perspective, yellow light, all of it at once) with a fixed
seed, scanned, and scored with eval_golden. `silent_errors` must stay 0; compare `correct`
(and `amounts_right`) before and after a change - the damage is the same every run.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent / "golden"
PAGES = ("page1.jpg", "page2.jpg")
AMOUNTS = ("amountPaid", "totalPrinted", "fee", "fare", "amount", "total")


def damage(img: np.ndarray, kind: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    if kind == "clean":
        return img
    if kind == "dark":
        return (f * 0.3).astype(np.uint8)
    if kind == "shadow":
        ramp = np.linspace(0.35, 1.0, w, dtype=np.float32)[None, :, None]
        return (f * ramp).astype(np.uint8)
    if kind == "faded":
        return (255 - (255 - f) * 0.35).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), 1.6)
    if kind == "motion":
        k = np.zeros((9, 9), np.float32)
        k[4, :] = 1 / 9
        return cv2.filter2D(img, -1, k)
    if kind == "lowres":
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "noise":
        return np.clip(f + rng.normal(0, 18, f.shape), 0, 255).astype(np.uint8)
    if kind == "tilt":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 4, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    if kind == "perspective":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0.06 * w, 0.03 * h], [0.95 * w, 0], [w, h], [0.02 * w, 0.97 * h]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(255, 255, 255))
    if kind == "yellow":
        return np.clip(f * np.array([0.55, 0.9, 1.0], np.float32), 0, 255).astype(np.uint8)
    if kind == "mix":
        out = damage(img, "shadow")
        out = damage(out, "yellow")
        out = cv2.GaussianBlur(out, (0, 0), 0.9)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 40])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError(kind)


KINDS = ("clean", "dark", "shadow", "faded", "blur", "motion", "lowres", "jpeg", "noise", "tilt", "perspective",
         "yellow", "mix")
_scanner = None


def _init() -> None:
    global _scanner
    sys.path.insert(0, str(HERE.parents[1]))
    from docscan.pipeline import DocScanner

    _scanner = DocScanner()


def _run(kind: str) -> tuple[str, dict]:
    files = []
    for name in PAGES:
        img = damage(cv2.imread(str(HERE / name)), kind)
        files.append((name, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()))
    res = _scanner.scan(files, today=date(2026, 10, 12))
    return kind, json.loads(res.model_dump_json(by_alias=True, exclude_none=True))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_golden import evaluate

    args = sys.argv[1:]
    jobs = int(args[args.index("-j") + 1]) if "-j" in args else 4
    expected = [e for e in json.loads((HERE / "expected.json").read_text()) if e["file"] in PAGES]
    os.environ.setdefault("DOCSCAN_INTRA_OP_THREADS", str(max(1, (os.cpu_count() or 2) // jobs)))
    with Pool(jobs, initializer=_init) as pool:
        results = pool.map(_run, KINDS)
    total = dict(fields=0, correct=0, missing=0, flagged_errors=0, silent_errors=0, amounts=0, amounts_right=0)
    for kind, res in results:
        st = evaluate(res, expected)
        # every golden receipt has exactly one amount; a receipt not found loses its amount too
        missed = sum(1 for d in st["details"] if d.get("field") in AMOUNTS) + st["expected_receipts"] - st["matched_receipts"]
        total["amounts"] += len(expected)
        total["amounts_right"] += len(expected) - missed
        for k in ("fields", "correct", "missing", "flagged_errors", "silent_errors"):
            total[k] += st[k]
        print(f"{kind:12s} correct {st['correct']:3d}/{st['fields']}  missing {st['missing']:2d}  "
              f"flagged {st['flagged_errors']:2d}  silent {st['silent_errors']}", flush=True)
        if "-v" in args:
            for d in st["details"]:
                print("     ", d)
    print(json.dumps(total, indent=1))


if __name__ == "__main__":
    main()
