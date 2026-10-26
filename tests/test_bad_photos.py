"""Bad photos: dim or faded pictures, uneven light, and merging a second reading (no OCR models needed)."""
import numpy as np

from docscan.imaging import auto_levels, flatten_page
from docscan.ocr import Line, merge_lines


def line(text, x0, x1, cy, h=20, score=0.9):
    y0 = cy - h / 2
    return Line(0, text, score, np.array([[x0, y0], [x1, y0], [x1, y0 + h], [x0, y0 + h]], np.float32))


def page(paper=230, ink=40, shade=None):
    img = np.full((400, 600), paper, np.float32)
    for y in range(60, 360, 40):                                   # rows of 'text': 6 px strokes, 20 px tall
        for x in range(40, 560, 14):
            img[y:y + 20, x:x + 6] = ink
    if shade is not None:
        img *= shade
    return np.dstack([img.clip(0, 255).astype(np.uint8)] * 3)


def contrast(img):
    g = img[..., 0].astype(int)
    return np.percentile(g, 99.5) - np.percentile(g, 0.5)


def test_a_dark_photo_is_stretched_to_the_full_range():
    dark = page(paper=75, ink=13)                                  # an evening photo: ink and paper 60 levels apart
    out = auto_levels(dark)
    assert contrast(dark) < 70 and contrast(out) > 240
    assert out[..., 0].mean() > dark[..., 0].mean()


def test_a_faded_receipt_gets_darker_ink():
    faded = page(paper=250, ink=160)
    assert auto_levels(faded)[60:80, 40:46, 0].max() < 20


def test_a_good_picture_is_left_alone():
    good = page()
    assert auto_levels(good) is good


def test_a_shadow_across_the_page_is_evened_out():
    yy, xx = np.mgrid[0:400, 0:600]
    shaded = page(shade=0.3 + 0.7 * xx / 600)                     # dark on the left, bright on the right
    flat = flatten_page(shaded, 20)[..., 0].astype(int)
    paper_left, paper_right = flat[45, 20:40].mean(), flat[45, 560:590].mean()
    assert abs(paper_left - paper_right) < 15                      # paper is paper everywhere
    assert flat[60:80, 40:46].mean() < 80 and flat[60:80, 544:550].mean() < 80   # ink is ink everywhere


def test_merging_two_readings_of_a_page():
    first = [line("TOTAL", 20, 120, 100, score=0.95), line("176O0O", 300, 400, 100, score=0.62),
             line("TAX", 20, 80, 140, score=0.97)]
    second = [line("TOTAL", 21, 121, 101, score=0.93),            # same line, not surer: first reading stays
              line("176000", 302, 401, 100, score=0.96),          # same place, much surer: replaces the misread
              line("CASH 200000", 20, 400, 180, score=0.9),       # only the cleaned-up copy shows it: added
              line("TAX 16000", 20, 400, 141, score=0.99)]        # holds 'TAX' whole and says more: replaces it
    got = merge_lines(first, second)
    assert [ln.text for ln in got] == ["TOTAL", "176000", "CASH 200000", "TAX 16000"]
    assert got[0] is first[0] and got[1].x0 == first[1].x0       # replaced text keeps the first reading's place


def test_a_second_reading_that_disagrees_across_lines_is_dropped():
    first = [line("SUBTOTAL", 20, 150, 220), line("160000", 300, 400, 220)]
    second = [line("SUBTOTAL 16000O", 20, 400, 221, score=0.99)]   # spans both, and reads the amount differently
    assert [ln.text for ln in merge_lines(first, second)] == ["SUBTOTAL", "160000"]


def test_merging_when_one_reading_is_empty():
    only = [line("TOTAL 5.000", 20, 200, 100)]
    assert [ln.text for ln in merge_lines([], only)] == ["TOTAL 5.000"]
    assert merge_lines(only, []) == only
