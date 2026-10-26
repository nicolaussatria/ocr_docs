"""Handwritten optical fields, stacked dates, and the actual uploaded photograph."""
from pathlib import Path

import numpy as np
import pytest

from docscan.layout import Segment, Unit
from docscan.ocr import Line
from docscan.templates.base import Ctx, Draft
from docscan.templates.medical import OpticalForm, _stacked_dates


def line(text, x0, y0, x1, y1):
    return Line(0, text, .96, np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32))


@pytest.mark.parametrize("parts,expected", [
    ([['02', '02'], ['10', '/10'], ['20.26', '2026']], '2026-10-02'),
    ([['17', '17'], ['03', '3'], ['20.25', '2025']], '2025-03-17'),
    ([['31', '31'], ['02', '02'], ['2026', '2026']], None),  # invalid calendar date
    ([['02', '13'], ['10', '10'], ['2026', '2026']], None),  # conflicting day
    ([['02', '02'], ['10', '10'], ['26', '26']], None),      # never supply missing century
    ([['OZ', 'OZ'], ['to', 'to'], ['2026', '2026']], None),  # letters alone are not evidence
])
def test_stacked_date_requires_all_three_numeric_components(parts, expected):
    readings = iter(parts)
    ctx = Ctx(lambda lines: [], read_zone=lambda *args, **kw: [(t, .9) for t in next(readings)])
    lines = [line('Place 02026', 100, 20, 370, 100), line('to', 240, 90, 275, 125)]
    result = _stacked_dates(lines, 200, ctx)
    assert [v for v, _ in result] == ([expected] if expected else [])


def test_prescription_columns_do_not_copy_other_eye_or_turn_blanks_into_zero():
    lines = [line(t, x, 200, x+24, 222) for t, x in
             [('R', 20), ('CYL', 90), ('AX', 160), ('Add', 230), ('L', 300), ('CYL', 370), ('AX', 440)]]
    lines += [line('-1.50', 15, 225, 55, 245), line('-', 90, 225, 114, 245),
              line('-275', 295, 225, 335, 245)]
    d = Draft('optical', 'medical', Segment([]))
    OpticalForm._prescription(d, lines)
    assert d.val('rightSphere') == -1.5
    assert d.val('leftSphere') == -2.75 and d.fields['leftSphere'].source == 'inferred'
    assert d.fields['leftSphere'].raw == '-275' and d.review
    assert 'rightCylinder' not in d.fields and 'leftCylinder' not in d.fields and 'addition' not in d.fields


def test_inline_optical_fields_do_not_require_separate_caption_boxes():
    lines = [line('OPTIK TEST', 10, 10, 150, 30), line('Nama: Sari', 10, 80, 160, 100),
             line('Alamat: Bandung', 10, 110, 190, 130), line('Lensa: Bluecut', 10, 140, 190, 160)]
    d = OpticalForm().parse(Segment([Unit([l]) for l in lines]), Ctx(lambda lines: [], refine=False))
    assert d.val('patientName') == 'Sari' and d.val('patientAddress') == 'Bandung'


@pytest.mark.slow
def test_original_optical_photo_extracts_claim_and_prescription():
    from docscan.pipeline import DocScanner
    p = Path(__file__).parent / 'golden' / 'optical-handwritten.jpg'
    result = DocScanner().scan([(p.name, p.read_bytes())])
    assert len(result.receipts) == 1
    receipt = result.receipts[0]
    assert receipt.doc_type == 'optical' and receipt.category == 'medical'
    assert receipt.transaction_date == '2026-10-02' and receipt.amount == 350000
    assert receipt.status == 'needs_review'
    fields = receipt.fields
    expected = {'merchantName': 'OPTIK RIZKI EYE+', 'merchantAddress': 'Jakarta',
                'receiptNo': '256651', 'patientName': 'Rita', 'patientAddress': 'Palembang',
                'patientPhone': '082151096882', 'merchantPhone': '085138546518',
                'lensType': 'Photochromic', 'rightSphere': -2.25, 'leftSphere': -2.25}
    assert {k: fields[k].value for k in expected} == expected
    assert fields['transactionDate'].source == 'ocr_refined'
    assert fields['transactionDate'].raw == '02/10/2026'
    assert fields['leftSphere'].source == 'inferred'
    assert any('Frame code' in reason for reason in receipt.review_reasons)
    assert 'frameCode' not in fields  # this OCR model cannot settle the handwritten prefix
