"""Runtime settings. Every value can be overridden with a DOCSCAN_* env var."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields


def _cast(raw: str, typ):
    if typ is bool or typ == "bool":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if typ is int or typ == "int":
        return int(raw)
    if typ is float or typ == "float":
        return float(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    # --- OCR engine -------------------------------------------------------
    max_side_len: int = 2000          # long side fed to the detector (px)
    rec_batch_num: int = 16
    text_score: float = 0.5           # drop OCR lines below this score
    intra_op_threads: int = -1        # onnxruntime threads (-1 = all cores)
    orientation_samples: int = 10     # crops used to decide 90 vs 270 rotation

    # --- accuracy features ------------------------------------------------
    refine_enabled: bool = True       # constraint-guided re-OCR of failing fields
    second_pass: bool = True          # re-read a page that read poorly from a copy with the light evened out
    review_threshold: float = 0.85    # receipt confidence below this => needs_review
    date_outlier_days: int = 45       # warn when a receipt is far from the batch median date
    mask_card_numbers: bool = True    # 0145********6480 in the JSON output

    # --- input limits -----------------------------------------------------
    max_upload_mb: int = 20
    max_files: int = 20
    max_pixels: int = 40_000_000
    pdf_dpi: int = 200
    max_pdf_pages: int = 10

    # --- server -----------------------------------------------------------
    # Scans decoded at once per worker. The OCR engine reads one image at a time, so more only
    # holds more decoded pages in RAM and makes every scan finish late; the rest wait as uploads.
    max_concurrent_scans: int = 2

    # --- optional vision-LLM fallback (OpenAI-compatible endpoint) ---------
    vlm_mode: str = "off"             # off | needs_review | always
    vlm_base_url: str = ""            # e.g. http://127.0.0.1:20128/v1
    vlm_api_key: str = ""
    vlm_model: str = ""
    vlm_timeout_s: float = 60.0
    vlm_max_receipts: int = 10        # cost guard per request

    @classmethod
    def from_env(cls, prefix: str = "DOCSCAN_") -> "Settings":
        kwargs = {}
        for f in fields(cls):
            raw = os.getenv(prefix + f.name.upper())
            if raw not in (None, ""):
                kwargs[f.name] = _cast(raw, f.type)
        return cls(**kwargs)
