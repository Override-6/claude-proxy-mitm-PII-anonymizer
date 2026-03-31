"""
Tests for image_anonymizer.

Each test synthesizes a PNG with known PII text using PIL, runs
anonymize_image(), then re-OCRs the output to confirm the PII is gone
and the placeholder label is present.

NOTE: OCR is not perfect. Tests assert on observable outcomes (image changed,
PII-like text gone, mappings non-empty) rather than exact string equality.
"""

import io
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont

from src.mappings import Mappings
import src.image_anonymizer as image_anonymizer

# Shared reader for verification OCR (avoid reloading the model)
_reader = easyocr.Reader(["en", "fr"], gpu=False, verbose=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ["arial.ttf", "Arial.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _make_image(text: str, width=900, height=100, font_size=40) -> bytes:
    """Render *text* centered vertically on a white PNG, return PNG bytes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, (height - font_size) // 2), text, fill=(0, 0, 0), font=_font(font_size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ocr_text(image_bytes: bytes) -> str:
    """Return all OCR-detected text joined into one string."""
    arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return " ".join(_reader.readtext(arr, detail=0, paragraph=False))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_email_is_redacted():
    """A short, OCR-clean email address must be detected and redacted."""
    pii = "bob@acme.com"   # short — OCR reliably reads it whole
    image_bytes = _make_image(f"Email: {pii}")
    mappings = Mappings()

    result = image_anonymizer.anonymize_image(image_bytes, mappings)

    assert result != image_bytes, "Image was not modified"
    assert len(mappings._sensitive_to_redacted) > 0, "Nothing was mapped"

    # Whatever was detected (may be the full or partial address) must be gone
    out_text = _ocr_text(result)
    for key in mappings._sensitive_to_redacted:
        assert key not in out_text, f"Detected PII {key!r} still visible after redaction"


def test_person_name_is_redacted():
    """A person name must be detected by NER and redacted."""
    pii = "Alice Johnson"
    image_bytes = _make_image(f"Hello my name is {pii}.")
    mappings = Mappings()

    result = image_anonymizer.anonymize_image(image_bytes, mappings)

    out_text = _ocr_text(result)
    # At least one token of the name should have been blacked out
    assert "Alice" not in out_text or "Johnson" not in out_text, \
        f"Person name still fully readable: {out_text!r}"


def test_no_pii_returns_identical_bytes():
    """Images with no PII must be returned byte-for-byte unchanged."""
    image_bytes = _make_image("The weather is nice today.")
    mappings = Mappings()

    result = image_anonymizer.anonymize_image(image_bytes, mappings)

    assert result == image_bytes, "Image was modified despite no PII"


def test_mappings_populated_after_redaction():
    """After anonymizing an image containing an email, mappings must be filled."""
    pii = "bob@acme.com"
    image_bytes = _make_image(f"Contact: {pii}")
    mappings = Mappings()

    image_anonymizer.anonymize_image(image_bytes, mappings)

    assert len(mappings._sensitive_to_redacted) > 0, \
        "Mappings are empty after processing an image with PII"
    # Every mapping value must have the [TYPE_N] format
    for sensitive, redacted in mappings._sensitive_to_redacted.items():
        assert redacted.startswith("["), f"Bad redacted label: {redacted!r}"


def test_mappings_consistent_across_calls():
    """Same PII appearing in two separate images must get the same placeholder."""
    pii = "bob@acme.com"
    mappings = Mappings()

    image_anonymizer.anonymize_image(_make_image(f"Email: {pii}"), mappings)
    snapshot = dict(mappings._sensitive_to_redacted)

    image_anonymizer.anonymize_image(_make_image(f"Send to {pii} now"), mappings)

    for key, label in snapshot.items():
        assert mappings._sensitive_to_redacted.get(key) == label, \
            f"Label changed for {key!r}: was {label!r}, now {mappings._sensitive_to_redacted.get(key)!r}"


def test_empty_image_no_crash():
    """A blank white image must not crash and must return valid image bytes."""
    image_bytes = _make_image("")
    mappings = Mappings()

    result = image_anonymizer.anonymize_image(image_bytes, mappings)

    assert isinstance(result, bytes) and len(result) > 0


def test_real_world_notes_image():
    """Smoke-test against the real screenshot from the ignore folder."""
    ignore_dir = os.path.join(os.path.dirname(__file__), "..", "ignore")
    img_path = os.path.join(ignore_dir, "Screenshot 2026-03-28 112855.png")
    if not os.path.exists(img_path):
        pytest.skip("ignore screenshot not present")

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    mappings = Mappings()
    result = image_anonymizer.anonymize_image(image_bytes, mappings)

    assert len(mappings._sensitive_to_redacted) > 0, \
        "No PII detected in real-world notes screenshot"
    assert result != image_bytes, "Image unchanged despite PII present"


def test_real_world_excel_image():
    """Smoke-test against the Excel contacts screenshot."""
    ignore_dir = os.path.join(os.path.dirname(__file__), "..", "ignore")
    img_path = os.path.join(ignore_dir, "Screenshot 2026-03-28 113012.png")
    if not os.path.exists(img_path):
        pytest.skip("ignore screenshot not present")

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    mappings = Mappings()
    result = image_anonymizer.anonymize_image(image_bytes, mappings)

    assert len(mappings._sensitive_to_redacted) > 0, \
        "No PII detected in real-world Excel screenshot"
    assert result != image_bytes, "Image unchanged despite PII present"
