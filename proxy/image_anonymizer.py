"""
Image anonymizer: OCR → NER/regex → black-box redaction.

Pipeline for each image:
  1. EasyOCR extracts per-region text with bounding boxes.
  2. Regions are grouped into lines (similar Y position) then joined
     smartly: no space is inserted only when 0 ≤ gap < 0.45× avg-char-width
     (catches OCR token splits).  Negative gaps always get a space.
  3. Entity detection runs on each merged line string using BOTH the standard
     finders (text_anonymizer) AND OCR-lax regexes that tolerate the kinds of
     partial reads EasyOCR produces (e.g. missing TLD dot → \\w+@\\w+ still hits).
  4. Every PII span is mapped back to source region(s), merged into one bbox,
     filled black.  The redacted label is drawn at the largest font size that
     fits entirely inside the black box (no overflow, no overlap).

Returns PNG bytes; original bytes returned unchanged when no PII is found.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import easyocr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from proxy.mappings import Mappings
from proxy.engine import DLPProxy, _add_non_overlapping  # DLPProxy imported for type reference in anonymize_image
from proxy.entity_finder import Entity
from proxy.entity_finder.regex_finder import RegexEntityFinder

log = logging.getLogger(__name__)

# Discard OCR results below this confidence (0.0–1.0)
_MIN_CONF = 0.3

# No space is inserted between two adjacent same-line regions when:
#   0 <= gap_px < avg_char_width * _NO_SPACE_GAP_FACTOR
# 0.45 catches real OCR token-splits (gap ≈7 px) while still putting a
# space between proper words (gap ≈8-12 px at typical screen font sizes).
# Negative gaps (overlapping bboxes from different screen layers) always
# get a space regardless of this factor.
_NO_SPACE_GAP_FACTOR = 0.45

# Two regions belong to the same visual line when their vertical gap is
# smaller than this fraction of the current line's tallest region height.
_SAME_LINE_GAP_FACTOR = 0.5

_FONT_CANDIDATES = [
    "arial.ttf",
    "Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_MIN_FONT_SIZE = 6


def _is_limited_arm() -> bool:
    """Return True on ARMv8.0 CPUs (e.g. Raspberry Pi 4 Cortex-A72) that lack
    the dot-product extension (asimddp). These CPUs crash with SIGILL when PyTorch
    runs multi-threaded NEON kernels (torch.bmm, DeBERTa attention, etc.).
    Returns False on x86, CUDA-capable machines, or modern ARM with asimddp."""
    if platform.machine() not in ("aarch64", "arm64"):
        return False
    try:
        with open("/proc/cpuinfo") as f:
            return "asimddp" not in f.read()
    except OSError:
        return False


_n_cpu = os.cpu_count() or 1
if _is_limited_arm():
    # ARMv8.0 baseline: force single-threaded inference to prevent SIGILL
    # from optimized multi-threaded NEON kernels unsupported on this CPU.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
else:
    # Modern CPU (x86, CUDA, ARMv8.2+): use all cores for best performance.
    torch.set_num_threads(_n_cpu)
    try:
        torch.set_num_interop_threads(_n_cpu)
    except RuntimeError:
        pass

# ---------------------------------------------------------------------------
# OCR-lax regex patterns
# These are intentionally looser than the production regexes in text_anonymizer
# because OCR output is noisy: dots can be dropped, TLDs split off, etc.
# They run ONLY on OCR text, never on real user text.
# ---------------------------------------------------------------------------

# Email: needs something@something with ≥2 chars after @.
# No TLD dot required — catches "finance@company" and "finance@companycom".
# Requires ≥2 chars on the right so single-char OCR noise like "x@y" is ignored.
_OCR_EMAIL_RE = re.compile(r'[\w._%+\-]+@[\w.\-]{2,}')

# Phone (OCR-lax): mirrors text_anonymizer PHONE_REGEX structure.
# Differences from the strict version:
#  - No backreference: OCR may produce inconsistent separators within one number
#  - () allowed inside groups: OCR sometimes fragments area codes with stray parens
_OCR_PHONE_RE = re.compile(
    r'(?<!\d)'
    r'(?:'
    # International +CC: compact (+33761647274) or with separators
    r'\+\d{1,3}(?:\d{6,12}|[\s\-.]?\(?\d{1,4}\)?(?:[\s\-().]?\d{2,4}){1,3})'
    r'|'
    # Local 0-prefix: compact (0761647274) or with separators (06 12 34 56 78)
    r'0\d{1,4}(?:\d{6,10}|(?:[\s\-().]?\d{2,4}){2,4})'
    r'|'
    # Parenthesized area code: (NXX) NXX-XXXX
    r'\(\d{2,4}\)[\s\-.]?\d{3,4}[\s\-.]?\d{4}'
    r'|'
    # Separator groups without prefix — mixed separators OK (no backreference)
    r'\d{2,4}[\s\-]\d{3,4}[\s\-]\d{3,4}'
    r')'
    r'(?!\d)'
)

_ocr_lax_finder = RegexEntityFinder([
    (_OCR_EMAIL_RE, "EMAIL"),
    (_OCR_PHONE_RE, "PHONE"),
])

# ---------------------------------------------------------------------------
# Image result cache
# ---------------------------------------------------------------------------

_CACHE_VERSION = 1
_CACHE_DIR = Path(__file__).parent.parent / "cache" / "images"


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _load_cache(h: str) -> Optional[dict]:
    path = _CACHE_DIR / f"{h}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == _CACHE_VERSION:
            return data
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return None


def _save_cache(h: str, merged: list, entities_per_line: list) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for (line_text, line_regions), entities in zip(merged, entities_per_line):
        lines.append({
            "text": line_text,
            "regions": [
                {"text": r.text, "left": r.left, "top": r.top,
                 "right": r.right, "bottom": r.bottom,
                 "start": r.start, "end": r.end}
                for r in line_regions
            ],
            "entities": [
                {"text": e.text, "type": e.type, "start": e.start, "end": e.end}
                for e in entities
            ],
        })
    path = _CACHE_DIR / f"{h}.json"
    path.write_text(json.dumps({"version": _CACHE_VERSION, "lines": lines}),
                    encoding="utf-8")


def _restore_cache(data: dict):
    """Return (merged, entities_per_line) from a cache dict."""
    from entity_finder import Entity
    merged = []
    entities_per_line = []
    for line in data["lines"]:
        regions = [
            _Region(text=r["text"], left=r["left"], top=r["top"],
                    right=r["right"], bottom=r["bottom"],
                    start=r["start"], end=r["end"])
            for r in line["regions"]
        ]
        merged.append((line["text"], regions))
        entities_per_line.append([
            Entity(e["text"], e["type"], e["start"], e["end"])
            for e in line["entities"]
        ])
    return merged, entities_per_line


def _detect_entities_ocr_batch(proxy: DLPProxy, texts: List[str], mappings: Mappings) -> List[List[Entity]]:
    """Batch version: runs standard+lax finders across all texts in one pass.

    NER is batched via nlp.pipe() — a single model forward pass instead of
    one per OCR line, which is the main OCR performance bottleneck.
    """

    accepted: dict[str, list[Entity]] = {}

    for finder in proxy.finders:
        for text, entities in zip(texts, finder.find_entities_batch(texts, proxy.mappings)):
            if not text in accepted:
                accepted[text] = []
            _add_non_overlapping(accepted[text], entities)

    return [
        _merge_ocr_entities(standard, _ocr_lax_finder.find_entities_batch([text], mappings))
        for text, standard in accepted.items()
    ]


def _merge_ocr_entities(standard: List[Entity], lax: List[Entity]) -> List[Entity]:
    """Merge standard + lax entity lists, dropping lax duplicates."""
    covered = {(e.start, e.end) for e in standard}
    extra = [e for e in lax if (e.start, e.end) not in covered
             and not any(s.start <= e.start and e.end <= s.end for s in standard)]
    return standard + extra


# Singleton reader — loaded once, reused for every call
_reader: Optional[easyocr.Reader] = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        use_gpu = torch.cuda.is_available()
        log.info("Loading EasyOCR reader (first call, gpu=%s)…", use_gpu)
        _reader = easyocr.Reader(["en", "fr"], gpu=use_gpu, verbose=False)
        log.info("EasyOCR reader ready.")
    return _reader


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Region:
    text: str
    left: int
    top: int
    right: int
    bottom: int
    start: int  # char offset in the merged line string
    end: int


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _bbox_to_ltrb(bbox) -> Tuple[int, int, int, int]:
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _ocr_raw(image: Image.Image) -> List[_Region]:
    arr = np.array(image)
    raw = _get_reader().readtext(arr, detail=1, paragraph=False, batch_size=8)
    regions = []
    for bbox, text, conf in raw:
        text = text.strip()
        if conf < _MIN_CONF or not text:
            continue
        left, top, right, bottom = _bbox_to_ltrb(bbox)
        regions.append(_Region(text=text, left=left, top=top,
                               right=right, bottom=bottom, start=0, end=0))
    return regions


# ---------------------------------------------------------------------------
# Line grouping
# ---------------------------------------------------------------------------

def _group_into_lines(regions: List[_Region]) -> List[List[_Region]]:
    """
    Group regions into visual lines using Y-center clustering.

    Two regions belong to the same line when the new region's Y-center is
    within (min_region_height × factor) of the current group's Y-center
    spread.  Using Y-centers (rather than top/bottom) avoids merging
    adjacent document lines whose bounding boxes slightly overlap vertically.
    """
    if not regions:
        return []

    def yc(r: _Region) -> float:
        return (r.top + r.bottom) / 2.0

    by_yc = sorted(regions, key=yc)
    lines: List[List[_Region]] = []
    current: List[_Region] = [by_yc[0]]

    for r in by_yc[1:]:
        centers = [yc(rr) for rr in current]
        max_c = max(centers)
        rc = yc(r)
        min_h = min(rr.bottom - rr.top for rr in current)
        if rc - max_c < min_h * _SAME_LINE_GAP_FACTOR:
            current.append(r)
        else:
            lines.append(sorted(current, key=lambda r: r.left))
            current = [r]

    lines.append(sorted(current, key=lambda r: r.left))
    return lines


# ---------------------------------------------------------------------------
# Smart joining
# ---------------------------------------------------------------------------

def _char_width(r: _Region) -> float:
    w = r.right - r.left
    return w / len(r.text) if r.text else max(float(w), 8.0)


def _merge_line(line: List[_Region]) -> Tuple[str, List[_Region]]:
    """
    Join sorted line regions into one string.

    A space is omitted only when:  0 <= gap < avg_char_width * NO_SPACE_GAP_FACTOR
    Negative gaps (overlapping bboxes) always get a space — they indicate
    regions from different visual layers that were mistakenly grouped together.
    """
    parts: List[str] = []
    updated: List[_Region] = []
    pos = 0

    for i, r in enumerate(line):
        if i > 0:
            prev = line[i - 1]
            gap = r.left - prev.right
            avg_cw = (_char_width(prev) + _char_width(r)) / 2 or 8.0
            threshold = avg_cw * _NO_SPACE_GAP_FACTOR
            # Only skip space for genuinely close positive gaps
            if not (0 <= gap < threshold):
                parts.append(" ")
                pos += 1

        start = pos
        parts.append(r.text)
        pos += len(r.text)
        updated.append(replace(r, start=start, end=pos))

    return "".join(parts), updated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regions_for_entity(regions: List[_Region], start: int, end: int) -> List[_Region]:
    return [r for r in regions if r.start < end and r.end > start]


# Matches text that looks like a bare TLD fragment optionally followed by
# punctuation: "com", ".com", "com)", "org!" etc.
_TLD_FRAGMENT = re.compile(r'^\.?[a-zA-Z]{2,6}[)>.,;:!?\s]?$')


def _precise_bbox(
        matched: List[_Region],
        entity_start: int,
        entity_end: int,
) -> Tuple[int, int, int, int]:
    """
    Return the tightest pixel bbox that covers exactly the entity characters.

    For each matched region we compute which fraction of its width belongs to
    the entity (by character count) and clip the left/right accordingly.
    This prevents including non-PII text that lives in the same OCR region.
    """
    sub_boxes = []
    for r in matched:
        region_len = r.end - r.start
        if region_len == 0:
            sub_boxes.append((r.left, r.top, r.right, r.bottom))
            continue
        char_s = max(entity_start, r.start) - r.start
        char_e = min(entity_end, r.end) - r.start
        w = r.right - r.left
        sub_left = r.left + int(char_s / region_len * w)
        sub_right = r.left + int(char_e / region_len * w)
        # Guarantee at least 1 px width
        sub_right = max(sub_right, sub_left + 1)
        sub_boxes.append((sub_left, r.top, sub_right, r.bottom))

    return (
        min(b[0] for b in sub_boxes),
        min(b[1] for b in sub_boxes),
        max(b[2] for b in sub_boxes),
        max(b[3] for b in sub_boxes),
    )


def _extend_for_tld(
        entity: Entity,
        line_regions: List[_Region],
        entity_end: int,
        bbox: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """
    Extend *bbox* to cover an adjacent TLD fragment when the entity is a
    partial email (has '@' but the TLD was split into a separate OCR region).

    Example: entity text is "finance@company" and the next close region
    contains "com" → extend right edge to cover that region.
    """
    if '@' not in entity.text:
        return bbox
    # Already has a proper TLD — nothing to extend
    if re.search(r'\.[a-zA-Z]{2,}$', entity.text):
        return bbox

    left, top, right, bottom = bbox
    for r in sorted(line_regions, key=lambda r: r.left):
        if r.start < entity_end:
            continue  # before or overlapping the entity
        if r.left - right > 15:  # too far away
            break
        if _TLD_FRAGMENT.match(r.text):
            right = max(right, r.right)
            top = min(top, r.top)
            bottom = max(bottom, r.bottom)
            break

    return left, top, right, bottom


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, max(size, _MIN_FONT_SIZE))
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str,
              box_w: int, box_h: int, initial_size: int
              ) -> Tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, float, float]:
    """
    Return the largest font (and text pixel dimensions) where *text* fits
    entirely inside a box of *box_w* × *box_h* pixels.
    Falls back to the default font at minimum size if nothing fits.
    """
    size = initial_size
    while size >= _MIN_FONT_SIZE:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= box_w - 4 and th <= box_h - 2:  # 4/2 px padding
            return font, tw, th
        size -= 1

    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    return font, bbox[2] - bbox[0], bbox[3] - bbox[1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def anonymize_image(image_bytes: bytes, proxy: "DLPProxy") -> Tuple[bytes, str]:
    """Detect and redact PII in *image_bytes*.

    Returns (result_bytes, ocr_text) where:
      - result_bytes is the redacted PNG (or the original bytes unchanged if no PII found)
      - ocr_text is all text OCR-detected in the image, one line per visual line
    """
    h = _image_hash(image_bytes)
    cached = _load_cache(h)

    if cached is not None:
        log.debug("Image cache hit %s…", h[:12])
        merged, entities_per_line = _restore_cache(cached)
    else:
        image_for_ocr = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw_regions = _ocr_raw(image_for_ocr)
        if not raw_regions:
            _save_cache(h, [], [])
            return image_bytes, ""

        lines = _group_into_lines(raw_regions)
        merged = [_merge_line(line) for line in lines]
        line_texts = [text for text, _ in merged]
        entities_per_line = _detect_entities_ocr_batch(proxy, line_texts, proxy.mappings)
        _save_cache(h, merged, entities_per_line)

    if not merged or not any(entities_per_line):
        return image_bytes, "\n".join(t for t, _ in merged)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    ocr_text = "\n".join(t for t, _ in merged)

    draw = ImageDraw.Draw(image)
    redacted_entities = []
    any_redacted = False

    for (line_text, line_regions), entities in zip(merged, entities_per_line):
        for entity in entities:
            matched = _regions_for_entity(line_regions, entity.start, entity.end)
            if not matched:
                continue

            redacted = proxy.mappings.get_or_set_redacted_text(entity.text, entity.type)
            left, top, right, bottom = _precise_bbox(matched, entity.start, entity.end)
            left, top, right, bottom = _extend_for_tld(entity, line_regions, entity.end,
                                                       (left, top, right, bottom))
            box_w, box_h = right - left, bottom - top

            draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))

            font, tw, th = _fit_font(draw, redacted, box_w, box_h, box_h)
            # Center the label vertically; align left with small padding
            ty = top + (box_h - th) // 2
            draw.text((left + 2, ty), redacted, fill=(255, 255, 255), font=font)

            redacted_entities.append({"entity": entity.text, "redacted": redacted})
            any_redacted = True
            log.debug("Redacted %r → %r at (%d,%d,%d,%d)",
                      entity.text, redacted, left, top, right, bottom)

    if not any_redacted:
        return image_bytes, ocr_text

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue(), ocr_text
