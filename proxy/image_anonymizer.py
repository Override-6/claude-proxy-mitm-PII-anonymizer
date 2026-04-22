"""
Image anonymizer: OCR → NER/regex → black-box redaction.

Pipeline for each image:
  1. PaddleOCR extracts per-word text with bounding boxes.
  2. Words are grouped into visual lines and joined with a single space.
  3. Entity detection runs on each merged line string using BOTH the standard
     finders AND OCR-lax regexes that tolerate noisy OCR output (e.g. missing
     TLD dot → \\w+@\\w+ still hits).
  4. Every PII span is mapped back to source word(s), merged into one bbox,
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
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from proxy.engine import DLPProxy, _add_non_overlapping
from proxy.entity_finder import Entity
from proxy.entity_finder.regex_finder import RegexEntityFinder

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_PADDLE_CACHE_DIR = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", str(_REPO_ROOT / "cache" / "paddlex")))
_HF_CACHE_DIR = _REPO_ROOT / "cache" / "huggingface"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_PADDLE_CACHE_DIR))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HOME", str(_HF_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_REPO_ROOT / "cache" / "xdg"))
os.environ.setdefault("MODELSCOPE_CACHE", str(_REPO_ROOT / "cache" / "modelscope"))

from paddleocr import PaddleOCR

# ---------------------------------------------------------------------------
# OCR-lax regex patterns
# These are intentionally looser than the production regexes in text_anonymizer
# because OCR output is noisy: dots can be dropped, TLDs split off, etc.
# They run ONLY on OCR text, never on real user text.
# ---------------------------------------------------------------------------

# Email: needs something@something with ≥2 chars after @.
# No TLD dot required — catches "finance@company" and "finance@companycom".
_OCR_EMAIL_RE = re.compile(r'[\w._%+\-]+@[\w.\-]{2,}')

# Phone (OCR-lax): mirrors text_anonymizer PHONE_REGEX structure.
# No backreference: OCR may produce inconsistent separators within one number.
_OCR_PHONE_RE = re.compile(
    r'(?<!\d)'
    r'(?:'
    r'\+\d{1,3}(?:\d{6,12}|[\s\-.]?\(?\d{1,4}\)?(?:[\s\-().]?\d{2,4}){1,3})'
    r'|'
    r'0\d{1,4}(?:\d{6,10}|(?:[\s\-().]?\d{2,4}){2,4})'
    r'|'
    r'\(\d{2,4}\)[\s\-.]?\d{3,4}[\s\-.]?\d{4}'
    r'|'
    r'\d{2,4}[\s\-]\d{3,4}[\s\-]\d{3,4}'
    r')'
    r'(?!\d)'
)

_ocr_lax_finder = RegexEntityFinder([
    (_OCR_EMAIL_RE, "EMAIL"),
    (_OCR_PHONE_RE, "PHONE"),
])

# ---------------------------------------------------------------------------
# OCR spacing normalisation
#
# PaddleOCR word segmentation splits punctuation into separate tokens, so
# "user@host.tld" becomes "user @ host . tld" in the merged line string.
# We strip those artificial spaces before running entity finders, then remap
# the resulting entity spans back to the original (spaced) string so the
# bounding-box logic still maps entities to the right pixel regions.
# ---------------------------------------------------------------------------

# Each pattern matches spaces that should be removed.  Order does not matter
# because we only mark characters for removal without changing positions.
#
# Symmetric chars (space on both sides collapsed):
#   @  .  _  /  :  ,  -  <  >
# Bracket pairs (inner space collapsed):
#   (…)  […]  {…}  <…>
_OCR_COLLAPSE_PATTERNS = [
    # @
    re.compile(r'(?<=\S) +(?=@)'),
    re.compile(r'(?<=@) +(?=\S)'),
    # .  (only between word chars to avoid collapsing sentence-ending dots)
    re.compile(r'(?<=\w) +(?=\.)'),
    re.compile(r'(?<=\.) +(?=\w)'),
    # _
    re.compile(r'(?<=\w) +(?=_)'),
    re.compile(r'(?<=_) +(?=\w)'),
    # /
    re.compile(r'(?<=\S) +(?=/)'),
    re.compile(r'(?<=/) +(?=\S)'),
    # :
    re.compile(r'(?<=\S) +(?=:)'),
    re.compile(r'(?<=:) +(?=\S)'),
    # ,
    re.compile(r'(?<=\S) +(?=,)'),
    re.compile(r'(?<=,) +(?=\S)'),
    # -
    re.compile(r'(?<=\S) +(?=-)'),
    re.compile(r'(?<=-) +(?=\S)'),
    # < >  (symmetric and bracket-style)
    re.compile(r'(?<=\S) +(?=<)'),
    re.compile(r'(?<=<) +(?=\S)'),
    re.compile(r'(?<=\S) +(?=>)'),
    re.compile(r'(?<=>) +(?=\S)'),
    # (  )
    re.compile(r'(?<=\S) +(?=\()'),
    re.compile(r'(?<=\() +(?=\S)'),
    re.compile(r'(?<=\S) +(?=\))'),
    re.compile(r'(?<=\)) +(?=\S)'),
    # [  ]
    re.compile(r'(?<=\S) +(?=\[)'),
    re.compile(r'(?<=\[) +(?=\S)'),
    re.compile(r'(?<=\S) +(?=\])'),
    re.compile(r'(?<=\]) +(?=\S)'),
    # {  }
    re.compile(r'(?<=\S) +(?=\{)'),
    re.compile(r'(?<=\{) +(?=\S)'),
    re.compile(r'(?<=\S) +(?=\})'),
    re.compile(r'(?<=\}) +(?=\S)'),
]


def _normalize_ocr_text(text: str) -> tuple[str, list[int]]:
    """Collapse OCR spacing artefacts around punctuation.

    Returns (normalized_text, orig_offsets) where orig_offsets[i] is the
    index in *text* that corresponds to position i in normalized_text.
    """
    keep = bytearray(b'\x01' * len(text))
    for pattern in _OCR_COLLAPSE_PATTERNS:
        for m in pattern.finditer(text):
            for j in range(m.start(), m.end()):
                keep[j] = 0

    normalized: list[str] = []
    orig_offsets: list[int] = []
    for i, ch in enumerate(text):
        if keep[i]:
            orig_offsets.append(i)
            normalized.append(ch)

    return ''.join(normalized), orig_offsets


# ---------------------------------------------------------------------------
# Image result cache
# ---------------------------------------------------------------------------

_CACHE_VERSION = 5  # v5: entity.text is normalised, spans still in original coords
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


def _detect_entities_ocr_batch(proxy: DLPProxy, texts: List[str]) -> List[List[Entity]]:
    """Batch version: runs standard+lax finders across all texts in one pass.

    NER is batched via nlp.pipe() — a single model forward pass instead of
    one per OCR line, which is the main OCR performance bottleneck.

    Each line is normalised first (OCR spacing artefacts collapsed) so finders
    see clean text.  Entity spans are then remapped back to positions in the
    original spaced text so bounding-box lookups remain correct.
    """
    norm_texts: list[str] = []
    offset_maps: list[list[int]] = []
    for text in texts:
        norm, offsets = _normalize_ocr_text(text)
        norm_texts.append(norm)
        offset_maps.append(offsets)

    accepted: list[list[Entity]] = [[] for _ in norm_texts]

    for finder in proxy.finders:
        for idx, entities in enumerate(finder.find_entities_batch(norm_texts, proxy.mappings)):
            _add_non_overlapping(accepted[idx], entities)

    laxes = list(_ocr_lax_finder.find_entities_batch(norm_texts, proxy.mappings))
    merged = _merge_ocr_entities(accepted, laxes)

    # Remap entity spans from normalised-text coordinates back to the original
    # spaced-text coordinates so _regions_for_entity can match word boxes.
    result: list[list[Entity]] = []
    for entities, offsets in zip(merged, offset_maps):
        remapped: list[Entity] = []
        for e in entities:
            if not offsets or e.end == 0:
                continue
            orig_start = offsets[min(e.start, len(offsets) - 1)]
            orig_end = offsets[min(e.end - 1, len(offsets) - 1)] + 1
            remapped.append(Entity(e.text, e.type, orig_start, orig_end))
        result.append(remapped)

    return result


def _merge_ocr_entities(standard: List[List[Entity]], lax: List[List[Entity]]) -> List[List[Entity]]:
    """Merge standard + lax entity lists, dropping lax duplicates."""

    def _sub(std: List[Entity], lx: List[Entity]) -> List[Entity]:
        covered = {(e.start, e.end) for e in std}
        extra = [e for e in lx if (e.start, e.end) not in covered
                 and not any(s.start <= e.start and e.end <= s.end for s in std)]
        return std + extra

    return [_sub(s, l) for s, l in zip(standard, lax)]


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


_paddle_ocr: PaddleOCR | None = None
_paddle_ocr_lock = threading.Lock()


def _get_paddle_ocr() -> PaddleOCR:
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    with _paddle_ocr_lock:
        if _paddle_ocr is not None:  # re-check after acquiring lock
            return _paddle_ocr
        _PADDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _paddle_ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            return_word_box=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            cpu_threads=os.cpu_count() or 4,
            enable_mkldnn=False,
            enable_cinn=False,
        )
    return _paddle_ocr


def _bbox_from_points(points) -> Tuple[int, int, int, int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _normalize_token(token: str) -> str:
    return token.strip()


def _extract_paddle_regions(result) -> List[_Region]:
    """Convert PaddleOCR output into per-word regions with image coordinates."""
    regions: List[_Region] = []
    word_texts = result["text_word"]
    word_regions = result["text_word_region"]
    line_texts = result["rec_texts"]
    line_polys = result["rec_polys"]

    for words, boxes in zip(word_texts, word_regions):
        if len(words) != len(boxes):
            continue
        for token, box in zip(words, boxes):
            text = _normalize_token(str(token))
            if not text:
                continue
            left, top, right, bottom = _bbox_from_points(box)
            regions.append(_Region(
                text=text,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                start=0,
                end=0,
            ))

    if regions:
        return regions

    for text, poly in zip(line_texts, line_polys):
        normalized = _normalize_token(str(text))
        if not normalized:
            continue
        left, top, right, bottom = _bbox_from_points(poly)
        regions.append(_Region(
            text=normalized,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            start=0,
            end=0,
        ))
    return regions


def _same_visual_line(region: _Region, line: List[_Region]) -> bool:
    line_top = min(item.top for item in line)
    line_bottom = max(item.bottom for item in line)
    line_center = (line_top + line_bottom) / 2
    region_center = (region.top + region.bottom) / 2
    vertical_overlap = min(region.bottom, line_bottom) - max(region.top, line_top)
    min_height = min(region.bottom - region.top, line_bottom - line_top)
    return (
        abs(region_center - line_center) <= max(8, min_height * 0.7)
        or vertical_overlap >= max(3, min_height * 0.4)
    )


def _group_regions_into_lines(regions: List[_Region]) -> List[List[_Region]]:
    lines: List[List[_Region]] = []
    for region in sorted(regions, key=lambda item: ((item.top + item.bottom) / 2, item.left)):
        matched = False
        for line in lines:
            if _same_visual_line(region, line):
                line.append(region)
                matched = True
                break
        if not matched:
            lines.append([region])

    lines.sort(key=lambda line: min(item.top for item in line))
    return [sorted(line, key=lambda item: item.left) for line in lines]


def _merge_line(line: List[_Region]) -> Tuple[str, List[_Region]]:
    """Join sorted line words into one string, one space between each word."""
    parts: List[str] = []
    updated: List[_Region] = []
    pos = 0

    for i, r in enumerate(line):
        if i > 0:
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(r.text)
        pos += len(r.text)
        updated.append(replace(r, start=start, end=pos))

    return "".join(parts), updated


def _collect_ocr_lines(image: Image.Image) -> List[List[_Region]]:
    arr = np.array(image.convert("RGB"))
    results = _get_paddle_ocr().predict(arr)
    if not results:
        return []
    regions = _extract_paddle_regions(results[0])
    return _group_regions_into_lines(regions)


def extract_text(image_bytes: bytes) -> str:
    """Return the best-effort OCR text for *image_bytes* without redaction."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    lines = _collect_ocr_lines(image)
    return "\n".join(_merge_line(line)[0] for line in lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regions_for_entity(regions: List[_Region], start: int, end: int) -> List[_Region]:
    return [r for r in regions if r.start < end and r.end > start]


_TLD_FRAGMENT = re.compile(r'^\.?[a-zA-Z]{2,6}[)>.,;:!?\s]?$')


def _precise_bbox(
        matched: List[_Region],
        entity_start: int,
        entity_end: int,
) -> Tuple[int, int, int, int]:
    """
    Return the tightest pixel bbox covering exactly the entity characters.

    For each matched region we compute which fraction of its width belongs
    to the entity (by character count) and clip left/right accordingly.
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
    partial email (has '@' but the TLD was split into a separate OCR token).
    """
    if '@' not in entity.text:
        return bbox
    if re.search(r'\.\s*[a-zA-Z]{2,}$', entity.text):
        return bbox

    left, top, right, bottom = bbox
    for r in sorted(line_regions, key=lambda r: r.left):
        if r.start < entity_end:
            continue
        if r.left - right > 15:
            break
        if _TLD_FRAGMENT.match(r.text):
            right = max(right, r.right)
            top = min(top, r.top)
            bottom = max(bottom, r.bottom)
            break

    return left, top, right, bottom


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "arial.ttf",
    "Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_MIN_FONT_SIZE = 6


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
    """
    size = initial_size
    while size >= _MIN_FONT_SIZE:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= box_w - 4 and th <= box_h - 2:
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
      - result_bytes is the redacted PNG (or the original bytes unchanged if
        no PII was found)
      - ocr_text is all text detected in the image, one line per visual line
    """
    h = _image_hash(image_bytes)
    cached = _load_cache(h)

    if cached is not None:
        log.debug("Image cache hit %s…", h[:12])
        merged, entities_per_line = _restore_cache(cached)
    else:
        image_for_ocr = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        lines = _collect_ocr_lines(image_for_ocr)

        if not lines:
            _save_cache(h, [], [])
            return image_bytes, ""

        merged = [_merge_line(line) for line in lines]
        line_texts = [text for text, _ in merged]
        entities_per_line = _detect_entities_ocr_batch(proxy, line_texts)
        _save_cache(h, merged, entities_per_line)

    if not merged or not any(entities_per_line):
        return image_bytes, "\n".join(t for t, _ in merged)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ocr_text = "\n".join(t for t, _ in merged)

    draw = ImageDraw.Draw(image)
    any_redacted = False

    for (line_text, line_regions), entities in zip(merged, entities_per_line):
        for entity in entities:
            matched = _regions_for_entity(line_regions, entity.start, entity.end)
            if not matched:
                continue

            redacted = proxy.mappings.get_or_set_redacted_text(entity.text, entity.type)
            left, top, right, bottom = _precise_bbox(matched, entity.start, entity.end)
            left, top, right, bottom = _extend_for_tld(
                entity, line_regions, entity.end, (left, top, right, bottom)
            )
            box_w, box_h = right - left, bottom - top

            draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))

            font, tw, th = _fit_font(draw, redacted, box_w, box_h, box_h)
            ty = top + (box_h - th) // 2
            draw.text((left + 2, ty), redacted, fill=(255, 255, 255), font=font)

            any_redacted = True
            log.debug("Redacted %r → %r at (%d,%d,%d,%d)",
                      entity.text, redacted, left, top, right, bottom)

    if not any_redacted:
        return image_bytes, ocr_text

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue(), ocr_text
