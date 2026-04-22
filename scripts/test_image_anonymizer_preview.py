#!/usr/bin/env python3
"""OCR, anonymize, and optionally save a redacted copy of a screenshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proxy.image_anonymizer import _collect_ocr_lines, _merge_line, anonymize_image, extract_text


DEFAULT_IMAGE = ROOT / "data" / "preview.webp"


def _build_proxy():
    """Build a minimal DLPProxy sufficient for entity detection."""
    from proxy.engine import DLPProxy, ProxyOptions
    from proxy.entity_finder.ner_finder import NEREntityFinder
    from proxy.entity_finder.presidio_finder import PresidioEntityFinder
    from proxy.mappings import Mappings
    from proxy.rules import load_rules
    return DLPProxy(
        mappings=Mappings(),
        rules=load_rules(),
        finders=[PresidioEntityFinder(), NEREntityFinder()],
        options=ProxyOptions(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR and anonymize an image.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write the anonymized image to this path (PNG).",
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="Also print each extracted word with its bounding box.",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.exists():
        print(f"[error] Image not found: {image_path}", file=sys.stderr)
        return 1

    image_bytes = image_path.read_bytes()
    print(f"[image] {image_path}")

    if args.output is not None:
        proxy = _build_proxy()
        result_bytes, ocr_text = anonymize_image(image_bytes, proxy)
        out_path = args.output.resolve()
        out_path.write_bytes(result_bytes)
        print(f"[output] Anonymized image written to {out_path}")
        print("\n===== full extracted text =====")
        print(ocr_text)
    else:
        print("\n===== full extracted text =====")
        print(extract_text(image_bytes))

    if args.show_boxes:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        lines = _collect_ocr_lines(image)
        print("\n===== extracted words with boxes =====")
        for line_idx, line in enumerate(lines):
            line_text, merged_regions = _merge_line(line)
            print(f"[line {line_idx}] {line_text}")
            for region in merged_regions:
                print(
                    f"  {region.text!r} "
                    f"box=({region.left},{region.top},{region.right},{region.bottom})"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
