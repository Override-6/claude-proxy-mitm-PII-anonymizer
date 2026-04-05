"""
Test NEREntityFinder on text passed as command-line arguments.

Usage:
    poetry run python scripts/test_ner.py "John Smith works at Apple in Paris"
    poetry run python scripts/test_ner.py "text one" "text two" "text three"
    echo "some text" | poetry run python scripts/test_ner.py -
"""

import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from src.entity_finder.ner_finder import NEREntityFinder  # noqa: E402
from mappings import Mappings  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: test_ner.py <text> [text2 ...]  or  test_ner.py -  (read from stdin)")
        sys.exit(1)

    if sys.argv[1] == "-":
        texts = [sys.stdin.read().strip()]
    else:
        texts = sys.argv[1:]

    finder = NEREntityFinder()
    mappings = Mappings()

    for text in texts:
        print(f"\nInput : {text}")
        t0 = time.time()
        entities = next(finder.find_entities_batch([text], mappings))
        t1 = time.time()
        print(f"Completed in {t1 - t0} seconds")
        if not entities:
            print("  (no entities found)")
        else:
            for e in entities:
                print(f"  [{e.type}] {e.text!r}  (chars {e.start}–{e.end})")


if __name__ == "__main__":
    main()
