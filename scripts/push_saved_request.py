"""
Replay requests from data/requests-sample.jsonl through the anonymization pipeline.

Directly invokes engine.anonymize_message from src/ — no network calls, no proxy,
no forwarding to the real API. Useful for benchmarking anonymization performance.

Usage:
    poetry run python scripts/push_saved_request.py                    # replay all
    poetry run python scripts/push_saved_request.py path/to/file.jsonl # specific file
    poetry run python scripts/push_saved_request.py --limit 5          # first 5
    poetry run python scripts/push_saved_request.py --index 2          # single entry
    poetry run python scripts/push_saved_request.py --dry-run          # list only
    poetry run python scripts/push_saved_request.py --output out.json  # save result
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from cache import clear_entity_cache
from engine import DLPProxy, ProxyOptions, anonymize_message
from entity_finder.ner_finder import NEREntityFinder
from entity_finder.presidio_finder import PresidioEntityFinder
from mappings import Mappings
from rules import load_rules
from anxious_filter import anxious_filter

_DEFAULT_LOG = Path(__file__).parent.parent / "data" / "requests-sample.jsonl"
_DEFAULT_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def _build_proxy() -> DLPProxy:
    return DLPProxy(
        mappings=Mappings(),
        rules=load_rules(),
        finders=[
            PresidioEntityFinder(),
            NEREntityFinder(),
        ],
        options=ProxyOptions(
            anxious_filter=True,
            save_redacted_images=False,
            inject_system_prompt=False,
            save_requests=False,
        ),
    )


def load_entries(path: Path) -> list[dict]:
    """Load JSONL entries, or wrap a raw JSON request body as a single entry."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "url" not in obj:
            print(f"[push] Detected raw request body — wrapping as single entry with URL {_DEFAULT_MESSAGES_URL}")
            return [{"url": _DEFAULT_MESSAGES_URL, "method": "POST",
                     "headers": {"content-type": "application/json"}, "body": obj}]
    except json.JSONDecodeError:
        pass

    entries = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[warn] Skipping malformed line {lineno}: {e}", file=sys.stderr)
    return entries


def _entry_to_parts(entry: dict) -> tuple[str, dict, bytes]:
    """Return (url, headers, body_bytes) from a log entry."""
    url = entry.get("url", _DEFAULT_MESSAGES_URL)
    headers = dict(entry.get("headers", {}))
    body = entry.get("body")

    if isinstance(body, (dict, list)):
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers.setdefault("content-type", "application/json")
    elif isinstance(body, str):
        body_bytes = body.encode("utf-8")
    else:
        body_bytes = b""

    return url, headers, body_bytes


async def _process(proxy: DLPProxy, entry: dict, output_path: Path | None = None) -> tuple[str, float, list]:
    """Run anonymization on one entry. Returns (status, elapsed_seconds, anxious_entities)."""
    proxy.mappings.reset()
    clear_entity_cache()

    url, headers, content = _entry_to_parts(entry)
    t0 = time.perf_counter()
    new_content = await anonymize_message(proxy, headers, content, url, proxy.rules.anonymise_requests)
    elapsed = time.perf_counter() - t0

    if new_content is None:
        return "no_match", elapsed, []

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(new_content)
        print(f"[output] Saved anonymized body → {output_path}")

    if any(p.match(url.split("?")[0]) for p in proxy.rules.anxious_filter_domains):
        triggered, entities = anxious_filter(proxy.mappings, new_content.decode("utf-8", errors="ignore"))
        if triggered:
            return "anxious", elapsed, entities

    return "ok", elapsed, []


def print_summary(times: list[float], by_status: dict[str, int]) -> None:
    n = len(times)
    if n == 0:
        return
    total = sum(times)
    print(f"\n--- Summary ({n} request{'s' if n > 1 else ''}) ---")
    print(f"  Total : {total:.3f}s")
    print(f"  Mean  : {total / n:.3f}s")
    print(f"  Min   : {min(times):.3f}s")
    print(f"  Max   : {max(times):.3f}s")
    for status, count in sorted(by_status.items()):
        print(f"  {status:<12}: {count}")


async def _run(proxy: DLPProxy, entries: list[dict], dry_run: bool, output_path: Path | None = None) -> None:
    if dry_run:
        for i, entry in enumerate(entries):
            print(f"  [{i}] {entry.get('method', '?')} {entry.get('url', '?')}")
        return

    times: list[float] = []
    by_status: dict[str, int] = {}

    for i, entry in enumerate(entries):
        label = f"[{i + 1}/{len(entries)}] {entry.get('method', '?')} {entry.get('url', '?')}"
        print(label, end=" ... ", flush=True)
        entry_output = output_path if len(entries) == 1 else (
            output_path.with_stem(f"{output_path.stem}_{i}") if output_path else None
        )
        try:
            status, elapsed, anxious_entities = await _process(proxy, entry, entry_output)
            times.append(elapsed)
            by_status[status] = by_status.get(status, 0) + 1
            suffix = f"  entities={[e.text for e in anxious_entities]}" if anxious_entities else ""
            print(f"{status}  {elapsed:.3f}s{suffix}")
        except Exception as e:
            by_status["error"] = by_status.get("error", 0) + 1
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    print_summary(times, by_status)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the anonymization pipeline on saved requests (no network)."
    )
    parser.add_argument("file", type=Path, nargs="?", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    log_path = args.file if args.file is not None else _DEFAULT_LOG
    if not log_path.exists():
        print(f"[error] Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_entries(log_path)
    if not entries:
        print("[error] No entries in log file.", file=sys.stderr)
        sys.exit(1)

    if args.index is not None:
        if args.index >= len(entries):
            print(f"[error] Index {args.index} out of range ({len(entries)} entries).", file=sys.stderr)
            sys.exit(1)
        entries = [entries[args.index]]
    elif args.limit is not None:
        entries = entries[:args.limit]

    print("[push] Loading models...")
    proxy = _build_proxy()
    print(f"[push] {len(entries)} request(s) from {log_path}")
    asyncio.run(_run(proxy, entries, args.dry_run, output_path=args.output))


if __name__ == "__main__":
    main()
