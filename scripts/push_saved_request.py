"""
Replay requests from data/requests-sample.jsonl through the anonymization pipeline.

Directly invokes apply_request_rules from src/ — no network calls, no proxy, no
forwarding to the real API. Useful for benchmarking anonymization performance.

Usage:
    python scripts/push_saved_request.py                        # replay all (default file)
    python scripts/push_saved_request.py path/to/file.jsonl     # use a specific file
    python scripts/push_saved_request.py --limit 5              # first 5
    python scripts/push_saved_request.py --index 2              # single entry
    python scripts/push_saved_request.py --dry-run              # list without processing
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Make src/ importable
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from rule_applier import apply_request_rules, rules, apply_mcp_request_rules, state  # noqa: E402
from proxy import anxious_filter  # noqa: E402
from text_anonymizer import clear_entity_cache  # noqa: E402

_DEFAULT_LOG = Path(__file__).parent.parent / "data" / "requests-sample.jsonl"


# ---------------------------------------------------------------------------
# Minimal fake mitmproxy flow — only the fields apply_request_rules touches
# ---------------------------------------------------------------------------

class _FakeRequest:
    def __init__(self, url: str, method: str, headers: dict, body: bytes):
        self.pretty_url = url
        self.method = method
        self.headers = dict(headers)
        self._content = body

    def get_content(self) -> bytes:
        return self._content

    def set_content(self, content: bytes) -> None:
        self._content = content


class _FakeFlow:
    def __init__(self, request: _FakeRequest):
        self.request = request
        self.response = None  # set to a Response object by apply_request_rules on error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def load_entries(path: Path) -> list[dict]:
    """Load JSONL entries, or wrap a raw JSON request body as a single entry."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # Detect raw JSON body (dict with no 'url' key at top level → not a log entry)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "url" not in obj:
            print(f"[push] Detected raw request body — wrapping as single entry with URL {_DEFAULT_MESSAGES_URL}")
            return [{"url": _DEFAULT_MESSAGES_URL, "method": "POST",
                     "headers": {"content-type": "application/json"}, "body": obj}]
    except json.JSONDecodeError:
        pass

    # Standard JSONL
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


def _entry_to_flow(entry: dict) -> _FakeFlow:
    url = entry.get("url", "")
    method = entry.get("method", "POST").upper()
    headers = entry.get("headers", {})
    body = entry.get("body")

    if isinstance(body, (dict, list)):
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers.setdefault("content-type", "application/json")
    elif isinstance(body, str):
        body_bytes = body.encode("utf-8")
    else:
        body_bytes = b""

    return _FakeFlow(_FakeRequest(url, method, headers, body_bytes))


async def _process(entry: dict, output_path: Path | None = None) -> tuple[str, float, list]:
    """Run anonymization on one entry. Returns (status, elapsed_seconds, anxious_entities).

    status is one of: 'ok', 'err_502', 'anxious_403'.
    """
    state["mappings"].reset()
    clear_entity_cache()
    flow = _entry_to_flow(entry)
    url = entry.get("url", "").split("?")[0]
    t0 = time.perf_counter()
    await apply_request_rules(flow, rules.request_rules)
    apply_mcp_request_rules(flow, rules.mcp_rules)
    elapsed = time.perf_counter() - t0

    if flow.response is not None:
        return "err_502", elapsed, []

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(flow.request.get_content())
        print(f"[output] Saved anonymized body → {output_path}")

    if any(p.match(url) for p in rules.anxiety_watchlist):
        body = flow.request.get_content().decode("utf-8", errors="ignore")
        triggered, entities = anxious_filter(state["mappings"], body)
        if triggered:
            return "anxious_403", elapsed, entities

    return "ok", elapsed, []


def print_summary(times: list[float], errors: int, anxious: int) -> None:
    n = len(times)
    if n == 0:
        return
    total = sum(times)
    print(f"\n--- Performance summary ({n} request{'s' if n > 1 else ''}) ---")
    print(f"  Total : {total:.3f}s")
    print(f"  Mean  : {total / n:.3f}s")
    print(f"  Min   : {min(times):.3f}s")
    print(f"  Max   : {max(times):.3f}s")
    print(f"  OK    : {n - errors - anxious}")
    print(f"  Errors (502)         : {errors}")
    print(f"  Anxious filter (403) : {anxious}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(entries: list[dict], dry_run: bool, output_path: Path | None = None) -> None:
    if dry_run:
        for i, entry in enumerate(entries):
            print(f"  [{i}] {entry.get('method', '?')} {entry.get('url', '?')}")
        return

    times: list[float] = []
    errors = 0
    anxious = 0
    for i, entry in enumerate(entries):
        label = f"[{i + 1}/{len(entries)}] {entry.get('method', '?')} {entry.get('url', '?')}"
        print(label, end=" ... ", flush=True)
        # Only write output for single-entry runs to avoid clobbering with multiple entries
        entry_output = output_path if len(entries) == 1 else (
            output_path.with_stem(f"{output_path.stem}_{i}") if output_path else None
        )
        try:
            status, elapsed, anxious_entities = await _process(entry, entry_output)
            times.append(elapsed)
            if status == "err_502":
                errors += 1
            elif status == "anxious_403":
                anxious += 1
                print(f"{status}  {elapsed:.3f}s  entities={[e.text for e in anxious_entities]}")
                continue
            print(f"{status}  {elapsed:.3f}s")
        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")

    print_summary(times, errors, anxious)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the anonymization pipeline on saved requests (no network)."
    )
    parser.add_argument("file", type=Path, nargs="?", default=None,
                        help="Path to JSONL request log (default: data/requests-sample.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of requests to process")
    parser.add_argument("--index", type=int, default=None,
                        help="Process only the request at this 0-based index")
    parser.add_argument("--dry-run", action="store_true",
                        help="List requests without processing them")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save anonymized request body to this file")
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
            print(f"[error] Index {args.index} out of range ({len(entries)} entries).",
                  file=sys.stderr)
            sys.exit(1)
        entries = [entries[args.index]]
    elif args.limit is not None:
        entries = entries[:args.limit]

    print(f"[push] {len(entries)} request(s) from {log_path}")
    asyncio.run(_run(entries, args.dry_run, output_path=args.output))


if __name__ == "__main__":
    main()
