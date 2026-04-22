"""
System prompt injection for the Anthropic Messages API.

Reads SYSTEM_PROMPT.md from the project root and prepends its content to the
`system` field of /v1/messages requests, after PII anonymization has run.

The file is re-read on every request — edit it and the next message picks up
the change without restarting the proxy.
"""

import re
from pathlib import Path

_PROMPT_FILE = Path(__file__).parent.parent / "assets" / "SYSTEM_PROMPT.md"

# Only inject into the Messages API (create + count_tokens), not other endpoints.
_MESSAGES_API_RE = re.compile(
    r"https://(api|a-api)\.anthropic\.com/v1/messages(/count_tokens)?"
)


def applies_to(url: str) -> bool:
    return bool(_MESSAGES_API_RE.fullmatch(url))


def _load() -> str:
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def inject(body: dict, url: str) -> dict:
    """Prepend SYSTEM_PROMPT.md content to the `system` field of *body*.

    Only acts when *url* matches the Anthropic Messages API.
    Handles both string and content-block-array forms of the `system` field.
    Returns *body* unchanged when the URL doesn't match or the file is empty.
    """
    if not applies_to(url):
        return body

    our_prompt = _load()
    if not our_prompt:
        return body

    existing = body.get("system")

    # Already injected — don't duplicate (Claude Desktop resends the full
    # conversation on every turn, so this request may have been processed before)
    if isinstance(existing, str) and existing.endswith(our_prompt):
        return body
    if isinstance(existing, list) and existing and \
            existing[-1].get("type") == "text" and existing[-1].get("text") == our_prompt:
        return body

    if not existing:
        new_system = our_prompt
    elif isinstance(existing, str):
        new_system = existing + "\n\n" + our_prompt
    elif isinstance(existing, list):
        new_system = existing + [{"type": "text", "text": our_prompt}]
    else:
        return body

    return {**body, "system": new_system}
