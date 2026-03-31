import re
import threading

from flashtext import KeywordProcessor

REDACTED_REGEX = re.compile(r'\[([A-Z_]+)_[0-9]+]')


class Mappings:
    # maps redacted text to their real values
    _redacted_to_sensitive: dict[str, str]
    # maps sensitive text to their redacted values
    _sensitive_to_redacted: dict[str, str]

    kp: KeywordProcessor

    def __init__(self):
        self._lock = threading.Lock()
        self.kp = KeywordProcessor(case_sensitive=False)
        self._redacted_to_sensitive = {}
        self._sensitive_to_redacted = {}

    def get_redacted_text_type(self, sensitive_value: str) -> str | None:
        with self._lock:
            redacted_text = self._sensitive_to_redacted.get(sensitive_value, None)
        if not redacted_text:
            return None
        m = REDACTED_REGEX.match(redacted_text)
        return m.group(1) if m else None

    def get_or_set_redacted_text(self, sensitive_value: str, value_type: str) -> str:
        key = sensitive_value.upper()  # case-insensitive dedup
        with self._lock:
            redacted_text = self._sensitive_to_redacted.get(key, None)
            if redacted_text:
                return redacted_text

            redacted_text_id = str(len(self._redacted_to_sensitive))
            redacted_text = "[" + value_type.upper() + "_" + redacted_text_id + "]"

            self._sensitive_to_redacted[key] = redacted_text
            self._redacted_to_sensitive[redacted_text] = sensitive_value  # preserve original casing
            self.kp.add_keyword(sensitive_value)
            return redacted_text

    def get_sensitive_value(self, redacted_text: str) -> str:
        with self._lock:
            return self._redacted_to_sensitive.get(redacted_text.upper(), redacted_text)

    def dump(self) -> list[dict]:
        """Return a snapshot of all mappings (safe to call from any thread)."""
        with self._lock:
            return [
                {"sensitive": s, "redacted": r}
                for s, r in self._sensitive_to_redacted.items()
            ]

    def clear_all(self) -> None:
        """Atomically clear all mappings."""
        with self._lock:
            self._sensitive_to_redacted.clear()
            self._redacted_to_sensitive.clear()
            self.kp = KeywordProcessor(case_sensitive=False)
