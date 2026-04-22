import re
import threading

import ahocorasick

REDACTED_REGEX = re.compile(r'\[([A-Z_]+)_[0-9]+]')


class Mappings:
    # maps redacted text to their real values
    _redacted_to_sensitive: dict[str, str]
    # maps sensitive text to their redacted values
    _sensitive_to_redacted: dict[str, str]

    def __init__(self):
        self._lock = threading.Lock()
        self._redacted_to_sensitive = {}
        self._sensitive_to_redacted = {}
        self._automaton: ahocorasick.Automaton | None = None  # rebuilt on demand

    def _invalidate_automaton(self) -> None:
        """Must be called under self._lock whenever mappings change."""
        self._automaton = None

    def build_automaton(self) -> ahocorasick.Automaton:
        """Return a ready-to-search Automaton, rebuilding if the mapping changed."""
        with self._lock:
            if self._automaton is not None:
                return self._automaton
            A = ahocorasick.Automaton(ahocorasick.STORE_INTS)
            for token, sensitive in self._redacted_to_sensitive.items():
                # Store the original-casing sensitive value (key: lowercase for case-insensitive)
                A.add_word(sensitive.lower(), len(sensitive))
            if len(A):
                A.make_automaton()
            self._automaton = A
            return self._automaton

    def get_redacted_text_type(self, sensitive_value: str) -> str | None:
        with self._lock:
            redacted_text = self._sensitive_to_redacted.get(sensitive_value.upper(), None)
        if not redacted_text:
            return None
        m = REDACTED_REGEX.match(redacted_text)
        return m.group(1) if m else None

    def get_or_set_redacted_text(self, sensitive_value: str, value_type: str) -> str:
        if REDACTED_REGEX.search(sensitive_value):
            return sensitive_value  # never store a value that contains a token
        key = sensitive_value.upper()  # case-insensitive dedup
        with self._lock:
            redacted_text = self._sensitive_to_redacted.get(key, None)
            if redacted_text:
                return redacted_text

            redacted_text_id = str(len(self._redacted_to_sensitive))
            redacted_text = "[" + value_type.upper() + "_" + redacted_text_id + "]"

            self._sensitive_to_redacted[key] = redacted_text
            self._redacted_to_sensitive[redacted_text] = sensitive_value
            self._invalidate_automaton()
            return redacted_text

    def get_sensitive_value(self, redacted_text: str) -> str:
        with self._lock:
            return self._redacted_to_sensitive.get(redacted_text.upper(), "<SECRET_NOT_FOUND-" + redacted_text + ">")

    def dump(self) -> list[dict]:
        """Return a snapshot of all mappings (safe to call from any thread)."""
        with self._lock:
            return [
                {"sensitive": s, "redacted": r}
                for s, r in self._sensitive_to_redacted.items()
            ]

    def reset(self) -> None:
        """Atomically clear all mappings."""
        with self._lock:
            self._sensitive_to_redacted.clear()
            self._redacted_to_sensitive.clear()
            self._automaton = None
