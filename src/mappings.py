class Mappings:
    # maps redacted text to their real values
    _redacted_to_sensitive: dict[str, str] = {}
    # maps sensitive text to their redacted values
    _sensitive_to_redacted: dict[str, str] = {}

    def get_redacted_text(self, sensitive_value: str, value_type: str) -> str:
        key = sensitive_value.upper()  # case-insensitive dedup
        redacted_text = self._sensitive_to_redacted.get(key, None)
        if redacted_text:
            return redacted_text

        redacted_text_id = str(len(self._redacted_to_sensitive))
        redacted_text = "[" + value_type.upper() + "_" + redacted_text_id + "]"

        self._sensitive_to_redacted[key] = redacted_text
        self._redacted_to_sensitive[redacted_text] = sensitive_value  # preserve original casing
        return redacted_text

    def get_sensitive_value(self, redacted_text: str) -> str:
        return self._redacted_to_sensitive.get(redacted_text.upper(), redacted_text)
