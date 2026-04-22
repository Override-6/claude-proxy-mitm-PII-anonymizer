import json
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BlockedUrl:
    url_pattern: re.Pattern


@dataclass
class AnonymiseRule:
    url_pattern: re.Pattern
    sensitive_fields: list[str] | bool


@dataclass
class DeanonymiseRule:
    url_pattern: re.Pattern
    sensitive_fields: list[str] | bool
    sse_fields: list[str] | None = None


@dataclass
class ProxyRules:
    anxious_filter_domains: list[re.Pattern]
    blocked_urls: list[BlockedUrl]
    anonymise_requests: list[AnonymiseRule]
    anonymise_responses: list[AnonymiseRule]
    deanonymise_responses: list[DeanonymiseRule]
    # URLs that are safe to forward as-is (no PII payload). Used by the
    # default-deny guard in the request hook: anything on an intercepted host
    # that is neither covered by an anonymise/deanonymise rule nor whitelisted
    # here gets blocked.
    known_safe_routes: list[re.Pattern]

    def matches_any_rule(self, url: str) -> bool:
        """True if *url* is covered by any rule or whitelisted as safe."""
        patterns = (
            [r.url_pattern for r in self.blocked_urls]
            + [r.url_pattern for r in self.anonymise_requests]
            + [r.url_pattern for r in self.anonymise_responses]
            + [r.url_pattern for r in self.deanonymise_responses]
            + self.known_safe_routes
        )
        return any(p.fullmatch(url) for p in patterns)


# ---------------------------------------------------------------------------
# JSONC (JSON with // line comments and /* */ block comments) parser
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Strip // and /* */ comments from JSON-like text, respecting string literals."""
    result = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        if in_string:
            c = text[i]
            if c == '\\' and i + 1 < n:
                result.append(c)
                result.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            result.append(c)
            i += 1
        else:
            if text[i] == '"':
                in_string = True
                result.append(text[i])
                i += 1
            elif text[i:i + 2] == '//':
                while i < n and text[i] != '\n':
                    i += 1
            elif text[i:i + 2] == '/*':
                i += 2
                while i < n - 1 and text[i:i + 2] != '*/':
                    i += 1
                i += 2
            else:
                result.append(text[i])
                i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_anonymise_rules(raw: list[dict]) -> list[AnonymiseRule]:
    rules = []
    for r in raw:
        fields = r["sensitive_fields"]
        rules.append(AnonymiseRule(
            url_pattern=re.compile(r["url_pattern"]),
            sensitive_fields=True if fields is True else list(fields),
        ))
    return rules


def _parse_deanonymise_rules(raw: list[dict]) -> list[DeanonymiseRule]:
    rules = []
    for r in raw:
        fields = r["sensitive_fields"]
        sse_raw = r.get("sse_fields")
        rules.append(DeanonymiseRule(
            url_pattern=re.compile(r["url_pattern"]),
            sensitive_fields=True if fields is True else list(fields),
            sse_fields=list(sse_raw) if sse_raw else None,
        ))
    return rules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rules(path: str = "assets/rules.jsonc") -> ProxyRules:
    with open(path, "r") as f:
        raw_text = f.read()

    data = json.loads(_strip_comments(raw_text))

    blocked = [
        BlockedUrl(re.compile(b["url_pattern"]))
        for b in data.get("blocked_urls", [])
    ]

    anxious = [
        re.compile(p)
        for p in data.get("anxious_filter_domains", [])
    ]

    anonymise = data.get("anonymise", {})
    deanonymise = data.get("deanonymise", {})

    known_safe = [re.compile(p) for p in data.get("known_safe_routes", [])]

    return ProxyRules(
        anxious_filter_domains=anxious,
        blocked_urls=blocked,
        anonymise_requests=_parse_anonymise_rules(anonymise.get("requests", [])),
        anonymise_responses=_parse_anonymise_rules(anonymise.get("responses", [])),
        deanonymise_responses=_parse_deanonymise_rules(deanonymise.get("responses", [])),
        known_safe_routes=known_safe,
    )
