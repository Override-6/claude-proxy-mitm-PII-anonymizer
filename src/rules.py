import json
import re
from dataclasses import dataclass


@dataclass
class RequestRule:
    url_pattern: re.Pattern
    sensitive_fields: list[str] | bool
    sse_fields: list[str] | None = None


@dataclass
class BlockedUrl:
    url_pattern: re.Pattern


@dataclass
class ProxyRules:
    blocked_urls: list[BlockedUrl]
    request_rules: list[RequestRule]
    response_rules: list[RequestRule]
    mcp_rules: list[RequestRule]
    exempt_words: frozenset[str]
    anxiety_watchlist: list[re.Pattern]


def _parse_rule_list(raw_rules: list[dict]) -> list[RequestRule]:
    result = []
    for rule in raw_rules:
        fields = rule["sensitive_fields"]
        sse_raw = rule.get("sse_fields")
        result.append(RequestRule(
            re.compile(rule["url_pattern"]),
            True if fields is True else list(fields),
            list(sse_raw) if sse_raw else None,
        ))
    return result


def load_rules() -> ProxyRules:
    with open("rules.json", "r") as f:
        rules = json.loads(f.read())

    blocked = [
        BlockedUrl(re.compile(b["url_pattern"]))
        for b in rules.get("blocked_urls", [])
    ]

    exempt = frozenset(w.lower() for w in rules.get("exempt_words", []))

    rules_watchlist = [re.compile(pattern) for pattern in rules["anxiety_watchlist"]]

    return ProxyRules(
        blocked,
        _parse_rule_list(rules.get("request_rules", [])),
        _parse_rule_list(rules.get("response_rules", [])),
        _parse_rule_list(rules.get("mcp_rules", [])),
        exempt,
        rules_watchlist
    )
