import json
import re
from dataclasses import dataclass


@dataclass
class RequestRule:
    url_pattern: re.Pattern
    sensitive_fields: list[list[str]] | bool
    sse_fields: list[list[str]] | None = None


@dataclass
class BlockedUrl:
    url_pattern: re.Pattern


@dataclass
class ProxyRules:
    blocked_urls: list[BlockedUrl]
    request_rules: list[RequestRule]
    response_rules: list[RequestRule]


def parse_jq_path(path: str) -> list[str]:
    """Parse a jq-style path into segments.

    '.messages[].content[].text' → ['messages', '[]', 'content', '[]', 'text']
    """
    segments = []
    for part in path.lstrip(".").split("."):
        if not part:
            continue
        if part.endswith("[]"):
            segments.append(part[:-2])
            segments.append("[]")
        else:
            segments.append(part)
    return segments


def _parse_fields(fields) -> list[list[str]] | bool:
    return True if fields is True else [parse_jq_path(p) for p in fields]


def _parse_rule_list(raw_rules: list[dict]) -> list[RequestRule]:
    result = []
    for rule in raw_rules:
        sse_raw = rule.get("sse_fields")
        sse_parsed = [parse_jq_path(p) for p in sse_raw] if sse_raw else None
        result.append(RequestRule(
            re.compile(rule["url_pattern"]),
            _parse_fields(rule["sensitive_fields"]),
            sse_parsed,
        ))
    return result


def load_rules() -> ProxyRules:
    with open("rules.json", "r") as f:
        rules = json.loads(f.read())

    blocked = [
        BlockedUrl(re.compile(b["url_pattern"]))
        for b in rules.get("blocked_urls", [])
    ]

    return ProxyRules(
        blocked,
        _parse_rule_list(rules.get("request_rules", [])),
        _parse_rule_list(rules.get("response_rules", [])),
    )
