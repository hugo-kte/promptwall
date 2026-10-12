"""Credential detection: keep API keys, tokens and private keys out of third-party models.

Patterns follow the documented formats of each provider (the same approach as gitleaks or
GitHub secret scanning), so they fire on real credentials and almost never on prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Secret:
    kind: str
    start: int
    end: int


_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("AWS_ACCESS_KEY_ID", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "GITHUB_TOKEN",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,90})\b"),
    ),
    ("OPENAI_API_KEY", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")),
    ("ANTHROPIC_API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("STRIPE_SECRET_KEY", re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b")),
    (
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z]+ )?PRIVATE KEY-----"
        ),
    ),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "PASSWORD_ASSIGNMENT",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\b\s*[:=]\s*[\"']?(?P<value>[^\s\"']{8,})"
        ),
    ),
]


def detect(text: str) -> list[Secret]:
    found: list[Secret] = []
    for kind, pattern in _RULES:
        for m in pattern.finditer(text):
            # for assignments only the value is sensitive, keep the key name readable
            start, end = (
                (m.start("value"), m.end("value")) if "value" in pattern.groupindex else m.span()
            )
            if all(end <= s.start or start >= s.end for s in found):
                found.append(Secret(kind, start, end))
    return sorted(found, key=lambda s: s.start)


def redact(text: str) -> tuple[str, list[Secret]]:
    found = detect(text)
    out, cursor = [], 0
    for s in found:
        out += [text[cursor : s.start], f"[REDACTED_{s.kind}]"]
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out), found
