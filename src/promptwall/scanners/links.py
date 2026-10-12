"""Neutralise remote images and links to non-allowlisted domains in model output.

A chat UI that renders markdown fetches remote images automatically, so an image URL carrying
conversation data in its query string leaks that data without any click (OWASP LLM05,
improper output handling). Images are removed unless their host is allowlisted; links are
kept as text but lose their target.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

_MD_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_LINK = re.compile(r"(?<!!)\[(?P<text>[^\]]+)\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_IMG = re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"']?(?P<url>[^\"'\s>]+)[^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class LinkFinding:
    kind: str  # "image" or "link"
    host: str


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _allowed(url: str, allowed: set[str]) -> bool:
    host = _host(url)
    if not host:  # relative URL: served by the application itself
        return True
    return any(host == d or host.endswith("." + d) for d in allowed)


def sanitise(text: str, allowed_domains: Iterable[str] = ()) -> tuple[str, list[LinkFinding]]:
    allowed = {d.lower() for d in allowed_domains}
    findings: list[LinkFinding] = []

    def image(m: re.Match[str]) -> str:
        if _allowed(m.group("url"), allowed):
            return m.group(0)
        findings.append(LinkFinding("image", _host(m.group("url"))))
        return "[image removed]"

    def link(m: re.Match[str]) -> str:
        if _allowed(m.group("url"), allowed):
            return m.group(0)
        findings.append(LinkFinding("link", _host(m.group("url"))))
        return f"{m.group('text')} (link removed)"

    text = _MD_IMAGE.sub(image, text)
    text = _HTML_IMG.sub(image, text)
    text = _MD_LINK.sub(link, text)
    return text, findings
