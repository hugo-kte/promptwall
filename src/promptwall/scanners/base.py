"""Shared types for scanners: what was found, and what to do about it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Action(StrEnum):
    ALLOW = "allow"
    MONITOR = "monitor"  # record the finding, change nothing
    REDACT = "redact"  # rewrite the offending span, let the request through
    BLOCK = "block"  # stop the request or response

    @property
    def severity(self) -> int:
        return {"allow": 0, "monitor": 1, "redact": 2, "block": 3}[self.value]


@dataclass(frozen=True)
class Finding:
    scanner: str
    kind: str
    action: Action
    score: float | None = None
    start: int | None = None
    end: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "scanner": self.scanner,
            "kind": self.kind,
            "action": self.action.value,
        }
        if self.score is not None:
            out["score"] = round(self.score, 4)
        if self.detail:
            out["detail"] = self.detail
        return out
