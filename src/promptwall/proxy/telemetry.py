"""Audit log, Prometheus metrics and per-client rate limiting."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from promptwall.scanners.base import Finding


def client_id(authorization: str | None, ip: str | None) -> str:
    """Stable pseudonymous client identifier (the raw key or IP is never logged)."""
    raw = authorization or ip or "anonymous"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class AuditLog:
    """One JSON line per request. Stores decisions and scores, not prompts."""

    def __init__(self, path: str | Path, store_text: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store_text = store_text
        self._lock = threading.Lock()

    def write(
        self,
        request_id: str,
        client: str,
        model: str,
        action: str,
        findings: list[Finding],
        guard_ms: float,
        upstream_ms: float | None,
        status: int,
        text: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "client": client,
            "model": model,
            "action": action,
            "status": status,
            "findings": [f.to_dict() for f in findings],
            "guard_ms": round(guard_ms, 2),
            "upstream_ms": None if upstream_ms is None else round(upstream_ms, 1),
        }
        if self.store_text and text is not None:
            record["text"] = text  # already pseudonymised by the guard
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


class Metrics:
    """Minimal Prometheus exposition: counters and a guard-latency histogram."""

    BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[str, int] = defaultdict(int)
        self.findings: dict[tuple[str, str], int] = defaultdict(int)
        self.latency_buckets = [0] * len(self.BUCKETS)
        self.latency_sum = 0.0
        self.latency_count = 0

    def observe(self, action: str, findings: list[Finding], guard_seconds: float) -> None:
        with self._lock:
            self.requests[action] += 1
            for f in findings:
                self.findings[(f.scanner, f.kind.split(":")[0])] += 1
            self.latency_sum += guard_seconds
            self.latency_count += 1
            for i, bound in enumerate(self.BUCKETS):
                if guard_seconds <= bound:
                    self.latency_buckets[i] += 1

    def render(self) -> str:
        lines = ["# TYPE promptwall_requests_total counter"]
        with self._lock:
            lines += [
                f'promptwall_requests_total{{action="{a}"}} {n}'
                for a, n in sorted(self.requests.items())
            ]
            lines.append("# TYPE promptwall_findings_total counter")
            lines += [
                f'promptwall_findings_total{{scanner="{s}",kind="{k}"}} {n}'
                for (s, k), n in sorted(self.findings.items())
            ]
            lines.append("# TYPE promptwall_guard_seconds histogram")
            for bound, n in zip(self.BUCKETS, self.latency_buckets, strict=True):
                lines.append(f'promptwall_guard_seconds_bucket{{le="{bound}"}} {n}')
            lines.append(f'promptwall_guard_seconds_bucket{{le="+Inf"}} {self.latency_count}')
            lines.append(f"promptwall_guard_seconds_sum {self.latency_sum:.6f}")
            lines.append(f"promptwall_guard_seconds_count {self.latency_count}")
        return "\n".join(lines) + "\n"


class RateLimiter:
    """Token bucket per client: ``burst`` requests at once, refilled at ``per_minute``."""

    def __init__(self, per_minute: float, burst: int) -> None:
        self.rate = per_minute / 60.0
        self.burst = float(burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(client, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            allowed = tokens >= 1.0
            self._buckets[client] = (tokens - 1.0 if allowed else tokens, now)
            return allowed
