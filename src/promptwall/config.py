"""Firewall policy, loaded from YAML and validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from promptwall.scanners.base import Action
from promptwall.scanners.pii import ALL_ENTITIES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Upstream(_Strict):
    base_url: str = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible API
    api_key_env: str | None = None  # name of the env var holding the upstream key
    timeout_s: float = 120.0


class InjectionPolicy(_Strict):
    enabled: bool = True
    action: Action = Action.BLOCK
    model_dir: Path = Path("artifacts/injection")
    threshold: float | None = None  # None: the threshold calibrated at training time
    roles: list[str] = Field(default_factory=lambda: ["user", "tool"])


class PIIPolicy(_Strict):
    enabled: bool = True
    action: Action = Action.REDACT
    entities: list[str] = Field(default_factory=lambda: list(ALL_ENTITIES))
    roles: list[str] = Field(default_factory=lambda: ["user", "tool"])


class SimplePolicy(_Strict):
    enabled: bool = True
    action: Action = Action.BLOCK
    roles: list[str] = Field(default_factory=lambda: ["user", "tool"])  # input side only


class LinksPolicy(_Strict):
    enabled: bool = True
    action: Action = Action.REDACT
    allowed_domains: list[str] = Field(default_factory=list)


class InputPolicy(_Strict):
    injection: InjectionPolicy = InjectionPolicy()
    pii: PIIPolicy = PIIPolicy()
    secrets: SimplePolicy = SimplePolicy(action=Action.BLOCK)
    max_chars: int = 50_000  # OWASP LLM10: bound the cost of a single request


class OutputPolicy(_Strict):
    canary: SimplePolicy = SimplePolicy(action=Action.BLOCK)
    secrets: SimplePolicy = SimplePolicy(action=Action.REDACT)
    pii: PIIPolicy = PIIPolicy(roles=["assistant"])
    links: LinksPolicy = LinksPolicy()


class RateLimit(_Strict):
    enabled: bool = True
    requests_per_minute: float = 60.0
    burst: int = 20


class Audit(_Strict):
    path: Path = Path("runtime/audit.jsonl")
    store_text: bool = False  # off by default: the audit log must not become a PII store
    # Only honoured when the guard rewrites user text (see Policy.audit_text_allowed).


class Policy(_Strict):
    mode: Literal["enforce", "monitor"] = "enforce"
    block_response: Literal["error", "message"] = "error"
    upstream: Upstream = Upstream()
    input: InputPolicy = InputPolicy()
    output: OutputPolicy = OutputPolicy()
    rate_limit: RateLimit = RateLimit()
    audit: Audit = Audit()

    @property
    def audit_text_allowed(self) -> bool:
        """``audit.store_text`` takes effect only if forwarded user text is always rewritten:
        enforce mode, PII redacted and secrets blocked or redacted in user messages."""
        inp = self.input
        return (
            self.audit.store_text
            and self.mode == "enforce"
            and inp.pii.enabled
            and inp.pii.action is Action.REDACT
            and "user" in inp.pii.roles
            and inp.secrets.enabled
            and inp.secrets.action in (Action.REDACT, Action.BLOCK)
            and "user" in inp.secrets.roles
        )

    @classmethod
    def load(cls, path: str | Path | None) -> Policy:
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})
