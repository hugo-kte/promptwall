"""Request and response guards: the policy applied to an OpenAI-style chat payload."""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from promptwall.config import Policy
from promptwall.scanners import canary, links, pii, secrets
from promptwall.scanners.base import Action, Finding

Message = dict[str, Any]


class Blocked(Exception):
    def __init__(self, finding: Finding, ctx: Context | None = None) -> None:
        super().__init__(f"{finding.scanner}:{finding.kind}")
        self.finding = finding
        self.ctx = ctx


@dataclass
class Context:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    vault: pii.Vault = field(default_factory=pii.Vault)
    canary: str | None = None
    input_findings: list[Finding] = field(default_factory=list)
    output_findings: list[Finding] = field(default_factory=list)

    @property
    def findings(self) -> list[Finding]:
        return self.input_findings + self.output_findings


def _map_text(message: Message, fn: Callable[[str], str]) -> None:
    """Apply ``fn`` to every text part of a message (plain string or content parts)."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = fn(content)
    elif isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                part["text"] = fn(part["text"])


def _texts(message: Message) -> list[str]:
    seen: list[str] = []
    _map_text(message, lambda t: seen.append(t) or t)
    return seen


class Guard:
    def __init__(self, policy: Policy, injection_model: Any = None) -> None:
        self.policy = policy
        self.model = injection_model
        cfg = policy.input.injection
        self.threshold = (
            cfg.threshold
            if cfg.threshold is not None
            else getattr(injection_model, "threshold", 1.0)
        )

    def _action(self, action: Action) -> Action:
        return (
            Action.MONITOR
            if self.policy.mode == "monitor" and action is not Action.ALLOW
            else action
        )

    # ------------------------------------------------------------------ request
    def check_request(self, body: dict[str, Any]) -> tuple[dict[str, Any], Context]:
        """Returns the rewritten request body. Raises :class:`Blocked` (carrying the context)."""
        ctx = Context()
        try:
            return self._check_request(copy.deepcopy(body), ctx), ctx
        except Blocked as err:
            err.ctx = ctx
            raise

    def _check_request(self, body: dict[str, Any], ctx: Context) -> dict[str, Any]:
        messages: list[Message] = body.get("messages") or []
        inp = self.policy.input

        total = sum(len(t) for m in messages for t in _texts(m))
        if total > inp.max_chars:
            self._input(
                ctx, Finding("limits", "max_chars", self._action(Action.BLOCK), detail=str(total))
            )

        for message in messages:
            role = message.get("role", "")
            if inp.secrets.enabled and role in inp.secrets.roles:
                self._scan_secrets(message, self._action(inp.secrets.action), ctx.input_findings)
            if inp.pii.enabled and role in inp.pii.roles:
                self._scan_pii(message, ctx)
            if inp.injection.enabled and self.model is not None and role in inp.injection.roles:
                texts = [t for t in _texts(message) if t.strip()]
                if texts:
                    score = float(max(self.model.score(texts)))
                    if score >= self.threshold:
                        self._input(
                            ctx,
                            Finding(
                                "injection",
                                "prompt_injection",
                                self._action(inp.injection.action),
                                score=score,
                                detail=f"role={role}",
                            ),
                        )

        if self.policy.output.canary.enabled:
            ctx.canary = canary.new_canary()
            system = next((m for m in messages if m.get("role") == "system"), None)
            if system is None:
                messages.insert(0, {"role": "system", "content": canary.plant("", ctx.canary)})
            else:
                planted = canary.plant(" ".join(_texts(system)), ctx.canary)
                if isinstance(system.get("content"), list):
                    system["content"] = [{"type": "text", "text": planted}]
                else:
                    system["content"] = planted
        body["messages"] = messages
        return body

    def _input(self, ctx: Context, finding: Finding) -> None:
        ctx.input_findings.append(finding)
        if finding.action is Action.BLOCK:
            raise Blocked(finding)

    def _scan_secrets(self, message: Message, action: Action, sink: list[Finding]) -> None:
        def fn(text: str) -> str:
            found = secrets.detect(text)
            for s in found:
                sink.append(Finding("secrets", s.kind, action))
                if action is Action.BLOCK:
                    raise Blocked(sink[-1])
            return secrets.redact(text)[0] if found and action is Action.REDACT else text

        _map_text(message, fn)

    def _scan_pii(self, message: Message, ctx: Context) -> None:
        cfg = self.policy.input.pii
        action = self._action(cfg.action)

        def fn(text: str) -> str:
            if action is Action.REDACT:
                text, found = pii.pseudonymise(text, ctx.vault, cfg.entities)
            else:
                found = pii.detect(text, cfg.entities)
            for ent in found:
                ctx.input_findings.append(Finding("pii", ent.kind, action))
            if found and action is Action.BLOCK:
                raise Blocked(ctx.input_findings[-1])
            return text

        _map_text(message, fn)

    # ----------------------------------------------------------------- response
    def check_response(self, completion: dict[str, Any], ctx: Context) -> dict[str, Any]:
        """Filters every choice; raises :class:`Blocked` on a canary leak."""
        completion = copy.deepcopy(completion)
        for choice in completion.get("choices", []):
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = self._filter_output(content, ctx)
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = ctx.vault.restore(fn["arguments"])
        leak = next(
            (f for f in ctx.output_findings if f.scanner == "canary" and f.action is Action.BLOCK),
            None,
        )
        if leak is not None:
            raise Blocked(leak, ctx)
        return completion

    def _filter_output(self, text: str, ctx: Context) -> str:
        out = self.policy.output
        sink = ctx.output_findings
        if out.canary.enabled and ctx.canary:
            how = canary.leaked(text, ctx.canary)
            if how:
                sink.append(
                    Finding(
                        "canary", "system_prompt_leak", self._action(out.canary.action), detail=how
                    )
                )
        if out.secrets.enabled:
            action = self._action(out.secrets.action)
            redacted, found = secrets.redact(text)
            sink.extend(Finding("secrets", s.kind, action) for s in found)
            if found and action is Action.REDACT:
                text = redacted
        if out.pii.enabled:
            action = self._action(out.pii.action)
            redacted, found = pii.redact(text, out.pii.entities)
            sink.extend(Finding("pii", e.kind, action) for e in found)
            if found and action is Action.REDACT:
                text = redacted
        if out.links.enabled:
            action = self._action(out.links.action)
            cleaned, found = links.sanitise(text, out.links.allowed_domains)
            sink.extend(Finding("links", f"{lf.kind}:{lf.host}", action) for lf in found)
            if found and action is Action.REDACT:
                text = cleaned
        # the user's own values come back last, after output PII redaction
        return ctx.vault.restore(text)


def refusal_completion(model: str, finding: Finding, request_id: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-promptwall-{request_id}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "I can't help with that request."},
                "finish_reason": "content_filter",
            }
        ],
        "promptwall": {"blocked": True, "reason": f"{finding.scanner}:{finding.kind}"},
    }


def to_sse(completion: dict[str, Any]) -> str:
    """Re-emit a buffered completion as a single-chunk OpenAI stream."""
    chunk = {
        "id": completion.get("id", "chatcmpl-promptwall"),
        "object": "chat.completion.chunk",
        "model": completion.get("model", ""),
        "choices": [
            {
                "index": c.get("index", 0),
                "delta": {
                    "role": "assistant",
                    "content": (c.get("message") or {}).get("content", ""),
                },
                "finish_reason": c.get("finish_reason", "stop"),
            }
            for c in completion.get("choices", [])
        ],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
