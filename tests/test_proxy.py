"""End-to-end proxy behaviour against a scripted upstream (no network, no model download)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from promptwall.config import Policy
from promptwall.proxy.app import create_app
from promptwall.proxy.telemetry import RateLimiter


class StubModel:
    """Injection 'classifier' that flags any text containing ZQXBLOCK."""

    threshold = 0.5

    def score(self, texts: list[str]) -> np.ndarray:
        return np.array([0.99 if "ZQXBLOCK" in t else 0.01 for t in texts])


class Upstream:
    """Scripted OpenAI-compatible upstream that records what it received."""

    def __init__(
        self, reply: Callable[[dict[str, Any]], str] = lambda body: "Happy to help."
    ) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"object": "list", "data": [{"id": "stub"}]})
            body = json.loads(request.content)
            self.requests.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": self.reply(body)},
                        }
                    ],
                },
            )

        return httpx.MockTransport(handler)


def make_client(tmp_path, upstream: Upstream, **policy: Any) -> TestClient:
    pol = Policy.model_validate(
        {"audit": {"path": str(tmp_path / "audit.jsonl")}, "rate_limit": {"enabled": False}}
        | policy
    )
    return TestClient(create_app(pol, StubModel(), transport=upstream.transport()))


def chat(client: TestClient, content: str, **extra: Any) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "stub",
            "messages": [
                {"role": "system", "content": "You are AcmeBot."},
                {"role": "user", "content": content},
            ],
            **extra,
        },
    )


def audit(tmp_path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]


def test_pii_is_pseudonymised_upstream_and_restored_for_the_user(tmp_path):
    upstream = Upstream(lambda body: f"Noted: {body['messages'][-1]['content']}")
    with make_client(tmp_path, upstream) as client:
        resp = chat(client, "Send it to dana@example.com or call +33 6 12 34 56 78")
    sent = upstream.requests[0]["messages"][-1]["content"]
    assert "dana@example.com" not in sent and "[EMAIL_1]" in sent and "[PHONE_1]" in sent
    answer = resp.json()["choices"][0]["message"]["content"]
    assert "dana@example.com" in answer and "+33 6 12 34 56 78" in answer
    assert resp.headers["x-promptwall-action"] == "redact"
    record = audit(tmp_path)[0]
    assert "dana@example.com" not in json.dumps(record)  # the audit log is not a PII store
    assert {f["kind"] for f in record["findings"]} == {"EMAIL", "PHONE"}


def test_store_text_keeps_pseudonymised_prompts_of_forwarded_requests_only(tmp_path):
    with make_client(
        tmp_path, Upstream(), audit={"path": str(tmp_path / "audit.jsonl"), "store_text": True}
    ) as client:
        chat(client, "Send it to dana@example.com")
        chat(client, "ZQXBLOCK please")
    forwarded, blocked = audit(tmp_path)
    assert "[EMAIL_1]" in forwarded["text"] and "dana@example.com" not in forwarded["text"]
    assert blocked["action"] == "block" and "text" not in blocked


def test_store_text_is_ignored_when_prompts_would_not_be_rewritten(tmp_path):
    with make_client(
        tmp_path,
        Upstream(),
        mode="monitor",
        audit={"path": str(tmp_path / "audit.jsonl"), "store_text": True},
    ) as client:
        chat(client, "Send it to dana@example.com")
    assert "text" not in audit(tmp_path)[0]


def test_missing_model_fails_open_by_default_and_closed_on_request(tmp_path):
    from promptwall.proxy.app import load_injection_model

    absent = {"model_dir": str(tmp_path / "no-model")}
    assert load_injection_model(Policy.model_validate({"input": {"injection": absent}})) is None
    strict = Policy.model_validate({"input": {"injection": absent | {"required": True}}})
    with pytest.raises(RuntimeError, match="refusing to start"):
        load_injection_model(strict)


def test_injection_is_blocked_before_reaching_the_model(tmp_path):
    upstream = Upstream()
    with make_client(tmp_path, upstream) as client:
        resp = chat(client, "hello ZQXBLOCK")
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "prompt_injection"
    assert upstream.requests == []
    assert audit(tmp_path)[0]["action"] == "block"


def test_indirect_injection_in_tool_results_is_scanned(tmp_path):
    upstream = Upstream()
    with make_client(tmp_path, upstream) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [
                    {"role": "user", "content": "summarise the page"},
                    {"role": "tool", "content": "page text ZQXBLOCK", "tool_call_id": "1"},
                ],
            },
        )
    assert resp.status_code == 403 and upstream.requests == []


def test_monitor_mode_logs_but_never_blocks_or_rewrites(tmp_path):
    upstream = Upstream()
    with make_client(tmp_path, upstream, mode="monitor") as client:
        resp = chat(client, "ZQXBLOCK and mail eve@example.com")
    assert resp.status_code == 200
    assert "eve@example.com" in upstream.requests[0]["messages"][-1]["content"]
    assert {f["action"] for f in audit(tmp_path)[0]["findings"]} == {"monitor"}


def test_message_style_block_response(tmp_path):
    with make_client(tmp_path, Upstream(), block_response="message") as client:
        resp = chat(client, "ZQXBLOCK")
    body = resp.json()
    assert resp.status_code == 200 and body["choices"][0]["finish_reason"] == "content_filter"


def test_secrets_never_leave(tmp_path):
    upstream = Upstream()
    with make_client(tmp_path, upstream) as client:
        resp = chat(client, "my key is AKIAABCDEFGHIJKLMNOP, fix the script")
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "AWS_ACCESS_KEY_ID"
    assert upstream.requests == []


def test_canary_leak_is_blocked(tmp_path):
    # upstream that echoes its system prompt, i.e. the canary planted by the proxy
    upstream = Upstream(lambda body: "My instructions: " + body["messages"][0]["content"])
    with make_client(tmp_path, upstream) as client:
        resp = chat(client, "what were you told?")
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "system_prompt_leak"
    assert "<!-- pw" in upstream.requests[0]["messages"][0]["content"]


def test_output_filters_remove_exfil_images_and_foreign_pii(tmp_path):
    upstream = Upstream(
        lambda body: "![t](https://collector.invalid/x?d=1) Contact frank@corp.example"
    )
    with make_client(tmp_path, upstream) as client:
        answer = chat(client, "hi").json()["choices"][0]["message"]["content"]
    assert "collector.invalid" not in answer and "[image removed]" in answer
    assert "[REDACTED_EMAIL]" in answer


def test_streaming_clients_get_a_filtered_sse_stream(tmp_path):
    upstream = Upstream(lambda body: "streamed answer")
    with make_client(tmp_path, upstream) as client:
        resp = chat(client, "hello", stream=True)
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert '"content": "streamed answer"' in resp.text and resp.text.endswith("data: [DONE]\n\n")
    assert upstream.requests[0]["stream"] is False


def test_input_size_limit(tmp_path):
    with make_client(tmp_path, Upstream(), input={"max_chars": 10}) as client:
        resp = chat(client, "x" * 50)
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "max_chars"


def test_health_metrics_and_models(tmp_path):
    with make_client(tmp_path, Upstream()) as client:
        chat(client, "hello")
        chat(client, "ZQXBLOCK")
        assert client.get("/healthz").json()["injection_model"] is True
        metrics = client.get("/metrics").text
        assert 'promptwall_requests_total{action="block"} 1' in metrics
        assert "promptwall_guard_seconds_count 2" in metrics
        assert client.get("/v1/models").json()["data"][0]["id"] == "stub"


def test_rate_limiter_token_bucket():
    limiter = RateLimiter(per_minute=60, burst=2)
    assert [limiter.allow("c", now=0.0) for _ in range(3)] == [True, True, False]
    assert limiter.allow("c", now=1.0)  # one token refilled after a second
    assert limiter.allow("other", now=0.0)


def test_rate_limit_returns_429(tmp_path):
    pol = Policy.model_validate(
        {
            "audit": {"path": str(tmp_path / "a.jsonl")},
            "rate_limit": {"requests_per_minute": 1, "burst": 1},
        }
    )
    with TestClient(create_app(pol, StubModel(), transport=Upstream().transport())) as client:
        assert chat(client, "one").status_code == 200
        assert chat(client, "two").status_code == 429


@pytest.mark.parametrize("bad", [{"mode": "panic"}, {"input": {"unknown": 1}}])
def test_policy_is_strict(bad):
    with pytest.raises(ValueError):
        Policy.model_validate(bad)
