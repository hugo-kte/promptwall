"""Latency overhead of the firewall, measured in-process against an instant mock upstream."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from promptwall.config import Policy
from promptwall.proxy.app import create_app


def mock_upstream(reply: str = "Sure, here is the answer.") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": body.get("model", "mock"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return httpx.MockTransport(handler)


async def _run(app: Any, prompts: list[str]) -> list[float]:
    timings = []
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://promptwall") as client,
    ):
        for prompt in prompts:
            t0 = time.perf_counter()
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            timings.append((time.perf_counter() - t0) * 1000)
    return timings


def overhead(model: Any, prompts: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        policy = Policy.model_validate(
            {"rate_limit": {"enabled": False}, "audit": {"path": str(Path(tmp) / "audit.jsonl")}}
        )
        app = create_app(policy, model, transport=mock_upstream())
        total = np.asarray(asyncio.run(_run(app, prompts)))
        records = [
            json.loads(line) for line in (Path(tmp) / "audit.jsonl").read_text().splitlines()
        ]
    guard = np.asarray([r["guard_ms"] for r in records])
    return {
        "requests": len(prompts),
        "guard_ms": {
            "p50": float(np.percentile(guard, 50)),
            "p95": float(np.percentile(guard, 95)),
        },
        "request_ms": {
            "p50": float(np.percentile(total, 50)),
            "p95": float(np.percentile(total, 95)),
        },
    }
