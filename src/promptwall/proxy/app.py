"""OpenAI-compatible reverse proxy: client -> PromptWall -> upstream LLM (Ollama, vLLM...)."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from promptwall import __version__
from promptwall.config import Policy
from promptwall.proxy.pipeline import Blocked, Context, Guard, refusal_completion, to_sse
from promptwall.proxy.telemetry import AuditLog, Metrics, RateLimiter, client_id
from promptwall.scanners.base import Action

log = logging.getLogger("promptwall")


def load_injection_model(policy: Policy) -> Any:
    if not policy.input.injection.enabled:
        return None
    from promptwall.classifier.model import InjectionModel

    try:
        return InjectionModel.load(policy.input.injection.model_dir)
    except FileNotFoundError:
        log.warning(
            "no injection model in %s: injection scanning disabled (run `promptwall evaluate`)",
            policy.input.injection.model_dir,
        )
        return None


def create_app(
    policy: Policy,
    injection_model: Any = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    guard = Guard(policy, injection_model)
    audit = AuditLog(policy.audit.path, policy.audit.store_text)
    metrics = Metrics()
    limiter = RateLimiter(policy.rate_limit.requests_per_minute, policy.rate_limit.burst)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        headers = {}
        if policy.upstream.api_key_env and os.environ.get(policy.upstream.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[policy.upstream.api_key_env]}"
        async with httpx.AsyncClient(
            base_url=policy.upstream.base_url.rstrip("/") + "/",
            timeout=policy.upstream.timeout_s,
            headers=headers,
            transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="PromptWall", version=__version__, lifespan=lifespan)

    def blocked_response(body: dict[str, Any], ctx: Context, err: Blocked) -> Response:
        headers = {"X-PromptWall-Request-Id": ctx.request_id, "X-PromptWall-Action": "block"}
        if policy.block_response == "message":
            return JSONResponse(
                refusal_completion(body.get("model", ""), err.finding, ctx.request_id),
                headers=headers,
            )
        return JSONResponse(
            status_code=403,
            headers=headers,
            content={
                "error": {
                    "message": f"Request blocked by PromptWall ({err.finding.scanner})",
                    "type": "promptwall_blocked",
                    "code": err.finding.kind,
                    "request_id": ctx.request_id,
                }
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        t0 = time.perf_counter()
        client = client_id(
            request.headers.get("authorization"), request.client and request.client.host
        )
        body = await request.json()
        model = str(body.get("model", ""))
        if policy.rate_limit.enabled and not limiter.allow(client):
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "rate limit exceeded", "type": "rate_limit"}},
            )
        ctx = Context()
        upstream_ms = None
        try:
            forwarded, ctx = guard.check_request(body)
            guard_s = time.perf_counter() - t0
            forwarded["stream"] = False  # buffer: outputs must be inspected before release
            t_up = time.perf_counter()
            try:
                resp = await request.app.state.client.post("chat/completions", json=forwarded)
            except httpx.HTTPError as exc:
                log.error("upstream error: %s", exc)
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": "upstream unavailable",
                            "type": "upstream_error",
                        }
                    },
                )
            upstream_ms = (time.perf_counter() - t_up) * 1000
            if resp.status_code != 200:
                return Response(
                    resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )
            t_out = time.perf_counter()
            completion = guard.check_response(resp.json(), ctx)
            guard_s += time.perf_counter() - t_out
            action = max(
                (f.action for f in ctx.findings), default=Action.ALLOW, key=lambda a: a.severity
            ).value
            metrics.observe(action, ctx.findings, guard_s)
            audit.write(
                ctx.request_id,
                client,
                model,
                action,
                ctx.findings,
                guard_s * 1000,
                upstream_ms,
                200,
            )
            headers = {"X-PromptWall-Request-Id": ctx.request_id, "X-PromptWall-Action": action}
            if body.get("stream"):
                return Response(to_sse(completion), media_type="text/event-stream", headers=headers)
            return JSONResponse(completion, headers=headers)
        except Blocked as err:
            ctx = err.ctx or ctx
            guard_s = time.perf_counter() - t0 - (upstream_ms or 0) / 1000
            metrics.observe("block", ctx.findings or [err.finding], guard_s)
            audit.write(
                ctx.request_id,
                client,
                model,
                "block",
                ctx.findings or [err.finding],
                guard_s * 1000,
                upstream_ms,
                403,
            )
            return blocked_response(body, ctx, err)

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        resp = await request.app.state.client.get("models")
        return Response(
            resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type")
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "mode": policy.mode,
            "injection_model": injection_model is not None,
        }

    @app.get("/metrics")
    async def prometheus() -> PlainTextResponse:
        return PlainTextResponse(metrics.render())

    app.state.guard, app.state.metrics, app.state.audit = guard, metrics, audit
    return app
