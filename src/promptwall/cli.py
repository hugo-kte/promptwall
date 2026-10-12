"""PromptWall command-line interface."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(help="PromptWall - an LLM firewall.", no_args_is_help=True, add_completion=False)
console = Console()

DATASETS = [
    "reshabhs/SPML_Chatbot_Prompt_Injection",
    "deepset/prompt-injections",
    "jackhhao/jailbreak-classification",
    "databricks/databricks-dolly-15k",
]
PolicyOpt = Annotated[Path | None, typer.Option("--policy", "-p", help="Policy YAML.")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def download(root: Path = Path("data/raw")) -> None:
    """Download the four public datasets (~20 MB of Parquet from Hugging Face)."""
    for name in DATASETS:
        url = f"https://datasets-server.huggingface.co/parquet?dataset={name}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            files = json.load(resp)["parquet_files"]
        for f in files:
            out = (
                root / name.replace("/", "__") / f"{f['config']}-{f['split']}-{Path(f['url']).name}"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                urllib.request.urlretrieve(f["url"], out)
            console.print(f"{name}: {f['split']} -> {out}")


@app.command()
def evaluate(
    model_dir: Path = Path("artifacts/injection"),
    reports_dir: Path = Path("reports"),
    target_fpr: float = 0.01,
) -> None:
    """Run every experiment, ship the multi-source model, write reports/ (metrics, figures)."""
    from promptwall.report import build_report

    m = build_report(model_dir, reports_dir, target_fpr=target_fpr)
    table = Table(title="Detection rate / false-positive rate on test data")
    for col in ("experiment", "SPML", "deepset", "jackhhao", "Dolly FPR"):
        table.add_column(col)
    for name, exp in m["experiments"].items():
        r = exp["results"]
        cell = {
            src: f"{r[src]['detection_rate']:.1%} / {r[src]['false_positive_rate']:.1%}"
            for src in ("spml", "deepset", "jackhhao")
        }
        table.add_row(
            name,
            cell["spml"],
            cell["deepset"],
            cell["jackhhao"],
            f"{r['dolly']['false_positive_rate']:.2%}",
        )
    console.print(table)
    console.print(f"Report: [bold]{reports_dir / 'REPORT.md'}[/bold]")


@app.command()
def scan(
    text: Annotated[str, typer.Argument(help="Text to inspect, as a user message.")],
    policy: PolicyOpt = None,
) -> None:
    """Run the input scanners on one message and show what the firewall would do."""
    from promptwall.config import Policy
    from promptwall.proxy.app import load_injection_model
    from promptwall.proxy.pipeline import Blocked, Guard

    pol = Policy.load(policy)
    guard = Guard(pol, load_injection_model(pol))
    try:
        body, ctx = guard.check_request({"messages": [{"role": "user", "content": text}]})
        console.print(
            f"[green]allowed[/green] -> forwarded as: {body['messages'][-1]['content']!r}"
        )
    except Blocked as err:
        ctx = err.ctx
        console.print(f"[red]blocked[/red] by {err.finding.scanner} ({err.finding.kind})")
    for f in ctx.input_findings if ctx else []:
        console.print(f"  - {f.to_dict()}")


@app.command()
def serve(
    policy: PolicyOpt = Path("configs/policy.yaml"),
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Start the OpenAI-compatible firewall proxy."""
    import uvicorn

    from promptwall.config import Policy
    from promptwall.proxy.app import create_app, load_injection_model

    pol = Policy.load(policy)
    console.print(f"PromptWall ({pol.mode}) on http://{host}:{port}/v1 -> {pol.upstream.base_url}")
    uvicorn.run(
        create_app(pol, load_injection_model(pol)), host=host, port=port, log_level="warning"
    )


@app.command()
def demo(
    policy: PolicyOpt = Path("configs/policy.yaml"),
    model: str = "llama3.2:3b",
) -> None:
    """End-to-end: a request with personal data, through the firewall, to the real upstream."""
    import asyncio

    import httpx

    from promptwall.config import Policy
    from promptwall.proxy.app import create_app, load_injection_model

    pol = Policy.load(policy)
    seen: dict[str, str] = {}

    class Spy(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            seen["upstream"] = payload["messages"][-1]["content"]
            return await super().handle_async_request(request)

    question = (
        "Hi, I'm Claire. Please write a two-sentence reply confirming that the order "
        "confirmation will be sent to claire.martin@example.com and that we may call "
        "her on +33 6 12 34 56 78 if needed."
    )
    application = create_app(pol, load_injection_model(pol), transport=Spy())

    async def run() -> httpx.Response:
        async with (
            application.router.lifespan_context(application),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application),
                base_url="http://promptwall",
                timeout=pol.upstream.timeout_s,
            ) as client,
        ):
            return await client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": "You are a concise support assistant."},
                        {"role": "user", "content": question},
                    ],
                },
            )

    resp = asyncio.run(run())
    console.rule("user sent")
    console.print(question)
    console.rule("upstream model received")
    console.print(seen.get("upstream", "(nothing: request blocked)"))
    console.rule(
        f"user received (HTTP {resp.status_code}, action {resp.headers.get('x-promptwall-action')})"
    )
    body = resp.json()
    console.print(body["choices"][0]["message"]["content"] if resp.status_code == 200 else body)


@app.command()
def info(model_dir: Path = Path("artifacts/injection")) -> None:
    """Show the shipped model's manifest (nothing is unpickled)."""
    manifest = json.loads((model_dir / "manifest.json").read_text())
    console.print_json(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "metadata"}
            | {"metadata": {k: v for k, v in manifest["metadata"].items() if k != "dataset_sizes"}}
        )
    )


if __name__ == "__main__":
    app()
