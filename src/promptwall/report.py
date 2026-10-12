"""Evaluation report: classifier experiments, PII benchmark, proxy overhead, figures."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from promptwall.classifier.data import load_all
from promptwall.classifier.evaluate import latency_ms, run_experiment
from promptwall.proxy.bench import overhead
from promptwall.scanners import pii_benchmark

log = logging.getLogger(__name__)

SOURCE_LABELS = {
    "spml": "SPML (unseen chatbots)",
    "deepset": "deepset",
    "jackhhao": "jackhhao (jailbreaks)",
    "dolly": "Dolly (benign)",
}


def build_report(
    model_dir: Path = Path("artifacts/injection"),
    reports_dir: Path = Path("reports"),
    data_root: Path = Path("data/raw"),
    target_fpr: float = 0.01,
) -> dict[str, Any]:
    data = load_all(data_root)
    sizes = data.groupby(["source", "split"])["label"].agg(["size", "sum"])
    experiments: dict[str, Any] = {}
    curves: dict[str, Any] = {}

    shipped, results, shipped_curves = run_experiment(data, "multi-source", target_fpr=target_fpr)
    experiments["multi-source"] = {"threshold": shipped.threshold, "results": results}
    curves["multi-source"] = shipped_curves
    shipped.metadata["dataset_sizes"] = {f"{s}/{p}": int(n) for (s, p), n in sizes["size"].items()}
    shipped.save(model_dir)
    log.info("shipped model saved to %s (threshold %.3f)", model_dir, shipped.threshold)

    single, results, _ = run_experiment(data, "single-source", target_fpr=target_fpr)
    experiments["single-source"] = {"threshold": single.threshold, "results": results}
    pooled, results, _ = run_experiment(
        data, "multi-source", target_fpr=target_fpr, calibration="pooled"
    )
    experiments["multi-source, pooled calibration"] = {
        "threshold": pooled.threshold,
        "results": results,
    }
    for features in ("word", "char"):
        model, results, _ = run_experiment(
            data, "multi-source", features=features, target_fpr=target_fpr
        )
        experiments[f"multi-source, {features} n-grams only"] = {
            "threshold": model.threshold,
            "results": results,
        }

    benign_test = data[(data["source"] == "dolly") & (data["split"] == "test")]["text"].tolist()
    rng = np.random.default_rng(0)
    bench_prompts = [
        p + f" My email is user{i}@example.com and my phone is 06 12 34 56 {i % 90 + 10}."
        for i, p in enumerate(rng.choice(benign_test, 300, replace=False))
    ]

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target_fpr": target_fpr,
        "dataset_sizes": {
            f"{s}/{p}": {"rows": int(r["size"]), "injections": int(r["sum"])}
            for (s, p), r in sizes.iterrows()
        },
        "experiments": experiments,
        "shipped_model": shipped.metadata | {"threshold": shipped.threshold},
        "classifier_latency_ms": latency_ms(shipped, benign_test),
        "proxy_overhead": overhead(shipped, bench_prompts),
        "pii_benchmark": pii_benchmark.run(),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    from promptwall import plots

    plots.cross_source(experiments, reports_dir / "figures" / "cross_source.png")
    plots.roc(curves["multi-source"], shipped.threshold, reports_dir / "figures" / "roc.png")
    (reports_dir / "REPORT.md").write_text(render(metrics))
    return metrics


def _pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def render(m: dict[str, Any]) -> str:
    exps = m["experiments"]
    sources = ["spml", "deepset", "jackhhao", "dolly"]
    lines = [
        "# PromptWall evaluation report",
        "",
        f"_Generated {m['generated_at']} by `promptwall evaluate`._",
        "",
        "## Prompt-injection classifier",
        "",
        f"Thresholds are calibrated on benign validation prompts for a **{m['target_fpr']:.0%} "
        "false-positive budget** on every source (worst case), unless stated otherwise. "
        "Cells show detection rate / false-positive rate on the test data.",
        "",
        "| Experiment | Threshold | " + " | ".join(SOURCE_LABELS[s] for s in sources) + " |",
        "|---|---:|" + "---:|" * len(sources),
    ]
    for name, exp in exps.items():
        cells = []
        for s in sources:
            r = exp["results"].get(s)
            if r is None:
                cells.append("–")
            elif s == "dolly":
                cells.append(f"FPR {_pct(r['false_positive_rate'], 2)}")
            else:
                cells.append(f"{_pct(r['detection_rate'])} / {_pct(r['false_positive_rate'])}")
        lines.append(f"| {name} | {exp['threshold']:.3f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "In the single-source experiment, deepset and jackhhao are entirely unseen sources (all "
        "rows are used); in the other experiments their held-out test split is used.",
        "",
        "![Detection across sources](figures/cross_source.png)",
        "",
        "![ROC](figures/roc.png)",
        "",
        "## PII detector (synthetic benchmark)",
        "",
        "| Entity | Precision | Recall | Support |",
        "|---|---:|---:|---:|",
    ]
    for kind, r in m["pii_benchmark"]["per_entity"].items():
        lines.append(f"| {kind} | {_pct(r['precision'])} | {_pct(r['recall'])} | {r['support']} |")
    pb = m["pii_benchmark"]
    lines += [
        f"| **All** | **{_pct(pb['precision'], 2)}** | **{_pct(pb['recall'], 2)}** | |",
        "",
        "## Latency",
        "",
        "| Measure | p50 | p95 |",
        "|---|---:|---:|",
        f"| Classifier, one prompt | {m['classifier_latency_ms']['p50']:.2f} ms | {m['classifier_latency_ms']['p95']:.2f} ms |",  # noqa: E501
        f"| Full guard (input + output scanners) | {m['proxy_overhead']['guard_ms']['p50']:.2f} ms | {m['proxy_overhead']['guard_ms']['p95']:.2f} ms |",  # noqa: E501
        f"| Proxied request, instant mock upstream | {m['proxy_overhead']['request_ms']['p50']:.2f} ms | {m['proxy_overhead']['request_ms']['p95']:.2f} ms |",  # noqa: E501
        "",
        "## Data",
        "",
        "| Source/split | Rows | Injections |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {k} | {v['rows']:,} | {v['injections']:,} |" for k, v in m["dataset_sizes"].items()
    ]
    return "\n".join(lines) + "\n"
