"""Experiments for the injection classifier.

* **single-source**: train on SPML only (+ benign Dolly), test on SPML chatbots never seen in
  training *and* on two other datasets. The gap between them is the honest number: real
  attacks will not look like the training set.
* **multi-source**: train on all three sources; test on their held-out splits. This is the
  model PromptWall ships.

Every model gets the same treatment as a production control: its threshold is calibrated on
benign validation prompts for a 1% false-positive budget, and detection is reported at that
threshold, alongside threshold-free ROC-AUC.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from promptwall.classifier.model import (
    Features,
    InjectionModel,
    build,
    calibrate_threshold,
    calibrate_worst_case,
)

log = logging.getLogger(__name__)

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "single-source": {
        "train": ["spml", "dolly"],
        # unseen sources are evaluated on all their rows
        "test": {
            "spml": ["test"],
            "deepset": ["train", "val", "test"],
            "jackhhao": ["train", "val", "test"],
            "dolly": ["test"],
        },
    },
    "multi-source": {
        "train": ["spml", "deepset", "jackhhao", "dolly"],
        "test": {"spml": ["test"], "deepset": ["test"], "jackhhao": ["test"], "dolly": ["test"]},
    },
}


def set_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    alert = scores >= threshold
    pos, neg = y == 1, y == 0
    out: dict[str, Any] = {
        "n": len(y),
        "n_injection": int(pos.sum()),
        "detection_rate": float(alert[pos].mean()) if pos.any() else None,
        "false_positive_rate": float(alert[neg].mean()) if neg.any() else None,
        "roc_auc": None,
        "pr_auc": None,
    }
    if pos.any() and neg.any():
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["pr_auc"] = float(average_precision_score(y, scores))
    return out


def run_experiment(
    data: pd.DataFrame,
    name: str,
    features: Features = "word+char",
    target_fpr: float = 0.01,
    calibration: str = "worst-case",
) -> tuple[InjectionModel, dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    spec = EXPERIMENTS[name]
    train = data[data["source"].isin(spec["train"]) & (data["split"] == "train")]
    val = data[data["source"].isin(spec["train"]) & (data["split"] == "val")]
    t0 = time.perf_counter()
    pipeline = build(features).fit(train["text"].tolist(), train["label"].to_numpy())
    fit_s = time.perf_counter() - t0
    model = InjectionModel(pipeline, threshold=1.0)
    benign = val[val["label"] == 0]
    if calibration == "pooled":
        model.threshold = calibrate_threshold(model.score(benign["text"].tolist()), target_fpr)
    else:
        by_source = {src: model.score(g["text"].tolist()) for src, g in benign.groupby("source")}
        model.threshold = calibrate_worst_case(by_source, target_fpr)

    results: dict[str, Any] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source, splits in spec["test"].items():
        part = data[(data["source"] == source) & data["split"].isin(splits)]
        y, scores = part["label"].to_numpy(), model.score(part["text"].tolist())
        results[source] = set_metrics(y, scores, model.threshold)
        curves[source] = (y, scores)
    model.metadata = {
        "experiment": name,
        "features": features,
        "target_fpr": target_fpr,
        "calibration": calibration,
        "train_rows": len(train),
        "train_sources": spec["train"],
        "fit_seconds": round(fit_s, 1),
    }
    log.info(
        "%s/%s: %s",
        name,
        features,
        {s: round(r["detection_rate"] or 0, 3) for s, r in results.items()},
    )
    return model, results, curves


def latency_ms(model: InjectionModel, texts: list[str], n: int = 500) -> dict[str, float]:
    """Single-prompt scoring latency, as seen by one proxied request."""
    sample = texts[:n]
    timings = []
    for text in sample:
        t0 = time.perf_counter()
        model.score([text])
        timings.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(timings)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
