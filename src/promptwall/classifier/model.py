"""Prompt-injection classifier: TF-IDF word + character n-grams, logistic regression.

A linear model on n-grams is fast (well under a millisecond per prompt), needs no GPU and its
decisions can be inspected feature by feature, which matters for a security control that
blocks traffic. The alert threshold is calibrated on benign validation prompts to a fixed
false-positive budget rather than left at 0.5.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

Features = Literal["word", "char", "word+char"]
MANIFEST, MODEL_FILE = "manifest.json", "model.joblib"


def build(features: Features = "word+char", C: float = 4.0) -> Pipeline:
    parts = []
    if "word" in features:
        parts.append(
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=200_000,
                ),
            )
        )
    if "char" in features:
        parts.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    max_features=300_000,
                ),
            )
        )
    return Pipeline(
        [
            ("features", FeatureUnion(parts)),
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000)),
        ]
    )


def calibrate_threshold(benign_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold such that at most ``target_fpr`` of benign prompts score above it."""
    return float(np.quantile(benign_scores, 1.0 - target_fpr, method="higher"))


def calibrate_worst_case(
    benign_scores_by_source: dict[str, np.ndarray], target_fpr: float
) -> float:
    """Threshold that meets the budget on *every* benign source, not just on their pool.

    A pooled quantile is dominated by the largest, easiest source (generic instructions) and
    can blow the budget on the traffic that matters (in-context chatbot questions).
    """
    return max(
        calibrate_threshold(s, target_fpr) for s in benign_scores_by_source.values() if len(s)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class InjectionModel:
    pipeline: Pipeline
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def score(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, 1]

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_FILE
        joblib.dump(self.pipeline, path, compress=3)
        manifest = {"threshold": self.threshold, "sha256": _sha256(path), "metadata": self.metadata}
        (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> InjectionModel:
        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST).read_text())
        path = directory / MODEL_FILE
        if _sha256(path) != manifest["sha256"]:
            raise ValueError(f"checksum mismatch for {path}: refusing to unpickle it")
        return cls(joblib.load(path), manifest["threshold"], manifest["metadata"])
