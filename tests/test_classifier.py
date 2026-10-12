"""Classifier mechanics on a tiny synthetic corpus (no dataset download in CI).

The positive class is marked by artificial tokens: these tests check the training,
calibration and persistence plumbing, not detection quality (see reports/ for that).
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from promptwall.classifier.data import load_all
from promptwall.classifier.evaluate import run_experiment, set_metrics
from promptwall.classifier.model import (
    InjectionModel,
    build,
    calibrate_threshold,
    calibrate_worst_case,
)

WORDS = [
    "router",
    "wifi",
    "order",
    "refund",
    "shipping",
    "warranty",
    "mesh",
    "password",
    "reset",
    "invoice",
    "account",
]


def corpus(n: int = 400, seed: int = 0) -> tuple[list[str], np.ndarray]:
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        words = rng.choices(WORDS, k=8)
        if i % 2:
            words.insert(rng.randint(0, 8), "zqxmarker zqxpayload")
        texts.append(" ".join(words))
        labels.append(i % 2)
    return texts, np.array(labels)


def test_model_learns_and_calibrates(tmp_path):
    texts, y = corpus()
    model = InjectionModel(build().fit(texts, y), threshold=1.0)
    scores = model.score(texts)
    model.threshold = calibrate_threshold(scores[y == 0], 0.05)
    assert np.mean(scores[y == 0] >= model.threshold) <= 0.05 + 1e-9
    assert np.mean(scores[y == 1] >= model.threshold) > 0.95
    model.save(tmp_path)
    loaded = InjectionModel.load(tmp_path)
    assert loaded.threshold == model.threshold
    assert np.allclose(loaded.score(texts[:5]), scores[:5])


def test_tampered_model_is_refused(tmp_path):
    texts, y = corpus()
    InjectionModel(build().fit(texts, y), threshold=0.5).save(tmp_path)
    with (tmp_path / "model.joblib").open("ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        InjectionModel.load(tmp_path)


def test_worst_case_calibration_is_never_looser_than_any_source():
    rng = np.random.default_rng(0)
    easy, hard = rng.beta(1, 20, 5000), rng.beta(2, 5, 300)
    thr = calibrate_worst_case({"easy": easy, "hard": hard}, 0.01)
    assert np.mean(hard >= thr) <= 0.01 + 1e-9 and np.mean(easy >= thr) <= 0.01
    assert thr >= calibrate_threshold(np.r_[easy, hard], 0.01)


def test_set_metrics_handles_single_class_sets():
    m = set_metrics(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.9, 0.3]), 0.5)
    assert m["false_positive_rate"] == 0.25 and m["detection_rate"] is None and m["roc_auc"] is None


def _write(tmp_path, name, frames):
    d = tmp_path / name
    d.mkdir(parents=True)
    for split, df in frames.items():
        df.to_parquet(d / f"default-{split}-0000.parquet")


def test_loaders_and_experiment_on_fake_sources(tmp_path):
    texts, y = corpus(600)
    groups = [f"system prompt {(i // 2) % 30}" for i in range(len(texts))]  # mixed classes
    _write(
        tmp_path,
        "reshabhs__SPML_Chatbot_Prompt_Injection",
        {
            "train": pd.DataFrame(
                {
                    "System Prompt": groups,
                    "User Prompt": texts,
                    "Prompt injection": y,
                    "Degree": 0,
                    "Source": None,
                }
            )
        },
    )
    ds_texts, ds_y = corpus(260, seed=1)  # distinct texts per source (no cross-source duplicates)
    ds = pd.DataFrame({"text": ds_texts, "label": ds_y})
    _write(tmp_path, "deepset__prompt-injections", {"train": ds.iloc[:200], "test": ds.iloc[200:]})
    jb_texts, jb_y = corpus(260, seed=2)
    jb = pd.DataFrame({"prompt": jb_texts, "type": np.where(jb_y == 1, "jailbreak", "benign")})
    _write(
        tmp_path,
        "jackhhao__jailbreak-classification",
        {"train": jb.iloc[:200], "test": jb.iloc[200:]},
    )
    dolly_texts, dolly_y = corpus(300, seed=3)
    benign = [t for t, label in zip(dolly_texts, dolly_y, strict=True) if label == 0]
    _write(
        tmp_path,
        "databricks__databricks-dolly-15k",
        {
            "train": pd.DataFrame(
                {"instruction": benign, "context": "", "response": "", "category": "open_qa"}
            )
        },
    )

    data = load_all(tmp_path)
    assert set(data["source"]) == {"spml", "deepset", "jackhhao", "dolly"}
    train_texts = set(data.loc[data["split"] == "train", "text"])
    assert not (set(data.loc[data["split"] == "test", "text"]) & train_texts)  # no leakage
    spml = data[data["source"] == "spml"]
    assert spml["split"].nunique() == 3
    model, results, _ = run_experiment(data, "multi-source", target_fpr=0.05)
    assert results["spml"]["detection_rate"] > 0.9
    assert model.metadata["calibration"] == "worst-case"
