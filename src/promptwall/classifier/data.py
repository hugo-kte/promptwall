"""Public prompt-injection datasets, normalised to one frame: text, label, source, split.

| source   | dataset (Hugging Face)                   | licence     | role                         |
|----------|------------------------------------------|-------------|------------------------------|
| spml     | reshabhs/SPML_Chatbot_Prompt_Injection   | MIT         | main training source         |
| deepset  | deepset/prompt-injections                | Apache-2.0  | other source (EN/DE)         |
| jackhhao | jackhhao/jailbreak-classification        | Apache-2.0  | other source (jailbreaks)    |
| dolly    | databricks/databricks-dolly-15k          | CC BY-SA 3.0| benign prompts, FPR estimate |

SPML is split by *system prompt*: every injection written against a given chatbot lands in
the same split, so the test set measures generalisation to chatbots never seen in training.
Exact duplicate texts are removed from evaluation splits when they also occur in training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

RAW = Path("data/raw")
COLUMNS = ["text", "label", "source", "split"]


def _read(pattern: str, root: Path) -> dict[str, pd.DataFrame]:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{root / pattern} not found: run `promptwall download` first")
    return {f.name.split("-")[1]: pd.read_parquet(f) for f in files}


def load_spml(root: Path = RAW, seed: int = 0) -> pd.DataFrame:
    df = _read("reshabhs__SPML_Chatbot_Prompt_Injection/*.parquet", root)["train"]
    df = pd.DataFrame(
        {
            "text": df["User Prompt"].astype(str),
            "label": df["Prompt injection"].astype(int),
            "group": df["System Prompt"].astype(str),
        }
    )
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val, test = next(outer.split(df, groups=df["group"]))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    tv = df.iloc[train_val]
    train, val = next(inner.split(tv, groups=tv["group"]))
    split = np.empty(len(df), dtype=object)
    split[tv.index[train]] = "train"
    split[tv.index[val]] = "val"
    split[test] = "test"
    return df.assign(source="spml", split=split)[COLUMNS]


def _given_splits(
    frames: dict[str, pd.DataFrame], text: str, label, source: str, seed: int
) -> pd.DataFrame:
    parts = []
    train = frames["train"]
    tr, va = train_test_split(train, test_size=0.2, random_state=seed, stratify=label(train))
    for name, part in (("train", tr), ("val", va), ("test", frames["test"])):
        parts.append(
            pd.DataFrame(
                {
                    "text": part[text].astype(str),
                    "label": label(part),
                    "source": source,
                    "split": name,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def load_deepset(root: Path = RAW, seed: int = 0) -> pd.DataFrame:
    frames = _read("deepset__prompt-injections/*.parquet", root)
    return _given_splits(
        frames, "text", lambda d: d["label"].astype(int).to_numpy(), "deepset", seed
    )


def load_jackhhao(root: Path = RAW, seed: int = 0) -> pd.DataFrame:
    frames = _read("jackhhao__jailbreak-classification/*.parquet", root)
    return _given_splits(
        frames,
        "prompt",
        lambda d: (d["type"] == "jailbreak").astype(int).to_numpy(),
        "jackhhao",
        seed,
    )


def load_dolly(root: Path = RAW, seed: int = 0) -> pd.DataFrame:
    df = _read("databricks__databricks-dolly-15k/*.parquet", root)["train"]
    # instruction + optional reference text: long benign inputs, like RAG prompts
    text = np.where(
        df["context"].str.len() > 0, df["instruction"] + "\n\n" + df["context"], df["instruction"]
    )
    out = pd.DataFrame({"text": text.astype(str), "label": 0, "source": "dolly"})
    rng = np.random.default_rng(seed).random(len(out))
    out["split"] = np.select([rng < 0.6, rng < 0.8], ["train", "val"], "test")
    return out[COLUMNS]


LOADERS = {
    "spml": load_spml,
    "deepset": load_deepset,
    "jackhhao": load_jackhhao,
    "dolly": load_dolly,
}


def load_all(root: Path = RAW, seed: int = 0) -> pd.DataFrame:
    df = pd.concat([load(root, seed) for load in LOADERS.values()], ignore_index=True)
    df = df[df["text"].str.strip().str.len() > 0]
    # an evaluation text that also appears in training would measure memorisation
    seen = set(df.loc[df["split"] == "train", "text"])
    leak = (df["split"] != "train") & df["text"].isin(seen)
    return df.loc[~leak].drop_duplicates(subset=["text", "split"]).reset_index(drop=True)
