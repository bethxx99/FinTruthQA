from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .constants import ID_TO_LABEL, SENTENCE_PAIR_TASKS, SINGLE_SENTENCE_TASKS


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class TextPairDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        task: str,
        model_name: str,
        max_length: int,
        question_col: str = "QUES",
        answer_col: str = "ANS",
        with_labels: bool = True,
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.task = task
        self.question_col = question_col
        self.answer_col = answer_col
        self.with_labels = with_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[index]
        question = str(row[self.question_col])
        answer = str(row[self.answer_col])

        if self.task in SINGLE_SENTENCE_TASKS:
            encoded = self.tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                question,
                answer,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if "token_type_ids" not in item:
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])

        if self.with_labels:
            item["labels"] = torch.tensor(row[self.task])
        return item


def load_data(data_path: str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    return pd.read_csv(data_path, usecols=columns)


def normalize_labels(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert released FinTruthQA labels to model ids."""
    if task not in df.columns:
        return df.copy(), {"has_labels": False, "id_to_label": ID_TO_LABEL[task]}

    normalized = df.copy()
    if task in SINGLE_SENTENCE_TASKS:
        normalized[task] = normalized[task].map(normalize_binary_label).astype(int)
    elif task in SENTENCE_PAIR_TASKS:
        normalized[task] = normalized[task].astype(int) - 1
    else:
        raise ValueError(f"Unsupported task: {task}")

    validate_label_ids(normalized[task], task)
    return normalized, {"has_labels": True, "id_to_label": ID_TO_LABEL[task]}


def normalize_binary_label(value: Any) -> int:
    value_clean = str(value).strip().lower()
    if value_clean == "positive":
        return 1
    if value_clean == "negative":
        return 0
    raise ValueError(f"Expected Positive/Negative label, got: {value}")


def validate_label_ids(labels: pd.Series, task: str) -> None:
    expected = set(ID_TO_LABEL[task].keys())
    actual = set(labels.dropna().astype(int).unique().tolist())
    if not actual.issubset(expected):
        raise ValueError(
            f"Unexpected label ids for {task}: {sorted(actual)}; "
            f"expected subset of {sorted(expected)}"
        )


def split_dataframe(
    df: pd.DataFrame,
    train_split: float = 2 / 3,
    validation_split: float = 1 / 6,
    seed: int = 123,
) -> DataSplits:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))

    train_end = int(len(df) * train_split)
    validation_end = int(len(df) * (train_split + validation_split))

    train = df.iloc[indices[:train_end]].reset_index(drop=True)
    validation = df.iloc[indices[train_end:validation_end]].reset_index(drop=True)
    test = df.iloc[indices[validation_end:]].reset_index(drop=True)
    return DataSplits(train=train, validation=validation, test=test)
