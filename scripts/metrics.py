from typing import Any

import numpy as np
from sklearn import metrics

from .constants import SINGLE_SENTENCE_TASKS


def predictions_from_logits(task: str, logits: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if task in SINGLE_SENTENCE_TASKS:
        probabilities = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        predictions = (probabilities >= threshold).astype(int)
        return predictions, probabilities

    probabilities = softmax(logits)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities


def compute_metrics(task: str, labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "accuracy": metrics.accuracy_score(labels, predictions),
    }

    if task in SINGLE_SENTENCE_TASKS:
        result.update(
            {
                "f1": metrics.f1_score(labels, predictions, zero_division=0),
                "precision": metrics.precision_score(labels, predictions, zero_division=0),
                "recall": metrics.recall_score(labels, predictions, zero_division=0),
            }
        )
    else:
        result.update(
            {
                "f1_micro": metrics.f1_score(labels, predictions, average="micro", zero_division=0),
                "f1_macro": metrics.f1_score(labels, predictions, average="macro", zero_division=0),
                "f1_weighted": metrics.f1_score(labels, predictions, average="weighted", zero_division=0),
                "quadratic_weighted_kappa": metrics.cohen_kappa_score(labels, predictions, weights="quadratic"),
            }
        )

    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

