import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import TextPairDataset, load_data, normalize_labels
from .metrics import compute_metrics, predictions_from_logits
from .model import BertClassifier
from .utils import get_device, load_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction with a trained checkpoint.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-path", default="outputs/predictions.csv")
    parser.add_argument("--model-name", default=None, help="Override model id stored in checkpoint.")
    parser.add_argument("--task", default=None, help="Override task stored in checkpoint.")
    parser.add_argument("--question-col", default=None)
    parser.add_argument("--answer-col", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)

    model_name = args.model_name or checkpoint["model_name"]
    task = args.task or checkpoint["task"]
    max_length = args.max_length or checkpoint["max_length"]
    question_col = args.question_col or checkpoint.get("question_col", "QUES")
    answer_col = args.answer_col or checkpoint.get("answer_col", "ANS")
    num_labels = checkpoint.get("num_labels", 1)

    df = load_data(args.data_path)
    model_df, label_info = normalize_labels(df, task)
    has_labels = label_info["has_labels"]

    dataset = TextPairDataset(
        data=model_df,
        task=task,
        model_name=model_name,
        max_length=max_length,
        question_col=question_col,
        answer_col=answer_col,
        with_labels=has_labels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BertClassifier(model_name=model_name, num_labels=num_labels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logits = collect_logits(model, loader, device)
    predictions, probabilities = predictions_from_logits(task, logits, threshold=args.threshold)

    output = df.copy()
    output["prediction_id"] = predictions
    id_to_label = checkpoint.get("label_info", {}).get("id_to_label") or label_info["id_to_label"]
    id_to_label = {int(key): value for key, value in id_to_label.items()}
    output["prediction_label"] = [id_to_label[int(prediction)] for prediction in predictions]
    if probabilities.ndim == 1:
        output["probability"] = probabilities
    else:
        for class_idx in range(probabilities.shape[1]):
            output[f"probability_{class_idx}"] = probabilities[:, class_idx]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    if has_labels:
        label_array = model_df[task].to_numpy()
        metrics = compute_metrics(task, label_array, predictions)
        save_json(metrics, output_path.with_suffix(".metrics.json"))
        print(metrics)

    print(f"Predictions saved to {output_path}")


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    logits_all: list[np.ndarray] = []
    for batch in tqdm(loader, desc="predict"):
        batch = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        logits_all.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_all)


if __name__ == "__main__":
    main()
