import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .constants import ALL_TASKS, SENTENCE_PAIR_TASKS, SINGLE_SENTENCE_TASKS, TASK_NUM_LABELS
from .data import TextPairDataset, load_data, normalize_labels, split_dataframe
from .metrics import compute_metrics, predictions_from_logits
from .model import BertClassifier
from .utils import get_device, save_json, set_seed, unwrap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BERT classifier.")
    parser.add_argument("--data-path", required=True, help="CSV file containing QES/ANS/task label columns.")
    parser.add_argument("--task", choices=ALL_TASKS, default="QUES_RELEVANCE")
    parser.add_argument("--model-name", default="bert-base-chinese", help="Hugging Face model id or local path.")
    parser.add_argument("--output-dir", default="outputs/ques_relevance")
    parser.add_argument("--question-col", default="QUES")
    parser.add_argument("--answer-col", default="ANS")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--train-split", type=float, default=2 / 3)
    parser.add_argument("--validation-split", type=float, default=1 / 6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--plot", action="store_true", help="Save training curves.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    use_fp16 = args.fp16 and device.type == "cuda"

    required_columns = [args.question_col, args.answer_col, args.task]
    df = load_data(args.data_path, columns=required_columns)
    df, label_info = normalize_labels(df, args.task)
    splits = split_dataframe(df, args.train_split, args.validation_split, args.seed)
    save_split_indices(splits, output_dir)

    train_loader = make_loader(splits.train, args, shuffle=True)
    validation_loader = make_loader(splits.validation, args, shuffle=False)
    test_loader = make_loader(splits.test, args, shuffle=False)

    model = BertClassifier(
        model_name=args.model_name,
        num_labels=TASK_NUM_LABELS[args.task],
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    criterion: nn.Module
    if args.task in SINGLE_SENTENCE_TASKS:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )
    scaler = GradScaler(enabled=use_fp16)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "validation_score": [],
        "test_score": [],
    }
    best_score = -1.0
    best_checkpoint_path = checkpoint_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            task=args.task,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_fp16=use_fp16,
        )
        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            args.task,
            args.threshold,
        )
        _, test_metrics = evaluate(model, test_loader, criterion, device, args.task, args.threshold)

        monitor_name = "accuracy" if args.task in SINGLE_SENTENCE_TASKS else "f1_macro"
        validation_score = float(validation_metrics[monitor_name])
        test_score = float(test_metrics[monitor_name])

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_score"].append(validation_score)
        history["test_score"].append(test_score)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}")
        print(f"  validation={validation_metrics}")
        print(f"  test={test_metrics}")

        if validation_score > best_score:
            best_score = validation_score
            save_checkpoint(best_checkpoint_path, model, args, epoch, validation_metrics, label_info)
            print(f"  saved new best checkpoint to {best_checkpoint_path}")

        save_json({"history": history}, output_dir / "training_history.json")

    _, final_test_metrics = evaluate(model, test_loader, criterion, device, args.task, args.threshold)
    save_json(
        {
            "best_checkpoint": str(best_checkpoint_path),
            "best_validation_score": best_score,
            "final_test_metrics": final_test_metrics,
            "args": vars(args),
            "label_info": label_info,
        },
        output_dir / "metrics.json",
    )

    if args.plot:
        plot_history(history, output_dir)


def make_loader(df: pd.DataFrame, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = TextPairDataset(
        data=df,
        task=args.task,
        model_name=args.model_name,
        max_length=args.max_length,
        question_col=args.question_col,
        answer_col=args.answer_col,
        with_labels=True,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    task: str,
    gradient_accumulation_steps: int,
    use_fp16: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc="train"), start=1):
        batch = move_batch_to_device(batch, device)
        amp_context = autocast(enabled=True) if use_fp16 else nullcontext()
        with amp_context:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            loss = compute_loss(task, criterion, logits, batch["labels"])
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if step % gradient_accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * gradient_accumulation_steps

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    threshold: float,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses: list[float] = []
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []

    for batch in tqdm(loader, desc="evaluate"):
        batch = move_batch_to_device(batch, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        loss = compute_loss(task, criterion, logits, batch["labels"])
        losses.append(loss.item())
        labels_all.append(batch["labels"].detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())

    labels = np.concatenate(labels_all)
    logits = np.concatenate(logits_all)
    predictions, _ = predictions_from_logits(task, logits, threshold)
    return float(np.mean(losses)), compute_metrics(task, labels, predictions)


def compute_loss(task: str, criterion: nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if task in SINGLE_SENTENCE_TASKS:
        return criterion(logits.squeeze(-1), labels.float())
    if task in SENTENCE_PAIR_TASKS:
        return criterion(logits, labels.long())
    raise ValueError(f"Unsupported task: {task}")


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    validation_metrics: dict[str, Any],
    label_info: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "model_name": args.model_name,
            "task": args.task,
            "max_length": args.max_length,
            "num_labels": TASK_NUM_LABELS[args.task],
            "question_col": args.question_col,
            "answer_col": args.answer_col,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "label_info": label_info,
        },
        path,
    )


def save_split_indices(splits: Any, output_dir: Path) -> None:
    for name, split in (("train", splits.train), ("validation", splits.validation), ("test", splits.test)):
        split.to_csv(output_dir / f"{name}.csv", index=False)


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Training loss")
    plt.plot(history["validation_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["validation_score"], label="Validation score")
    plt.plot(history["test_score"], label="Test score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_curve.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
