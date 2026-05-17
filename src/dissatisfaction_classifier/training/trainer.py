import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, Trainer, TrainingArguments

logger = logging.getLogger(__name__)


def compute_class_weights(train_labels: list[int], device: torch.device) -> torch.Tensor:
    """Compute balanced class weights from training split label distribution."""
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
    tensor = torch.tensor(weights, dtype=torch.float).to(device)
    logger.info("Class weights — negative: %.4f, positive: %.4f", weights[0], weights[1])
    return tensor


class WeightedLossTrainer(Trainer):
    """HuggingFace Trainer subclass with class-weighted CrossEntropyLoss.

    Pass class_weights (torch.Tensor on the correct device) at construction via
    model_init or by setting trainer.class_weights after construction.
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT with LoRA")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.training_config)
    data_cfg = OmegaConf.load(args.data_config)

    from dissatisfaction_classifier.data.dataset import DissatisfactionDataset
    from dissatisfaction_classifier.evaluation.metrics import compute_metrics
    from dissatisfaction_classifier.models.backbone import load_backbone
    from dissatisfaction_classifier.models.lora_wrapper import apply_lora

    processed_dir = data_cfg.data.processed_dir
    train_df = pd.read_parquet(f"{processed_dir}/train.parquet")
    val_df = pd.read_parquet(f"{processed_dir}/val.parquet")
    logger.info("Train: %d rows | Val: %d rows", len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model.backbone)
    train_ds = DissatisfactionDataset(train_df, tokenizer, model_cfg.model.max_length)
    val_ds = DissatisfactionDataset(val_df, tokenizer, model_cfg.model.max_length)

    base_model = load_backbone(model_cfg)
    model = apply_lora(base_model, model_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weights(train_df["label"].tolist(), device)

    tc = train_cfg.training
    training_args = TrainingArguments(
        output_dir=tc.output_dir,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        evaluation_strategy=tc.evaluation_strategy,
        save_strategy=tc.save_strategy,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        fp16=tc.fp16,
        seed=tc.seed,
        logging_steps=tc.logging_steps,
        report_to=tc.report_to,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    logger.info("Starting training — %d epochs, lr=%.2e", tc.num_train_epochs, tc.learning_rate)
    trainer.train()

    best_dir = f"{tc.output_dir}/best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info("Best model saved to %s", best_dir)


if __name__ == "__main__":
    main()
