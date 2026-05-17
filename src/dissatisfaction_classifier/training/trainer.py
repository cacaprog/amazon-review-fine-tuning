import logging

import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer

logger = logging.getLogger(__name__)


def compute_class_weights(train_labels: list[int], device: torch.device) -> torch.Tensor:
    """Compute balanced class weights from training split label distribution."""
    weights = compute_class_weight("balanced", classes=[0, 1], y=train_labels)
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
