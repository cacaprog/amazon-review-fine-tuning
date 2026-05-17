import logging

from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


def load_backbone(config: DictConfig) -> AutoModelForSequenceClassification:
    """Load a HuggingFace sequence classification backbone from config.model.backbone.

    No model name is hardcoded. The backbone is resolved entirely from config at runtime.
    """
    backbone_name = config.model.backbone
    num_labels = config.model.num_labels

    logger.info("Loading backbone: %s (num_labels=%d)", backbone_name, num_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        backbone_name, num_labels=num_labels
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %s", f"{total_params:,}")

    return model
