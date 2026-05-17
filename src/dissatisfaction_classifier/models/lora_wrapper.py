import logging

from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Attention projection layer names per backbone family.
# Keys are matched against config.model.backbone as a prefix/substring.
LORA_TARGET_MODULES: dict[str, list[str]] = {
    "distilbert": ["q_lin", "v_lin"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "deberta": ["query_proj", "value_proj"],
}


def _resolve_target_modules(backbone_name: str) -> list[str]:
    backbone_lower = backbone_name.lower()
    for family, modules in LORA_TARGET_MODULES.items():
        if family in backbone_lower:
            return modules
    # Auto-detect by scanning for common attention projection names
    logger.warning(
        "Unknown backbone family for '%s'. Falling back to ['query', 'value'].",
        backbone_name,
    )
    return ["query", "value"]


def apply_lora(
    model: AutoModelForSequenceClassification, config: DictConfig
) -> "peft.PeftModel":
    """Wrap backbone with LoRA adapters via peft.get_peft_model().

    Resolves target_modules from backbone family lookup in LORA_TARGET_MODULES.
    Logs trainable parameter count and percentage of total.
    """
    target_modules = _resolve_target_modules(config.model.backbone)

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    pct = 100 * trainable / total

    logger.info(
        "LoRA applied — trainable: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )

    return peft_model
