# Contract: Model Interface

**Module**: `src/dissatisfaction_classifier/models/`

---

## `load_backbone`

```python
def load_backbone(config: DictConfig) -> AutoModelForSequenceClassification:
    """
    Loads a HuggingFace sequence classification backbone from config.model.backbone.

    Args:
        config: OmegaConf DictConfig with at minimum:
            config.model.backbone  (str) — HuggingFace model ID, e.g. "distilbert-base-uncased"
            config.model.num_labels (int) — number of output classes (2 for binary)

    Returns:
        AutoModelForSequenceClassification with classification head initialized.
        Base weights loaded from HuggingFace Hub.

    Invariants:
    - No model name or checkpoint path MAY appear as a string literal in this function.
    - MUST log the resolved backbone name and total parameter count.
    - MUST NOT apply LoRA — apply_lora() handles that.
    """
```

---

## `apply_lora`

```python
def apply_lora(
    model: AutoModelForSequenceClassification,
    config: DictConfig,
) -> PeftModel:
    """
    Wraps the backbone with LoRA adapters via peft.get_peft_model().

    Args:
        model: Base backbone from load_backbone()
        config: OmegaConf DictConfig with config.lora.{r, lora_alpha, lora_dropout,
                bias, task_type} and config.model.backbone (for family detection)

    Returns:
        PeftModel with LoRA adapters applied and base weights frozen.

    Invariants:
    - target_modules MUST be resolved from backbone family lookup, not hardcoded per call.
    - Supported families: distilbert, bert, roberta, deberta-v2/v3 (see research.md §1).
    - MUST log trainable parameter count and % of total parameters.
    - Trainable params MUST be < 5% of total params for default LoRA config.
    """
```

**Backbone family → target_modules mapping**:
```python
LORA_TARGET_MODULES = {
    "distilbert": ["q_lin", "v_lin"],
    "bert":       ["query", "value"],
    "roberta":    ["query", "value"],
    "deberta":    ["query_proj", "value_proj"],
}
```
