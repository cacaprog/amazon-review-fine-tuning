# Contract: Dataset Interface

**Module**: `src/dissatisfaction_classifier/data/`

---

## `DissatisfactionDataset`

```python
class DissatisfactionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,        # columns: [text, label]
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,   # from config.model.max_length
    ) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict:
        # Returns:
        # {
        #   "input_ids":      torch.LongTensor of shape (max_length,)
        #   "attention_mask": torch.LongTensor of shape (max_length,)
        #   "labels":         torch.LongTensor scalar (0 or 1)
        # }
        ...
```

**Invariants**:
- `df` MUST have columns `text` (str) and `label` (int ∈ {0,1}) after `validate_reviews_df()` passes.
- `max_length` MUST match the tokenizer's maximum sequence length.
- Padding strategy: `padding='max_length'`, `truncation=True`.
- Returns `"labels"` (plural) to match HuggingFace `Trainer` expected key.

---

## `validate_reviews_df`

```python
def validate_reviews_df(
    df: pd.DataFrame,
    fail_on_invalid: bool = True,   # from config.data.validation.fail_on_invalid
) -> pd.DataFrame:
    """
    Validates df against the ProcessedReview pandera schema.

    If fail_on_invalid=True:  raises ValueError with full failure case report on any violation.
    If fail_on_invalid=False: logs violations and returns df with invalid rows removed.

    Schema enforces:
    - review_id: unique, non-null str
    - star_rating: int in {1, 2, 4, 5}
    - verified_purchase: str == "Y"
    - text: str, len(text.split()) >= 10
    - label: int in {0, 1}
    - split: str in {"train", "val", "test"}
    """
```

**Invariants**:
- MUST be called before any `DissatisfactionDataset` instantiation.
- MUST use `lazy=True` pandera validation to surface all errors in a single report.
- Return value is always a valid DataFrame (raises or filters, never returns invalid rows).
