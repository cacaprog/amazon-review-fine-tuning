# Research: Critical Dissatisfaction Early Warning

**Phase 0 output** | **Date**: 2026-05-17

---

## 1. LoRA Target Modules per Backbone Family

**Decision**: Resolve `target_modules` from a family lookup dict in `lora_wrapper.py`;
fall back to auto-detection by scanning `model.named_modules()` for known attention
projection layer patterns.

| Backbone Family | `target_modules` | Notes |
|----------------|-----------------|-------|
| `distilbert-*` | `['q_lin', 'v_lin']` | MultiHeadSelfAttention projection layers |
| `bert-*` | `['query', 'value']` | BertSelfAttention |
| `roberta-*` | `['query', 'value']` | RobertaSelfAttention (same names as BERT) |
| `deberta-v2-*` / `deberta-v3-*` | `['query_proj', 'value_proj']` | DisentangledSelfAttention |
| Unknown | Auto-detect | Scan `named_modules()` for layers matching `q_proj|query|q_lin` |

**Rationale**: Hardcoding per-family names in `lora_wrapper.py` covers the four most
common English sequence-classification backbones. Auto-detection handles novel backbones
without code changes.

**Alternatives considered**: PEFT's `TaskType.SEQ_CLS` auto-selection — unreliable across
backbone families; not used.

---

## 2. SHAP with Merged LoRA Weights

**Decision**: Merge LoRA adapters into the base model with `model.merge_and_unload()`
before SHAP computation. Wrap merged model in a `transformers.pipeline`, pass to
`shap.Explainer`.

**Implementation pattern**:
```python
merged = lora_model.merge_and_unload()
pipe = pipeline("text-classification", model=merged, tokenizer=tokenizer,
                return_all_scores=True, device=0)
explainer = shap.Explainer(pipe, masker=shap.maskers.Text(r"\W"))
shap_values = explainer(texts[:10])  # limit batch size — SHAP is O(n_tokens²)
```

**Rationale**: PEFT's `PeftModel` is not directly compatible with `shap.Explainer`'s
pipeline wrapping. Merging weights produces a standard HuggingFace model that the SHAP
pipeline wrapper understands correctly.

**Limitation**: SHAP token attribution runs perturbation-based computation — limit to
≤10 samples in `04_explainability.ipynb`. Document this constraint inline.

**Alternatives considered**: `shap.DeepExplainer` on raw model — requires manual forward
function wrapping and is less stable with transformer architectures; not used.

---

## 3. Calibration Strategy

**Decision**: Apply Isotonic Regression calibration on the validation split; evaluate
calibration quality on the test split with ECE and a reliability diagram.

**When to use each method**:

| Method | When | scikit-learn |
|--------|------|--------------|
| Isotonic Regression | ≥1000 calibration samples, non-monotonic miscalibration | `IsotonicRegression()` |
| Platt Scaling | <1000 samples, monotonic miscalibration | `LogisticRegression()` on logits |

**Implementation**:
```python
# Fit on validation set
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(val_probs, val_labels)
# Apply on test set
calibrated_probs = iso.transform(test_probs)
# Persist
joblib.dump(iso, 'outputs/calibration/isotonic.pkl')
```

**Trigger for calibration**: Apply when ECE (Expected Calibration Error) > 0.05 on the
validation set. If ECE ≤ 0.05, raw scores are sufficiently calibrated and the identity
function is used.

**Rationale**: The furniture review dataset has ~500k rows; the validation split
(2017-H1) will have thousands of samples, making Isotonic Regression the stronger choice.
Platt scaling is documented as the fallback.

---

## 4. Integrated Gradients with Captum

**Decision**: Attribute over the word embedding layer; use zero-embedding baseline;
resolve embedding layer from `model.base_model.embeddings.word_embeddings` for a
backbone-agnostic implementation.

**Implementation pattern**:
```python
from captum.attr import IntegratedGradients

def forward_func(embeddings):
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    return outputs.logits[:, 1]  # positive class logit

ig = IntegratedGradients(forward_func)
baseline = torch.zeros_like(input_embeddings)
attributions, delta = ig.attribute(
    input_embeddings, baseline, n_steps=50, return_convergence_delta=True
)
# Normalize per token
token_scores = attributions.sum(dim=-1).squeeze(0)
token_scores = token_scores / torch.norm(token_scores)
```

**Baseline choice**: Zero embeddings (equivalent to an uninformative reference). This
is the standard choice for transformer attribution — the all-PAD-token embedding is an
acceptable alternative if zero baseline produces artifacts.

**Backbone-agnostic embedding access**:
```python
embedding_layer = model.base_model.embeddings.word_embeddings
# For PeftModel wrapping distilbert: model.base_model.distilbert.embeddings.word_embeddings
# Resolve dynamically to avoid hardcoding:
embedding_layer = dict(model.named_modules()).get(
    'base_model.embeddings.word_embeddings',
    dict(model.named_modules()).get('base_model.distilbert.embeddings.word_embeddings')
)
```

**Rationale**: IG is chosen as primary because it is axiomatically grounded (completeness,
sensitivity, implementation invariance) and gradient-native to the model. SHAP is
secondary because it is perturbation-based and computationally expensive.

---

## 5. Pandera Schema Enforcement

**Decision**: Use `pa.DataFrameSchema` with `strict='filter'` on raw data (remove
unexpected columns) and hard validation on processed data. Use `lazy=True` to surface
all errors at once rather than failing on the first violation.

**Schema for processed DataFrame**:
```python
import pandera as pa

processed_schema = pa.DataFrameSchema(
    {
        "review_id": pa.Column(str, unique=True, nullable=False),
        "star_rating": pa.Column(int, pa.Check.isin([1, 2, 4, 5])),
        "verified_purchase": pa.Column(str, pa.Check.eq("Y")),
        "text": pa.Column(str, pa.Check(lambda s: s.str.split().str.len() >= 10,
                                        element_wise=False)),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
        "split": pa.Column(str, pa.Check.isin(["train", "val", "test"])),
    },
    strict="filter",
)

def validate_reviews_df(df, fail_on_invalid=True):
    try:
        return processed_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        if fail_on_invalid:
            raise ValueError(f"Data validation failed:\n{e.failure_cases}") from e
        invalid_idx = e.failure_cases["index"].dropna().unique()
        return df.drop(index=invalid_idx).reset_index(drop=True)
```

**Rationale**: `lazy=True` surfaces all violations in a single error report, which is
more useful for debugging ETL issues than fail-fast on the first bad row. The
`fail_on_invalid` flag mirrors the config setting and allows the same function to serve
both hard-stop (production ETL) and soft-filter (exploratory) modes.

---

## 6. Class Weight Computation

**Decision**: Compute balanced class weights from the training split label distribution
using `sklearn.utils.class_weight.compute_class_weight`, then inject into the HuggingFace
`Trainer` via a custom subclass that overrides `compute_loss`.

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=[0, 1], y=train_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# In custom Trainer:
def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
    loss = loss_fn(logits, labels)
    return (loss, outputs) if return_outputs else loss
```

**Rationale**: Natural class imbalance (~15–20% positive) means the majority-class
classifier achieves 80–85% accuracy trivially. Class weighting penalizes
false negatives (missed dissatisfaction) more heavily, which aligns with the business
cost framing (reputation damage >> unnecessary escalation).
