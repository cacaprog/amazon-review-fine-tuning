# Data Model: Critical Dissatisfaction Early Warning

**Phase 1 output** | **Date**: 2026-05-17

---

## Entities

### RawReview

Source row from the Kaggle Amazon US Customer Reviews TSV.

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `marketplace` | str | No | 2-letter country code |
| `customer_id` | str | No | Author aggregation key |
| `review_id` | str | No | Unique review identifier |
| `product_id` | str | No | Product identifier |
| `product_parent` | str | Yes | Product aggregation key |
| `product_title` | str | Yes | Product display name |
| `product_category` | str | No | Always "Furniture" for this dataset |
| `star_rating` | int | No | 1–5; 3★ excluded at ETL |
| `helpful_votes` | int | No | Number of helpful votes |
| `total_votes` | int | No | Total votes received |
| `vine` | str | No | "Y"/"N" — Vine program participation |
| `verified_purchase` | str | No | "Y"/"N"; only "Y" admitted to pipeline |
| `review_headline` | str | Yes | Review title |
| `review_body` | str | Yes | Review text body |
| `review_date` | date | No | Review submission date (temporal split key) |

**ETL filtering rules** (applied in this order):
1. `verified_purchase == 'Y'`
2. `star_rating != 3`
3. `len((review_headline + ". " + review_body).split()) >= 10`

---

### ProcessedReview

Output of `data/etl/prepare_reviews.py`. Stored as Parquet in `data/processed/`.

| Field | Type | Nullable | Validation |
|-------|------|----------|------------|
| `review_id` | str | No | Unique, non-null (pandera) |
| `star_rating` | int | No | `isin([1, 2, 4, 5])` |
| `verified_purchase` | str | No | `== "Y"` |
| `text` | str | No | `len(text.split()) >= 10` |
| `label` | int | No | `isin([0, 1])`; 1 = dissatisfied (1–2★) |
| `split` | str | No | `isin(["train", "val", "test"])` |
| `review_date` | date | No | Used for split assignment |
| `helpful_votes` | int | No | Used for error analysis breakdown |

**Text construction**:
```
text = review_headline + ". " + review_body
```

**Label construction**:
```
label = 1 if star_rating <= 2 else 0
```

**Split assignment** (temporal, non-overlapping):
```
split = "train"  if review_date <= 2016-12-31
split = "val"    if 2017-01-01 <= review_date <= 2017-06-30
split = "test"   if 2017-07-01 <= review_date <= 2018-12-31
```
Reviews outside [2013-01-01, 2018-12-31] are excluded from the analysis window.

---

### TokenizedSample

Output of `DissatisfactionDataset.__getitem__()`. Fed to the model forward pass.

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `input_ids` | `torch.LongTensor` | `(max_length,)` | Tokenized text; default `max_length=256` |
| `attention_mask` | `torch.LongTensor` | `(max_length,)` | 1 = real token, 0 = padding |
| `label` | `torch.LongTensor` | `()` | Scalar: 0 or 1 |

---

### ModelCheckpoint

Saved to `outputs/checkpoints/{checkpoint-epoch-N}/`. Contents managed by HuggingFace
`Trainer`.

| Artifact | Description |
|----------|-------------|
| `adapter_model.safetensors` | LoRA adapter weights (PEFT format) |
| `adapter_config.json` | LoRA configuration |
| `tokenizer.json` | Matched tokenizer |
| `training_args.bin` | Serialized `TrainingArguments` |
| `trainer_state.json` | Epoch/step/metric history |

Best checkpoint (highest validation F1) is loaded at end of training
(`load_best_model_at_end: true`).

---

### CalibrationModel

Persisted isotonic regression model. Stored at `outputs/calibration/isotonic.pkl`.

| Field | Type | Notes |
|-------|------|-------|
| `calibrator` | `sklearn.isotonic.IsotonicRegression` | Fit on validation split probabilities |
| `fit_ece` | float | ECE on validation set before calibration |
| `post_ece` | float | ECE on validation set after calibration |

Loaded by `DissatisfactionPredictor` at inference time if present; identity function used
otherwise.

---

### Prediction

Standard output of `DissatisfactionPredictor.predict()`.

| Field | Type | Notes |
|-------|------|-------|
| `risk_score` | float | Calibrated probability in [0, 1] |
| `label` | int | 0 or 1 at configured threshold (default 0.5) |
| `explanation.tokens` | `list[str]` | Tokenized input words |
| `explanation.scores` | `list[float]` | IG attribution scores normalized to [-1, 1] |
| `explanation.html` | str | Inline HTML heatmap for notebook rendering |

---

## State Transitions

```
RawReview (Kaggle TSV)
   │  ETL filter + label + split assignment
   ▼
ProcessedReview (Parquet)
   │  DissatisfactionDataset tokenization
   ▼
TokenizedSample (torch.Tensor)
   │  LoRA-fine-tuned backbone + classification head
   ▼
Logits → Sigmoid → raw risk_score
   │  IsotonicRegression (if ECE > 0.05)
   ▼
calibrated risk_score
   │  DissatisfactionPredictor.predict()
   ▼
Prediction (risk_score + label + explanation)
```

---

## Class Balance

| Split | Expected positive rate | Handling |
|-------|----------------------|---------|
| Train (2013–2016) | ~15–20% | Class weights in CrossEntropyLoss |
| Val (2017-H1) | ~15–20% | Natural imbalance preserved |
| Test (2017-H2–2018) | ~15–20% | Natural imbalance preserved |

Class weights computed via `sklearn.utils.class_weight.compute_class_weight('balanced')`
on the training split label distribution at the start of each training run.
