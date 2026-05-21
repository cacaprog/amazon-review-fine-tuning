# Critical Dissatisfaction Early Warning — Study Notes

Everything learned building this project end-to-end.

---

## 1. Project Goal

Classify Amazon Furniture reviews as **critically dissatisfied** (1–2★) vs. satisfied (4–5★) using a fine-tuned transformer with LoRA. The output is a calibrated risk score [0, 1] + token-level explanations. 3★ reviews are excluded as ambiguous.

The broader thesis: can a small fine-tuned model + parameter-efficient adaptation outperform a TF-IDF bag-of-words baseline by enough to justify the added complexity?

**Answer: yes** — +5 F1 points, +0.5 AUC points, and the model captures negation scope and conditional phrasing that TF-IDF cannot.

---

## 2. Dataset

**Source**: Amazon US Customer Reviews — Furniture category (TSV, 350 MB)  
**Kaggle slug**: `cynthiarempel/amazon-us-customer-reviews-dataset`

### Actual date range (discovered by inspection)

The dataset was assumed to span 2000–2021. Actual range: **2000-03-17 → 2015-08-31**. All dates are before 2016, so the original split boundaries (train≤2016, val≤2017, test≤2018) put all 576k rows in `train` with empty val/test.

**Lesson**: always inspect `df['date'].min()` and `.max()` before writing config values. The plan's assumed date range was wrong by 6 years.

### After inspection — recalibrated splits

| Split | Boundary | Rows | % Dissatisfied |
|-------|----------|------|----------------|
| Train | ≤ 2014-12-31 | 406,119 | 16.3% |
| Val | 2015-01-01 → 2015-04-30 | 88,160 | 15.8% |
| Test | 2015-05-01 → 2015-08-31 | 81,715 | 16.3% |

Class balance is consistent across splits (good — no temporal drift in dissatisfaction rate).

### ETL pipeline (`data/etl/prepare_reviews.py`)

1. Load TSV with `pd.read_csv(..., parse_dates=['review_date'])`
2. `build_text(row)`: concatenates `review_headline + ". " + review_body`
3. `assign_label(star_rating)`: 1 if star_rating ≤ 2, else 0
4. `filter_reviews`: keeps `verified_purchase == 'Y'`, `star_rating != 3`, `len(tokens) >= 10`
5. `assign_split(review_date, config)`: temporal boundaries from config YAML
6. Pandera schema validation with `lazy=True` (collect all errors before raising)
7. Save train/val/test as Parquet

**Key insight**: text is headline + body, not body alone. This matters for IG attribution — the headline ("One Star") carries a lot of weight because the star rating is embedded in text form.

---

## 3. Architecture

### Backbone: DistilBERT

`distilbert-base-uncased` — 6 transformer layers, 768 hidden dim, 66.9M parameters.

Chosen over BERT-base because:
- 40% fewer parameters, 60% faster inference
- Retains 97% of BERT's performance on GLUE
- Fits comfortably on an RTX 3060 12GB at batch_size=16, max_length=256

### LoRA (Low-Rank Adaptation via PEFT)

Instead of fine-tuning all 66.9M parameters, LoRA adds trainable rank decomposition matrices to the attention layers only.

**Important conceptual clarification:** LoRA is **not** a "redirector" or mere "perturbation." It is a **low-rank additive adaptation**. The output of a LoRA layer is:

`h = W₀x + BAx`

The original weights `W₀` remain frozen, but the new parameters `A` and `B` **add to** the original transformation. The model gains a new, low-rank computational path that combines with pre-existing knowledge — it is not simply "pointed" at the task.

**Config**:
```yaml
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1
  bias: "none"
  task_type: "SEQ_CLS"
```

**Target modules for DistilBERT**: `["q_lin", "v_lin"]`  
(DistilBERT uses `q_lin`/`v_lin` instead of BERT's `query`/`value`)

**Result**: **739,586 trainable / 67,694,596 total = 1.09%** of parameters trained.

This is the core efficiency gain: full fine-tuning would require storing and updating 67M gradients per step. LoRA stores only ~740k.

### Why LoRA target modules differ by backbone family

```python
LORA_TARGET_MODULES = {
    "distilbert": ["q_lin", "v_lin"],
    "bert":       ["query", "value"],
    "roberta":    ["query", "value"],
    "deberta":    ["query_proj", "value_proj"],
}
```

The names reflect each model's internal attention implementation. Always check `model.named_modules()` when adding a new backbone.

---

## 4. Training

### Setup

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 5 |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| fp16 | True |
| Seed | 42 |

### Class weighting

The class imbalance (16.3% positive) was handled with `sklearn.utils.class_weight.compute_class_weight('balanced')`:
- Negative (satisfied): **0.5973**
- Positive (dissatisfied): **3.0694**

Applied via a custom `WeightedLossTrainer(Trainer)` that overrides `compute_loss` to use `nn.CrossEntropyLoss(weight=class_weights)`.

**Important**: `compute_class_weight` requires `classes` as `np.ndarray`, not a plain Python list — `classes=[0, 1]` raises `InvalidParameterError`.

### Per-epoch validation metrics

| Epoch | F1 | AUC-ROC | Precision | Recall | ECE | Brier |
|-------|-----|---------|-----------|--------|-----|-------|
| 1 | 0.9221 | 0.9947 | 0.8898 | 0.9567 | 0.0159 | 0.0200 |
| 2 | 0.9375 | 0.9950 | 0.9423 | 0.9328 | 0.0109 | 0.0160 |
| 3 | 0.9406 | 0.9960 | 0.9312 | 0.9502 | 0.0117 | 0.0158 |
| **4** | **0.9427** | 0.9957 | 0.9399 | 0.9455 | **0.0105** | **0.0151** |
| 5 | 0.9426 | 0.9957 | 0.9365 | 0.9489 | 0.0117 | 0.0154 |

**Best checkpoint: epoch 4** (`checkpoint-101532`). Epoch 5 shows no gain — the model converged. Saved to `outputs/checkpoints/best/`.

### Hardware / runtime

- GPU: NVIDIA RTX 3060 12GB, CUDA 12.4
- PyTorch: 2.2.2 (cu121 wheels — cu124 wheels don't exist for 2.2.2)
- Training time: ~3h15m for 5 epochs, ~12 it/s
- Steps per epoch: 25,383 (406,119 rows / batch 16)

---

## 5. Evaluation on Temporal Test Set

### Final test metrics (81,715 reviews, May–Aug 2015)

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.9961** |
| F1 | **0.9434** |
| Precision | 0.9426 |
| Recall | 0.9441 |
| ECE | 0.0105 |
| Optimal threshold | 0.574 |

**ECE of 0.0105** is well below the 0.05 calibration threshold → no isotonic regression calibration needed. The model's raw probabilities are already well-calibrated.

**Optimal threshold** of 0.574 (vs. default 0.5) is only marginally higher — further evidence of good calibration.

### Baseline comparison

| Model | AUC-ROC | F1 |
|-------|---------|-----|
| Majority class | 0.500 | 0.000 |
| TF-IDF + Logistic Regression | 0.991 | 0.893 |
| **DistilBERT + LoRA** | **0.996** | **0.943** |

The TF-IDF baseline is surprisingly strong (AUC 0.991) — this is typical for sentiment tasks where lexical features are highly predictive. But the transformer gains 5 F1 points by capturing:
- **Negation scope**: "not bad" vs. "bad"
- **Conditional structure**: "looks nice but falls apart"
- **Context across sentence boundaries**: the headline "One Star" boosting the body's attribution

### Qualitative Error Analysis: Where the Transformer Actually Wins

| Case Type | Example | Why TF-IDF Fails | Why Transformer Wins |
|-----------|---------|------------------|----------------------|
| Scoped negation | "Not bad at all" | Weights "bad" positively | Reads the full scope |
| Adversative conjunction | "Looks nice but falls apart" | Equal weights for both | Understands concession |
| Title as prior | "One Star. Very low quality..." | Title is just another token | Uses title as prior for body |
| Cross-sentence context | "Beautiful design. Assembly was a nightmare." | Treats sentences independently | Integrates both sentences |

### Threshold analysis

The precision-recall curve shows the model maintains >0.93 precision at recall levels up to 0.95. This means it can flag ~95% of truly dissatisfied reviews while keeping false positives to <7% — operationally viable for customer service triage.

---

## 6. Calibration

**Expected Calibration Error (ECE)** measures how well predicted probabilities match actual frequencies. A model with ECE=0.0 is perfectly calibrated (a prediction of 0.7 means 70% of those cases are truly positive).

This model's ECE of 0.0105 on both val and test means: if the model says a review has 80% dissatisfaction risk, about 79–81% of such reviews truly are dissatisfied.

**Isotonic regression** was prepared as a fallback (fit on val probabilities, saved to `outputs/calibration/isotonic.pkl`) but was not needed. The calibration output `None` means identity transform.

**Why good calibration matters**: if a downstream system uses the risk score to prioritize support tickets, a miscalibrated model (ECE > 0.1) would mis-rank cases even if its binary accuracy is high.

---

## 7. Explainability

### Two methods used

**Integrated Gradients (primary)** — captum library
- Computes how much each input token's embedding contributed to the positive class logit
- Method: interpolate inputs from a zero-embedding baseline to the actual input in `n_steps` steps, integrate gradients along the path
- Result: one score per token, normalized to [-1, 1]
- Red = drives dissatisfaction prediction; Blue = drives satisfaction prediction
- Cost: one forward+backward pass per step → fast, batchable

**SHAP (secondary)** — shap library  
- Shapley values: marginal contribution of each token to the prediction, averaged over all subsets
- Implementation: `shap.Explainer` wraps a HuggingFace `pipeline`
- Requires `model.merge_and_unload()` first (merges LoRA weights into base model for SHAP pipeline compatibility)
- Cost: O(n_tokens²) perturbations per sample → limit to ≤10 texts

### Key findings from IG on the high-confidence 1★ review

Text: *"One Star. Very low quality. Ordered two and both had dings and dents. Pore packaging and thin metal. Returned both."*

Top attributions:
```
one (0.674) → star (0.485) → very (0.286) → low (0.256) → returned (0.139)
→ dent (0.101) → quality (0.083) → packaging (0.071) → thin (0.023)
```

The headline tokens ("one", "star") dominate because the star rating is embedded in the text — the model uses it as a strong prior. Physical failure terms ("dent", "thin metal", "packaging") are secondary signals from the body.

### IG vs. SHAP token alignment issue

IG uses WordPiece tokenization (subwords: `ding`, `##s`, `dent`).  
SHAP's TextMasker uses whitespace tokenization (words: `dings`, `dents.`).

The `merge` on `token` column mostly produces NaN because the token vocabularies don't align. This is expected — comparing them at the word level (by aggregating subword attributions) would give a cleaner picture. Worth fixing if you extend this work.

### Bugs fixed in explainability

1. **`_resolve_embedding_layer`**: hardcoded paths like `base_model.distilbert.embeddings.word_embeddings` broke when PEFT added the `base_model.model.*` prefix. Fixed by searching all named modules for any ending with `"word_embeddings"`.

2. **Captum attention mask expansion**: `forward_func` received `attention_mask` of shape `[1, seq_len]` but Captum passes `n_steps` interpolated inputs simultaneously, expecting shape `[n_steps, seq_len]`. Fixed by `attention_mask.expand(embeddings.shape[0], -1)`.

---

## 8. LoRA Trade-offs and Limitations (Critical Addendum)

LoRA is not a cost-free improvement. Understanding its limitations is essential for production decisions.

### 8.1 Parameter Efficiency ≠ Inference Computational Efficiency

- Full fine-tuning: `h = Wx`
- LoRA (without merging): `h = W₀x + BAx`

The addition of `BAx` has a cost. To eliminate this latency, you must **merge** the weights after training: `W_merged = W₀ + BA`. This solves the problem but makes it impossible to swap adapters dynamically for different tasks.

**Production implication:** If you need multiple adapters (one per customer segment, one per product category), LoRA without merging incurs latency overhead. If you merge, you lose adapter swappability.

### 8.2 Fundamental Limitation: LoRA Cannot "Correct" Pre-existing Knowledge

LoRA adds a low-rank adaptation. If the base model has a severe bias (e.g., associating "cheap" with "bad" in every context), LoRA may not be sufficient to correct it. Full fine-tuning is still superior for domains very different from pre-training.

**Evidence from literature:** Xu et al. (2023) classify LoRA as an "addition-based" method, contrasting it with full fine-tuning which can reweigh all features.

### 8.3 The Rank Problem: More Capacity Is Not Always Better

Increasing `r` from 8 to 16 or 32 adds expressivity, but also increases the risk of **overfitting** — especially on small datasets. The conservative `r=8` used here was a deliberate choice, not just due to VRAM.

**Recommendation:** When increasing rank, monitor validation metrics closely. If F1 stops improving or validation loss starts increasing, you've found the sweet spot.

---

## 9. Key Engineering Decisions

### Package manager: uv (not conda)

- `uv sync --group dev` replaces `conda env create`
- PyTorch sourced from `[tool.uv.sources]` with an explicit index
- `requires-python = ">=3.11,<3.12"` — the upper bound is critical: uv resolves for all Python versions in range, and `torch==2.2.2` has no wheels for Python 3.12+
- CUDA 12.4 driver → use `cu121` index (not `cu124`): PyTorch 2.2.x only published cu118/cu121 wheels; cu124 support arrived in 2.3.0. CUDA backward compatibility means cu121 binaries run on a 12.4 driver.

### Config-driven everything

All paths, dates, hyperparameters, and model IDs live in YAML files. No hardcoded values in source code. This means reproducing any experiment is a config change + re-run, not a code change.

### Temporal splitting (not random)

Random splitting would leak future reviews into training. A model trained on 2015 reviews that is evaluated on 2013 reviews is implicitly looking at future data (in deployment, future reviews won't be available at training time).

Temporal split simulates real deployment: train on historical data, evaluate on a later slice.

### Pandera for data validation

`lazy=True` collects all violations before raising, giving a full error report instead of stopping at the first bad row. `fail_on_invalid: true` in config controls whether ETL halts or drops invalid rows.

---

## 10. Bugs Encountered and Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `uv sync` no solution for Python 3.12 | `requires-python = ">=3.11"` made uv resolve for 3.12+ where torch 2.2.2 has no wheels | Add upper bound: `<3.12` |
| All 576k rows go to train | Split dates (≤2016) were beyond the dataset's actual end (2015-08-31) | Inspect actual date range; recalibrate to ≤2014/2015-H1/2015-H2 |
| `uv sync` torch index 404 | cu124 PyTorch index has no torch 2.2.2 wheels | Switch to cu121 index (backward compatible) |
| `compute_class_weight` TypeError | `classes=[0, 1]` is a list; sklearn requires `np.ndarray` | `classes=np.array([0, 1])` |
| `isinstance(model, AutoModelForSequenceClassification)` always False | `Auto*` classes are factories, not base classes | Use `isinstance(model, PreTrainedModel)` |
| Test fixture text too short | "Falls apart..." had 9 words; schema requires ≥10 tokens | Add one word to fixture text |
| IG `_resolve_embedding_layer` fails on PEFT model | Hardcoded paths didn't account for PEFT's `base_model.model.*` prefix | Search all modules for suffix `word_embeddings` |
| IG `RuntimeError: shape invalid` in Captum | `attention_mask` shape `[1, seq_len]` not expanded to `[n_steps, seq_len]` | `attention_mask.expand(embeddings.shape[0], -1)` |
| IG convergence delta > 0.05 | `n_steps=50` is too low for a 6-layer transformer | Warning logged; use n_steps=200 for production |
| `nbconvert` timeout on evaluation notebook | Running inference on 170k rows via Jupyter kernel exceeded 600s | Write standalone Python scripts instead of using nbconvert |

---

## 11. Project Structure

```
amazon-review-cl/
├── configs/
│   ├── data_config.yaml        # Raw path, split dates, validation flags
│   ├── model_config.yaml       # Backbone, num_labels, max_length, LoRA params
│   └── training_config.yaml    # Epochs, LR, batch size, W&B, class weighting
├── data/
│   ├── etl/prepare_reviews.py  # Full ETL pipeline (CLI entrypoint)
│   └── raw/                    # Gitignored — download from Kaggle
├── src/dissatisfaction_classifier/
│   ├── data/
│   │   ├── dataset.py          # DissatisfactionDataset (torch Dataset)
│   │   ├── preprocessing.py    # build_text, assign_label, assign_split, filter_reviews
│   │   └── validation.py       # Pandera schema + validate_reviews_df
│   ├── models/
│   │   ├── backbone.py         # load_backbone(config) → PreTrainedModel
│   │   └── lora_wrapper.py     # apply_lora(model, config) → PeftModel
│   ├── training/
│   │   └── trainer.py          # WeightedLossTrainer + main() CLI
│   ├── evaluation/
│   │   └── metrics.py          # compute_metrics, fit_calibration, export_metrics_json
│   ├── explainability/
│   │   ├── integrated_gradients.py  # get_integrated_gradients()
│   │   └── shap_explainer.py        # explain_with_shap()
│   └── inference/
│       └── predictor.py        # DissatisfactionPredictor (single + batch)
├── scripts/
│   ├── evaluate.py             # Standalone evaluation (replaces nbconvert)
│   └── explain.py              # Standalone explainability analysis
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_explainability.ipynb
├── outputs/
│   ├── checkpoints/best/       # Best checkpoint (adapter + tokenizer)
│   ├── reports/metrics.json    # Final test metrics
│   ├── figures/                # Precision-recall curve, confusion matrix, reliability diagram
│   └── explainability/         # IG heatmaps HTML, SHAP plot, token comparison CSV
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_explainability.py
│   └── test_inference.py       # 31 tests, all passing
├── demo/app.py                 # Gradio interface
└── pyproject.toml              # uv project, requires-python = ">=3.11,<3.12"
```

---

## 12. How to Reproduce

```bash
# 1. Environment
uv sync --group dev

# 2. Dataset (download from Kaggle)
kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset \
    -f amazon_reviews_us_Furniture_v1_00.tsv.gz -p data/raw/
gunzip data/raw/amazon_reviews_us_Furniture_v1_00.tsv.gz

# 3. ETL
uv run python data/etl/prepare_reviews.py --config configs/data_config.yaml

# 4. Tests (must all pass before training)
uv run pytest tests/ -v

# 5. Training (~3h15m on RTX 3060)
wandb login
uv run python -m dissatisfaction_classifier.training.trainer \
    --model-config configs/model_config.yaml \
    --training-config configs/training_config.yaml \
    --data-config configs/data_config.yaml

# 6. Evaluation
uv run python scripts/evaluate.py

# 7. Explainability
uv run python scripts/explain.py

# 8. Demo
uv run python demo/app.py
```

---

## 13. Next Steps and Proposed Improvements

### Immediate (short-term)

- **Increase IG `n_steps` to 200** (reduces convergence delta from ~0.97 to ~0.05)
- **Aggregate subword attributions to word level** for proper IG vs. SHAP comparison
- **Add early stopping** (`patience=2`) — the model converged at epoch 4

### Model and Architecture

- **Increase LoRA rank**: `r=8` is conservative. Try `r=16` or `r=32` — more capacity for the task at the cost of ~2% more trainable params. But monitor for overfitting.
- **Try RoBERTa or DeBERTa**: both generally outperform DistilBERT on classification tasks. DeBERTa-v3-base is particularly strong on imbalanced sentiment. The `lora_wrapper.py` already supports them — just change `backbone` in `model_config.yaml`.
- **Longer sequences**: `max_length=256` truncates long reviews. Try 384 or 512 — trades memory/speed for potentially better coverage of long complaints.

### Training efficiency

- **Gradient accumulation**: effective batch size of 64 or 128 (4–8 accumulation steps) often improves stability. The Trainer supports `gradient_accumulation_steps` in `TrainingArguments`.
- **Early stopping**: the model converged at epoch 4. Add `EarlyStoppingCallback(early_stopping_patience=2)` to avoid running unnecessary epochs.
- **Larger batch with half the max_length**: switching to `max_length=128` roughly halves VRAM usage, allowing `batch_size=32`, which is ~2x faster per epoch.

### Explainability

- **Fix IG vs. SHAP token alignment**: aggregate WordPiece subword scores to word level before comparing with SHAP's whitespace tokens. Sum subword attributions grouped by word.
- **Increase IG n_steps to 200**: convergence delta dropped from ~0.97 to ~0.05 in tests at 200 steps. Takes ~4x longer but produces trustworthy attributions.
- **Layer-wise Relevance Propagation (LRP)**: another axiomatic attribution method worth comparing with IG, now supported in captum.

### Evaluation

- **Stratified threshold search**: instead of a single optimal threshold from F1, find the threshold that minimises a business cost function `FN_cost * FN + FP_cost * FP`. Requires an estimate of relative costs.
- **Temporal drift analysis**: slice the test set by month (May vs. June vs. July vs. August 2015) and check if metrics degrade over time — an early signal of concept drift in deployment.

### Production readiness

- **Upload adapter to HuggingFace Hub**: `hub.push_adapter: true` in `model_config.yaml`. Enables zero-clone inference via `PeftModel.from_pretrained`.
- **ONNX export**: convert the merged model to ONNX for CPU inference at 3–5x the speed of PyTorch. Useful if serving without GPU.
- **Confidence thresholding in predictor**: route reviews with risk score between 0.4 and 0.6 to a human reviewer rather than making a binary call — the model is least reliable in this range.
- **Monitoring**: track ECE over time in production. If ECE rises above 0.05, re-fit the isotonic calibrator on recent labeled data without retraining the backbone.

---

## 14. Glossary

### Models and Architectures

**DistilBERT** — A distilled (compressed) version of BERT with 6 transformer layers, 768 hidden dimensions, and 66.9M parameters. Retains ~97% of BERT's GLUE performance at 40% fewer parameters and 60% faster inference. Backbone used in this project.

**BERT** (Bidirectional Encoder Representations from Transformers) — Google's foundational transformer encoder model. Pre-trained on masked language modeling and next-sentence prediction. Uses attention module names `query` / `value`.

**LoRA** (Low-Rank Adaptation) — A parameter-efficient fine-tuning technique that injects trainable rank-decomposition matrices into frozen attention layers. **Conceptually important:** LoRA is a **low-rank additive adaptation** (`h = W₀x + BAx`), not a mere "redirector." Only ~1% of parameters are trained, drastically reducing GPU memory and compute. Configured via `r` (rank), `lora_alpha`, `lora_dropout`, and `target_modules`.

**PEFT** (Parameter-Efficient Fine-Tuning) — HuggingFace library implementing LoRA and related methods. Wraps a base model into a `PeftModel` and adds the `base_model.model.*` prefix to module paths.

**PeftModel** — The wrapper class produced by applying LoRA to a `PreTrainedModel`. Supports `merge_and_unload()` to fuse LoRA weights back into the base model for deployment. Note: merging improves inference speed but prevents dynamic adapter swapping.

**TF-IDF + Logistic Regression** — Bag-of-words baseline. TF-IDF (Term Frequency–Inverse Document Frequency) converts text to sparse vectors; logistic regression classifies them. Strong for lexical tasks; cannot model negation scope or cross-sentence context.

### LoRA Trade-offs (Critical)

| Aspect | Full Fine-Tuning | LoRA (no merge) | LoRA (merged) |
|--------|------------------|-----------------|---------------|
| Trainable params | 100% | ~1% | ~1% |
| Inference speed | Baseline | Slower (adds BAx) | Same as baseline |
| Can swap adapters? | N/A | Yes | No |
| Can correct deep biases? | Yes | Limited | Limited |

### Training Concepts

**Fine-tuning** — Adapting a pre-trained model to a downstream task by continuing training on task-specific data, either updating all weights (full fine-tuning) or a small subset (parameter-efficient fine-tuning).

**Temporal splitting** — Dividing data by time rather than randomly to simulate real deployment conditions. Prevents future data from leaking into training. This project uses train ≤ 2014, val = 2015-H1, test = 2015-H2.

**Class weighting** — Technique for imbalanced datasets: assign higher loss weight to the minority class. Computed via `sklearn.utils.class_weight.compute_class_weight('balanced')`. Requires `classes` as `np.ndarray`.

**Early stopping** — Halting training when a monitored validation metric stops improving for `patience` consecutive evaluations. Prevents overfitting and wasted compute.

### Evaluation Metrics

**F1 score** — Harmonic mean of precision and recall: `2 * (P * R) / (P + R)`. Balances both false positives and false negatives. More informative than accuracy on imbalanced datasets.

**AUC-ROC** (Area Under the Receiver Operating Characteristic Curve) — Measures the model's ability to rank positive instances above negatives across all thresholds. 1.0 = perfect, 0.5 = random. Threshold-independent.

**ECE** (Expected Calibration Error) — Measures how well predicted probabilities match observed frequencies. ECE = 0 means a predicted 70% risk corresponds to exactly 70% true dissatisfaction. Values below 0.05 are considered well-calibrated; this project achieves 0.0105.

**Optimal threshold** — The classification cutoff that maximizes F1 (or a business cost function) on the validation set. Found to be 0.574 in this project vs. the default 0.5.

### Calibration

**Calibration** — The degree to which a model's predicted probabilities reflect true empirical frequencies. A well-calibrated model that predicts 0.8 is right ~80% of the time.

**Isotonic regression** — A non-parametric monotone calibration method. Fit on validation set probabilities and true labels; maps raw scores to calibrated probabilities. Used as a fallback if ECE exceeds 0.05.

### Explainability

**Integrated Gradients (IG)** — Attribution method from the captum library. Interpolates the input from a zero-embedding baseline to the actual input in `n_steps` steps and integrates the gradients along the path. Produces one score per token, normalized to [-1, 1].

**SHAP** (SHapley Additive exPlanations) — Framework for explaining model predictions using Shapley values from cooperative game theory. O(n²) perturbations per sample — expensive; limit to ≤ 10 texts.

**`merge_and_unload()`** — PeftModel method that merges LoRA weight deltas into the base model's weight matrices, producing a standard `PreTrainedModel`. Required for SHAP pipeline compatibility.

### Attention Module Names by Backbone

| Backbone | Query module | Value module |
|----------|-------------|-------------|
| DistilBERT | `q_lin` | `v_lin` |
| BERT | `query` | `value` |
| RoBERTa | `query` | `value` |
| DeBERTa | `query_proj` | `value_proj` |

These names reflect each model's internal attention implementation and must match `lora_config.target_modules` exactly. Always verify with `model.named_modules()`.

---

## 15. Key Conceptual Corrections (Based on Literature Review)

Based on the critical review by Xu et al. (2023, arXiv:2312.12148) and the original LoRA paper (Hu et al., 2021), the following corrections were applied to the initial understanding:

| Original (incorrect) framing | Corrected framing |
|------------------------------|-------------------|
| LoRA "redirects" or "points" the model | LoRA performs **low-rank additive adaptation**: `h = W₀x + BAx` |
| LoRA has no inference cost | LoRA without merging is **slower than full fine-tuning** due to the added `BAx` term |
| Higher rank (`r`) is always better | Higher rank increases **risk of overfitting**; optimal rank is task-dependent |
| LoRA can replace full fine-tuning | LoRA **cannot correct deep biases** in the backbone; full fine-tuning is still superior for out-of-domain tasks |
| Parameter efficiency = compute efficiency | Parameter efficiency ≠ inference efficiency; merging solves this but removes adapter swappability |

---

## 16. References

- Dataset: *Amazon US Customer Reviews — Furniture* (Kaggle, 2000–2015)
- DistilBERT: Sanh et al. (2019). *DistilBERT, a distilled version of BERT*. arXiv:1910.01108
- LoRA: Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
- Critical PEFT review: Xu et al. (2023). *Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment*. arXiv:2312.12148
```
