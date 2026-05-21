# Critical Dissatisfaction Early Warning

Fine-tuned transformer classifier that flags critically dissatisfied Amazon reviews (1–2★) with a calibrated risk score and token-level explanations.

**Architecture**: DistilBERT + LoRA (1.09% of parameters trained) · **Test AUC-ROC**: 0.9961 · **Test F1**: 0.9434 · **ECE**: 0.0105

---

## Results

| Model | AUC-ROC | F1 |
|---|---|---|
| Majority class baseline | 0.500 | 0.000 |
| TF-IDF + Logistic Regression | 0.991 | 0.893 |
| **DistilBERT + LoRA (this project)** | **0.996** | **0.943** |

The transformer gains +5 F1 points over the TF-IDF baseline by capturing negation scope, adversative conjunctions, and cross-sentence context that bag-of-words cannot model.

---

## Project Structure

```
amazon-review-cl/
├── configs/
│   ├── data_config.yaml        # Raw path, split dates, validation flags
│   ├── model_config.yaml       # Backbone, num_labels, max_length, LoRA params
│   └── training_config.yaml    # Epochs, LR, batch size, W&B, class weighting
│
├── data/
│   ├── etl/prepare_reviews.py  # ETL pipeline (CLI entrypoint)
│   ├── processed/              # train.parquet, val.parquet, test.parquet (gitignored)
│   └── raw/                    # Source TSV — download from Kaggle (gitignored)
│
├── src/dissatisfaction_classifier/
│   ├── data/
│   │   ├── dataset.py          # DissatisfactionDataset (torch Dataset)
│   │   ├── preprocessing.py    # build_text, assign_label, assign_split, filter_reviews
│   │   └── validation.py       # Pandera schema + validate_reviews_df
│   ├── models/
│   │   ├── backbone.py         # load_backbone(config) → PreTrainedModel
│   │   └── lora_wrapper.py     # apply_lora(model, config) → PeftModel
│   ├── training/
│   │   ├── trainer.py          # WeightedLossTrainer + main() CLI
│   │   └── callbacks.py        # Training callbacks
│   ├── evaluation/
│   │   └── metrics.py          # compute_metrics, fit_calibration, export_metrics_json
│   ├── explainability/
│   │   ├── integrated_gradients.py  # get_integrated_gradients()
│   │   └── shap_explainer.py        # explain_with_shap()
│   └── inference/
│       └── predictor.py        # DissatisfactionPredictor (single + batch)
│
├── scripts/
│   ├── evaluate.py             # Standalone evaluation script
│   └── explain.py              # Standalone explainability analysis
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_explainability.ipynb
│
├── outputs/
│   ├── checkpoints/best/       # Best checkpoint — epoch 4 (adapter + tokenizer)
│   ├── reports/metrics.json    # Final test metrics
│   ├── figures/                # Precision-recall curve, confusion matrix, reliability diagram
│   └── explainability/         # IG heatmaps HTML, SHAP plot, token comparison CSV
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_explainability.py
│   └── test_inference.py
│
├── demo/app.py                 # Gradio demo interface
├── pyproject.toml              # uv project — requires Python >=3.11,<3.12
└── STUDY_NOTES.md              # End-to-end project reference + glossary
```

---

## Quickstart

### 1. Install dependencies

```bash
uv sync --group dev
```

> Requires Python 3.11 and an NVIDIA GPU with CUDA 12.x. PyTorch 2.2.2 is sourced from the cu121 index (backward-compatible with CUDA 12.4).

### 2. Download the dataset

```bash
kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset \
    -f amazon_reviews_us_Furniture_v1_00.tsv.gz -p data/raw/
gunzip data/raw/amazon_reviews_us_Furniture_v1_00.tsv.gz
```

### 3. Run ETL

```bash
uv run python data/etl/prepare_reviews.py --config configs/data_config.yaml
```

Produces `data/processed/train.parquet`, `val.parquet`, `test.parquet` with temporal splits (train ≤ 2014, val = 2015-H1, test = 2015-H2).

### 4. Run tests

```bash
uv run pytest tests/ -v
```

All 31 tests must pass before training.

### 5. Train (~3h15m on RTX 3060)

```bash
wandb login
uv run python -m dissatisfaction_classifier.training.trainer \
    --model-config configs/model_config.yaml \
    --training-config configs/training_config.yaml \
    --data-config configs/data_config.yaml
```

### 6. Evaluate

```bash
uv run python scripts/evaluate.py
```

Outputs `outputs/reports/metrics.json` and figures.

### 7. Explainability

```bash
uv run python scripts/explain.py
```

Generates IG attribution heatmaps (HTML) and SHAP plots under `outputs/explainability/`.

### 8. Demo

```bash
uv run python demo/app.py
```

---

## Model

**Backbone**: `distilbert-base-uncased` — 6 transformer layers, 768 hidden dim, 66.9M parameters.

**Adaptation**: LoRA with `r=8`, `lora_alpha=16`, `target_modules=["q_lin", "v_lin"]` — only **739,586 parameters trained (1.09%)**.

**Class imbalance**: 16.3% dissatisfied reviews. Handled with `compute_class_weight('balanced')` applied via a custom `WeightedLossTrainer`.

**Calibration**: ECE of 0.0105 on the test set — no post-hoc calibration needed.

---

## Explainability

Two complementary methods are applied to representative examples:

- **Integrated Gradients** (captum) — token-level attribution scores normalized to [-1, 1]. Red = drives dissatisfaction, blue = drives satisfaction.
- **SHAP** (shap library) — Shapley values measuring each token's marginal contribution. Limited to ≤ 10 texts due to O(n²) cost.

---

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 3060 12GB |
| CUDA | 12.4 (cu121 wheels) |
| PyTorch | 2.2.2 |
| Training time | ~3h15m for 5 epochs |

---

## Key dependencies

| Package | Version |
|---|---|
| transformers | 4.40.1 |
| peft | 0.10.0 |
| torch | 2.2.2 |
| captum | 0.7.0 |
| shap | 0.45.0 |
| pandera | 0.19.3 |
| wandb | 0.16.6 |
| gradio | 4.27.0 |
