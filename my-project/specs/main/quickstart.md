# Quickstart: Critical Dissatisfaction Early Warning

**Audience**: Anyone cloning the repository to reproduce results or run the demo.

**Hardware required**: NVIDIA GPU with ≥12GB VRAM (RTX 3060 reference), CUDA 12.x.
CPU-only execution is possible but training will be impractically slow.

---

## Prerequisites

- [ ] conda or mamba installed
- [ ] CUDA 12.x driver installed (`nvidia-smi` should show driver ≥525)
- [ ] Kaggle account with API token at `~/.kaggle/kaggle.json`

---

## Step 1: Clone and Create Environment

```bash
git clone https://github.com/<yourname>/critical-dissatisfaction-early-warning.git
cd critical-dissatisfaction-early-warning

conda env create -f environment.yml
conda activate dissatisfaction-classifier
```

---

## Step 2: Download the Dataset

```bash
kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset \
    -f amazon_reviews_us_Furniture_v1_00.tsv.gz \
    -p data/raw/

gunzip data/raw/amazon_reviews_us_Furniture_v1_00.tsv.gz
```

The raw file is gitignored. This step requires Kaggle credentials.

---

## Step 3: Run ETL Pipeline

```bash
python data/etl/prepare_reviews.py \
    --config configs/data_config.yaml
```

Outputs (gitignored):
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`

Validation failures halt the pipeline with a full error report. Check
`configs/data_config.yaml` if this happens (`fail_on_invalid: true` by default).

---

## Step 4: Exploratory Data Analysis (Optional)

```bash
jupyter lab notebooks/01_eda.ipynb
```

Key question: *Are there lexical signals separating extreme dissatisfaction even before
fine-tuning?*

---

## Step 5: Fine-Tune the Model

```bash
# Backbone and LoRA config: configs/model_config.yaml
# Training hyperparameters: configs/training_config.yaml
python -m dissatisfaction_classifier.training.trainer \
    --model-config configs/model_config.yaml \
    --training-config configs/training_config.yaml \
    --data-config configs/data_config.yaml
```

Training logs to W&B (requires `wandb login`). Checkpoints saved to
`outputs/checkpoints/`. Best checkpoint selected by validation F1.

Training notebook alternative:

```bash
jupyter lab notebooks/02_training.ipynb
```

---

## Step 6: Evaluate

```bash
jupyter lab notebooks/03_evaluation.ipynb
```

Key question: *At what threshold does the model become operationally useful?*

Exports `outputs/reports/metrics.json` and `outputs/calibration/isotonic.pkl`.

---

## Step 7: Explainability Analysis

```bash
jupyter lab notebooks/04_explainability.ipynb
```

Key question: *Which tokens in a furniture review most reliably signal critical
dissatisfaction?*

Requires the best checkpoint from Step 5. SHAP cells are limited to ≤10 samples
(runtime warning if exceeded).

---

## Step 8: Run the Local Demo

```bash
python demo/app.py
```

Opens a Gradio interface at `http://localhost:7860`. Paste any furniture review text
to get the dissatisfaction risk score and an IG heatmap explanation.

---

## Zero-Clone Inference (No GPU Required)

If the HuggingFace Hub adapter is uploaded (see `configs/model_config.yaml`:
`hub.push_adapter: true`):

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = PeftModel.from_pretrained(base, "<yourname>/dissatisfaction-distilbert-lora")

# Then use DissatisfactionPredictor with the loaded model
```

---

## Run Tests

```bash
pytest tests/ -v
```

All tests MUST pass before any training or inference run. Tests cover:
- `test_dataset.py` — DissatisfactionDataset and validation
- `test_model.py` — backbone loading and LoRA wrapping
- `test_explainability.py` — IG and SHAP modules
- `test_inference.py` — DissatisfactionPredictor

---

## Key Configuration Files

| File | Controls |
|------|---------|
| `configs/data_config.yaml` | Raw path, split dates, seed, validation strictness |
| `configs/model_config.yaml` | Backbone, num_labels, max_length, LoRA params, Hub upload |
| `configs/training_config.yaml` | Epochs, batch size, LR, class weighting, W&B |
