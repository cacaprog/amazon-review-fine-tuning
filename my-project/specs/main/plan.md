# Implementation Plan: Critical Dissatisfaction Early Warning

**Branch**: `main` | **Date**: 2026-05-17 | **Spec**: `../../project.md`

**Input**: Feature specification from `project.md` (root-level project spec)

## Summary

Fine-tune a transformer-based classifier (DistilBERT default, swappable via YAML config)
with LoRA on verified Amazon Furniture purchase reviews to produce a calibrated
dissatisfaction risk score (probability of 1–2★ extreme dissatisfaction) and a
token-level textual explanation (Integrated Gradients primary, SHAP secondary).
The full pipeline spans ETL → tokenization → LoRA fine-tuning → calibration →
explainability and is targeted at a public GitHub ML portfolio.

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**: transformers, peft, datasets, captum, shap, torch,
scikit-learn, omegaconf, pandera, pytest, black, isort, gradio, wandb

**Storage**: Parquet files (`data/processed/`), model checkpoints
(`outputs/checkpoints/`), isotonic calibration models (`outputs/calibration/`),
metrics JSON (`outputs/reports/`), figures (`outputs/figures/`)

**Testing**: pytest (unit tests for dataset, model, explainability, inference modules)

**Target Platform**: Linux / NVIDIA RTX 3060 12GB VRAM, CUDA 12.x

**Project Type**: ML research/portfolio — Python library + Jupyter notebooks (×4)
+ local Gradio demo

**Performance Goals**: Training convergence ≤5 epochs on training split; batch
inference suitable for evaluation runs; SHAP limited to ≤10 samples (documented
limitation in 04_explainability.ipynb)

**Constraints**: 12GB VRAM ceiling → fp16 training + LoRA (reduces trainable params
to ~1% of backbone); no production deployment, no containerization

**Scale/Scope**: Single machine; ~500k raw Amazon Furniture reviews spanning
2012-03-08 → 2021-12-14; analysis window 2013-2018 (see Temporal Integrity);
4 narrative notebooks + 1 Gradio demo

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Status |
|-----------|-------|--------|
| I. Reproducibility First | Seeds fixed in configs; `environment.yml` with pinned deps; `outputs/reports/` exports; Hub adapter upload | ✅ PASS |
| II. Config-Driven Architecture | All backbone names, LoRA params, data paths, training HPs declared in `configs/*.yaml`; no string literals in `src/` | ✅ PASS |
| III. Temporal Integrity | Splits temporal (train ≤2016-12-31, val 2017-H1, test 2017-H2–2018); dataset true range 2012-03-08 → 2021-12-14 declared in constitution | ✅ PASS |
| IV. Explainability as First-Class | IG + SHAP both implemented; `DissatisfactionPredictor` returns `explanation` in standard output | ✅ PASS |
| V. Test-Driven Data Quality | pandera schema enforces hard-stop on invalid data; `fail_on_invalid: true` default; pytest covers all 4 core modules | ✅ PASS |

**Result**: No violations. No Complexity Tracking entries required. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/main/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── dataset-interface.md
│   ├── model-interface.md
│   ├── explainability-interface.md
│   └── inference-interface.md
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
critical-dissatisfaction-early-warning/
│
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── data/
│   ├── raw/                             # gitignored — Kaggle TSV
│   ├── processed/                       # gitignored — train/val/test parquet
│   └── etl/
│       └── prepare_reviews.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_explainability.ipynb
│
├── src/
│   └── dissatisfaction_classifier/
│       ├── __init__.py
│       ├── data/
│       │   ├── dataset.py
│       │   ├── preprocessing.py
│       │   └── validation.py
│       ├── models/
│       │   ├── backbone.py
│       │   └── lora_wrapper.py
│       ├── training/
│       │   ├── trainer.py
│       │   └── callbacks.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── explainability/
│       │   ├── integrated_gradients.py
│       │   └── shap_explainer.py
│       └── inference/
│           └── predictor.py
│
├── demo/
│   └── app.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_explainability.py
│   └── test_inference.py
│
└── outputs/
    ├── checkpoints/                     # gitignored
    ├── figures/
    ├── reports/
    └── calibration/
```

**Structure Decision**: Single Python project under `src/dissatisfaction_classifier/`.
No backend/frontend split (no web service). Notebooks are narrative artifacts, not
runnable production code. Tests in `tests/` at repo root.
