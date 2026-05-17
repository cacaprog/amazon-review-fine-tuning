---
description: "Task list for Critical Dissatisfaction Early Warning ML pipeline"
---

# Tasks: Critical Dissatisfaction Early Warning

**Input**: Design documents from `specs/main/`

**Note**: No spec.md exists. User stories derived from `project.md` goals (G1–G7)
and validated against constitution Principles I–V.

**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Included — required by Constitution Principle V (Test-Driven Data Quality).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1–US4)
- Exact file paths included in all implementation tasks

---

## User Stories (derived from project.md goals)

| Story | Goal | Priority | Independent Test |
|-------|------|----------|-----------------|
| US1 | Reproducible ETL Pipeline | P1 | ETL produces valid Parquet files; `pytest tests/test_dataset.py` passes |
| US2 | Config-Driven Model Training & Evaluation | P2 | Training completes; `outputs/reports/metrics.json` written; `pytest tests/test_model.py` passes |
| US3 | Explainability (IG + SHAP) | P3 | `DissatisfactionPredictor.predict()` returns `explanation`; `pytest tests/test_explainability.py` passes |
| US4 | Inference Predictor & Local Demo | P4 | `pytest tests/test_inference.py` passes; `python demo/app.py` launches at localhost:7860 |

---

## Phase 1: Setup

**Purpose**: Project initialization and configuration files

- [ ] T001 Create full directory structure per plan.md (configs/, data/raw/, data/processed/, data/etl/, notebooks/, src/dissatisfaction_classifier/, demo/, tests/, outputs/figures/, outputs/reports/, outputs/calibration/, outputs/checkpoints/)
- [ ] T002 Create `pyproject.toml` with package metadata: name=`dissatisfaction-classifier`, packages under `src/dissatisfaction_classifier`, entry points for ETL script
- [ ] T003 [P] Create `environment.yml` with pinned Python 3.11 and all dependencies: transformers, peft, datasets, captum, shap, torch (CUDA 12.x), scikit-learn, omegaconf, pandera, pytest, black, isort, gradio, wandb
- [ ] T004 [P] Create `.gitignore` ignoring `data/raw/`, `data/processed/`, `outputs/checkpoints/`, `*.pyc`, `.wandb/`, `__pycache__/`
- [ ] T005 [P] Create `configs/data_config.yaml` with raw_path, processed_dir, min_tokens=10, train_end_date=2016-12-31, val_end_date=2017-06-30, test_end_date=2018-12-31, seed=42, validation.schema_strict=true, validation.fail_on_invalid=true
- [ ] T006 [P] Create `configs/model_config.yaml` with model.backbone=distilbert-base-uncased, model.num_labels=2, model.max_length=256, lora.r=8, lora.lora_alpha=16, lora.lora_dropout=0.1, lora.bias=none, lora.task_type=SEQ_CLS, hub.push_adapter=true, hub.repo_id placeholder
- [ ] T007 [P] Create `configs/training_config.yaml` with output_dir=outputs/checkpoints, num_train_epochs=5, per_device_train_batch_size=16, per_device_eval_batch_size=32, learning_rate=2e-4, weight_decay=0.01, warmup_ratio=0.1, lr_scheduler_type=cosine, evaluation_strategy=epoch, save_strategy=epoch, load_best_model_at_end=true, metric_for_best_model=f1, fp16=true, seed=42, class_weight=balanced, logging_steps=100, report_to=wandb

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before any user story can begin

**⚠️ CRITICAL**: No user story implementation can begin until this phase is complete

- [ ] T008 Create `src/dissatisfaction_classifier/__init__.py` (empty package marker) and `src/dissatisfaction_classifier/data/__init__.py`, `models/__init__.py`, `training/__init__.py`, `evaluation/__init__.py`, `explainability/__init__.py`, `inference/__init__.py`
- [ ] T009 [P] Implement `src/dissatisfaction_classifier/data/preprocessing.py`: `build_text(row)` concatenating review_headline + ". " + review_body; `assign_label(star_rating)` returning 1 if ≤2 else 0; `assign_split(review_date, config)` returning train/val/test per data_config.yaml date boundaries; `filter_reviews(df, config)` applying verified_purchase, star_rating != 3, and min_tokens filters
- [ ] T010 [P] Implement `src/dissatisfaction_classifier/data/validation.py`: define pandera `ProcessedReview` schema (review_id unique non-null, star_rating isin[1,2,4,5], verified_purchase eq "Y", text len>=10, label isin[0,1], split isin[train/val/test]); implement `validate_reviews_df(df, fail_on_invalid=True)` using lazy=True validation per contract in `specs/main/contracts/dataset-interface.md`
- [ ] T011 Implement `src/dissatisfaction_classifier/models/backbone.py`: `load_backbone(config)` calling `AutoModelForSequenceClassification.from_pretrained(config.model.backbone, num_labels=config.model.num_labels)`; log backbone name and total parameter count; no model name string literals
- [ ] T012 Implement `src/dissatisfaction_classifier/models/lora_wrapper.py`: define `LORA_TARGET_MODULES = {"distilbert": ["q_lin","v_lin"], "bert": ["query","value"], "roberta": ["query","value"], "deberta": ["query_proj","value_proj"]}`; `apply_lora(model, config)` resolving family from config.model.backbone, calling `peft.get_peft_model()`, logging trainable param count and % per contract in `specs/main/contracts/model-interface.md`

**Checkpoint**: All five foundational modules importable; backbone loads without hardcoded names; LoRA family lookup resolves for distilbert.

---

## Phase 3: US1 — Reproducible ETL Pipeline (Priority: P1) 🎯 MVP

**Goal**: Anyone cloning the repo can go from raw Kaggle TSV to validated, temporally-split Parquet files

**Independent Test**: `python data/etl/prepare_reviews.py --config configs/data_config.yaml` produces `data/processed/{train,val,test}.parquet`; `pytest tests/test_dataset.py -v` passes all tests

### Tests for US1 ⚠️ Write FIRST — verify they FAIL before implementing

- [ ] T013 [P] [US1] Write `tests/test_dataset.py`: test `DissatisfactionDataset.__len__()` returns correct count; test `__getitem__()` returns dict with input_ids shape (256,), attention_mask shape (256,), labels scalar in {0,1}; test `validate_reviews_df()` raises ValueError on invalid rows when fail_on_invalid=True; test `validate_reviews_df()` drops invalid rows and returns clean df when fail_on_invalid=False

### Implementation for US1

- [ ] T014 [US1] Implement `src/dissatisfaction_classifier/data/dataset.py`: `DissatisfactionDataset(df, tokenizer, max_length=256)` with `__len__` and `__getitem__` returning {"input_ids", "attention_mask", "labels"} per contract in `specs/main/contracts/dataset-interface.md`; padding='max_length', truncation=True
- [ ] T015 [US1] Implement `data/etl/prepare_reviews.py`: load `configs/data_config.yaml` via omegaconf; read raw TSV; call `filter_reviews()`, `build_text()`, `assign_label()`, `assign_split()`; call `validate_reviews_df(fail_on_invalid=True)`; save each split as Parquet to `data/processed/{train,val,test}.parquet`; log row counts per split and class balance
- [ ] T016 [P] [US1] Create `notebooks/01_eda.ipynb` with sections: dataset provenance (source URL, size, temporal range), temporal distribution (monthly review volume plot), label distribution and class imbalance bar chart, verified vs unverified purchase counts with justification for filtering, star rating 1–5 histogram, text length distribution by label, 3 sample 1★ vs 3 sample 5★ reviews; conclude with key question: *Are there lexical signals separating extreme dissatisfaction even before fine-tuning?*

**Checkpoint**: ETL pipeline runs end-to-end; three Parquet files exist; `pytest tests/test_dataset.py` passes; `01_eda.ipynb` executes without errors.

---

## Phase 4: US2 — Config-Driven Model Training & Evaluation (Priority: P2)

**Goal**: DistilBERT (default, swappable via config) fine-tuned with LoRA, calibrated risk scores, full evaluation with baselines

**Independent Test**: Training completes with best checkpoint in `outputs/checkpoints/`; `outputs/reports/metrics.json` written; `pytest tests/test_model.py -v` passes; `02_training.ipynb` and `03_evaluation.ipynb` execute without errors

### Tests for US2 ⚠️ Write FIRST — verify they FAIL before implementing

- [ ] T017 [P] [US2] Write `tests/test_model.py`: test `load_backbone()` returns `AutoModelForSequenceClassification` with `num_labels=2`; test `apply_lora()` returns `PeftModel`; test trainable parameter % < 5% for default LoRA config (r=8); test `LORA_TARGET_MODULES` keys cover {distilbert, bert, roberta, deberta}

### Implementation for US2

- [ ] T018 [US2] Implement `src/dissatisfaction_classifier/training/trainer.py`: subclass `Trainer`, override `compute_loss()` to use `nn.CrossEntropyLoss(weight=class_weights)`; `compute_class_weights(train_labels)` using `sklearn.utils.class_weight.compute_class_weight('balanced', classes=[0,1], y=train_labels)`; expose class weights as trainer attribute
- [ ] T019 [P] [US2] Implement `src/dissatisfaction_classifier/training/callbacks.py`: `MetricsCallback` logging train/val loss and F1 to WandB each epoch; `compute_objective` returning F1 for best model selection
- [ ] T020 [P] [US2] Implement `src/dissatisfaction_classifier/evaluation/metrics.py`: `compute_metrics(eval_pred)` returning {"auc_roc", "f1", "precision", "recall", "brier_score", "ece"} for HuggingFace Trainer; `fit_calibration(val_probs, val_labels, output_path)` fitting IsotonicRegression, computing pre/post ECE, exporting to `outputs/calibration/isotonic.pkl`; `export_metrics_json(metrics_dict, output_path)` writing to `outputs/reports/metrics.json`
- [ ] T021 [US2] Create `notebooks/02_training.ipynb` with sections: ETL execution (call prepare_reviews.py inline), config loading via omegaconf, backbone instantiation + LoRA wrapping with trainable param count, class weight computation and justification, training run with loss curves (train + val), checkpoint selection rationale (best F1); key question: *How much does LoRA reduce compute cost without sacrificing performance?*
- [ ] T022 [US2] Create `notebooks/03_evaluation.ipynb` with sections: performance on temporal test set (AUC-ROC, F1, precision, recall as table), calibration analysis (reliability diagram, Brier score, ECE, isotonic fit), confusion matrix at default (0.5) and optimal threshold, precision-recall curve with business cost framing (FP cost = unnecessary escalation, FN cost = reputation damage), baseline comparison table (majority class / TF-IDF+LR / frozen transformer / full fine-tune / LoRA), error analysis (3 representative FPs + 3 FNs, breakdown by star_rating and text length quartile and helpful_votes); call `export_metrics_json()` to write `outputs/reports/metrics.json`; key question: *At what threshold does the model become operationally useful?*

**Checkpoint**: Training converges; `outputs/reports/metrics.json` committed; calibration model in `outputs/calibration/isotonic.pkl`; `pytest tests/test_model.py` passes; both notebooks execute without errors.

---

## Phase 5: US3 — Explainability (Priority: P3)

**Goal**: Token-level attribution via Integrated Gradients and SHAP; methodological comparison in notebook

**Independent Test**: `pytest tests/test_explainability.py -v` passes; `DissatisfactionPredictor.predict()` returns `explanation` dict with tokens + scores + html; `04_explainability.ipynb` runs end-to-end with ≤10 SHAP samples

### Tests for US3 ⚠️ Write FIRST — verify they FAIL before implementing

- [ ] T023 [P] [US3] Write `tests/test_explainability.py`: test `get_integrated_gradients()` returns dict with keys {tokens, scores, html, delta}; test len(scores) == len(tokens); test all scores in [-1.01, 1.01]; test abs(delta) < 0.1 (convergence check); test `explain_with_shap()` raises ValueError when len(texts) > 10; test `explain_with_shap()` returns `shap.Explanation` for ≤10 texts

### Implementation for US3

- [ ] T024 [US3] Implement `src/dissatisfaction_classifier/explainability/integrated_gradients.py`: `get_integrated_gradients(model, tokenizer, text, target_label=1, n_steps=50, device="cuda")` resolving embedding layer backbone-agnostically, using zero-embedding baseline, computing captum IntegratedGradients, L2-normalizing token scores, generating HTML heatmap; return {tokens, scores, html, delta} per contract in `specs/main/contracts/explainability-interface.md`
- [ ] T025 [US3] Implement `src/dissatisfaction_classifier/explainability/shap_explainer.py`: `explain_with_shap(pipeline, texts)` enforcing len(texts) <= 10 with ValueError, creating `shap.Explainer` with `shap.maskers.Text(r"\W")`, returning `shap.Explanation`; per contract in `specs/main/contracts/explainability-interface.md`
- [ ] T026 [US3] Create `notebooks/04_explainability.ipynb` with sections: IG heatmaps on 3 selected examples (high-confidence 1★, borderline 2★, false negative from 03_evaluation.ipynb); SHAP text plots on same 3 examples; side-by-side token attribution comparison table; methodological discussion (theoretical grounding, computational cost, faithfulness, practical use); key question: *Which tokens in a furniture review most reliably signal critical dissatisfaction?*

**Checkpoint**: `pytest tests/test_explainability.py` passes; IG and SHAP produce outputs on same examples; notebook executes without errors.

---

## Phase 6: US4 — Inference Predictor & Local Demo (Priority: P4)

**Goal**: End-to-end inference via `DissatisfactionPredictor`; interactive local Gradio demo

**Independent Test**: `pytest tests/test_inference.py -v` passes; `python demo/app.py` launches at localhost:7860; pasting a 1★ furniture review returns risk_score > 0.5

### Tests for US4 ⚠️ Write FIRST — verify they FAIL before implementing

- [ ] T027 [P] [US4] Write `tests/test_inference.py`: test `DissatisfactionPredictor.predict()` returns dict with keys {risk_score, label, explanation}; test risk_score in [0.0, 1.0]; test label in {0, 1}; test explanation has keys {tokens, scores, html}; test predict() on empty string raises ValueError; test predict_batch() returns list of same structure; test calibrated vs uncalibrated paths (with/without calibration_path)

### Implementation for US4

- [ ] T028 [US4] Implement `src/dissatisfaction_classifier/inference/predictor.py`: `DissatisfactionPredictor.__init__()` loading backbone + LoRA from checkpoint_dir, tokenizer, optional isotonic calibration from calibration_path, setting eval() mode; `predict(text)` returning {risk_score, label, explanation} per contract in `specs/main/contracts/inference-interface.md`; `predict_batch(texts)` running IG (no SHAP) for each text
- [ ] T029 [US4] Implement `demo/app.py`: Gradio `Interface` with single text input (gr.Textbox, placeholder "Paste a furniture review here…"), two outputs (gr.Number label="Dissatisfaction Risk Score", gr.HTML label="Token Attribution"); instantiate `DissatisfactionPredictor` at startup from best checkpoint; prediction function calls `predictor.predict(text)` and returns risk_score + explanation.html

**Checkpoint**: `pytest tests/test_inference.py` passes; Gradio app launches; demo works end-to-end.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Reproducibility validation, committed artifacts, documentation

- [ ] T030 [P] Create placeholder files to commit empty output directories: `outputs/figures/.gitkeep`, `outputs/reports/.gitkeep`, `outputs/calibration/.gitkeep`; verify `.gitignore` only ignores `outputs/checkpoints/`
- [ ] T031 [P] Commit `outputs/reports/metrics.json` (from 03_evaluation.ipynb), `outputs/calibration/isotonic.pkl`, and representative figures from `outputs/figures/` — required by Constitution Principle I (Reproducibility First)
- [ ] T032 [P] Run `pytest tests/ -v` — confirm all 4 test files pass; fix any regressions before proceeding
- [ ] T033 Execute `quickstart.md` validation — run all 8 steps sequentially in a clean conda environment; update quickstart.md with any corrections found
- [ ] T034 [P] Verify WandB integration: `wandb login` documented in quickstart.md; `report_to: wandb` in `configs/training_config.yaml` logs runs; confirm experiment URL is accessible
- [ ] T035 [P] Upload LoRA adapter to HuggingFace Hub: set correct `hub.repo_id` in `configs/model_config.yaml`; verify zero-clone inference pattern from `specs/main/quickstart.md` §"Zero-Clone Inference" works

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup (Phase 1) — **BLOCKS all user stories**
- **US1 (Phase 3)**: Depends on Foundational — no dependencies on US2/US3/US4
- **US2 (Phase 4)**: Depends on Foundational + US1 (needs Parquet data)
- **US3 (Phase 5)**: Depends on Foundational + US2 (needs trained model checkpoint)
- **US4 (Phase 6)**: Depends on Foundational + US2 (needs checkpoint) + US3 (needs IG module)
- **Polish (Phase 7)**: Depends on all user stories complete

### User Story Dependencies

```
Setup → Foundational → US1 → US2 → US3 → US4 → Polish
                                 ↘ US4 (also needs US3 IG module)
```

### Within Each User Story

- Tests MUST be written and FAIL before implementation begins
- For US1: preprocessing.py and validation.py (Foundational) before ETL script
- For US2: dataset.py and model modules (Foundational) before Trainer
- For US3: IG before SHAP (SHAP notebook needs both for comparison)
- For US4: IG module (from US3) before predictor.predict()

---

## Parallel Opportunities

### Phase 1 — All parallel after T001:
```
T001 (structure) → T002, T003, T004, T005, T006, T007 (all in parallel)
```

### Phase 2 — T009 and T010 parallel; T011 before T012:
```
T008 → T009 [P] + T010 [P] → T011 → T012
```

### US1 — T013 and T014 parallel; then T015 and T016 parallel:
```
T013 [P] + T014 [P] → T015 → T016 [P]
```

### US2 — Tests and some implementation parallel:
```
T017 [P] + T018 → T019 [P] + T020 [P] → T021 → T022
```

### US3 — Tests and both explainability modules parallel:
```
T023 [P] → T024 + T025 → T026
```

### US4 — Tests before implementation:
```
T027 [P] → T028 → T029
```

---

## Implementation Strategy

### MVP (US1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 ETL Pipeline
4. **STOP and VALIDATE**: Run ETL; confirm Parquet files and passing tests
5. Commit: reproducible dataset ready

### Incremental Delivery

1. Setup + Foundational → repo skeleton ready
2. US1 → clean dataset, EDA notebook ✅
3. US2 → trained model, evaluation notebooks, metrics JSON ✅
4. US3 → explainability modules and notebook ✅
5. US4 → predictor + Gradio demo ✅
6. Polish → Hub upload, quickstart validated, all tests green ✅

---

## Notes

- `[P]` = different files, no dependencies within phase — safe to run in parallel
- `[Story]` label maps tasks to user stories for traceability
- Tests must be written and confirmed failing before implementation in each story
- SHAP cells in 04_explainability.ipynb MUST be limited to ≤10 texts (documented limitation)
- 3★ reviews MUST be excluded in T015 (ETL) — see Constitution Principle "Scientific Integrity"
- All config values MUST come from YAML files — no string literals in `src/` code (Constitution Principle II)
