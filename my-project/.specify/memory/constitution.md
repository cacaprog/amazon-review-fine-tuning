<!--
## Sync Impact Report

**Version change**: (template) → 1.0.0

**Modified principles**: N/A — initial ratification from template

**Added sections**:
- Core Principles (I–V: Reproducibility First, Config-Driven Architecture, Temporal
  Integrity, Explainability as First-Class Concern, Test-Driven Data Quality)
- Scientific Integrity & Ethics
- Portfolio & Documentation Standards
- Governance

**Removed sections**: N/A

**Templates requiring updates**:
- `.specify/templates/plan-template.md` — Constitution Check section is generic; references
  principles by name when filled in by /speckit-plan. ✅ No structural update needed.
- `.specify/templates/spec-template.md` — No explicit constitution references. ✅ No update needed.
- `.specify/templates/tasks-template.md` — Task types (observability, testing, validation)
  align with Principles IV and V. ✅ No update needed.

**Follow-up TODOs**: None. All placeholders resolved.
-->

# Critical Dissatisfaction Early Warning Constitution

## Core Principles

### I. Reproducibility First

Every experiment MUST be reproducible end-to-end by anyone cloning the repository.

- All random seeds MUST be fixed and declared in `configs/data_config.yaml` (ETL) and
  `configs/training_config.yaml` (training). Both files use `seed: 42`.
- `environment.yml` with pinned dependency versions MUST be committed and kept current.
- Exported metrics JSON MUST be written to `outputs/reports/` after each evaluation run
  so results can be inspected without retraining or a GPU.
- The HuggingFace Hub adapter MUST be uploaded at the end of a training run to enable
  zero-clone inference for portfolio reviewers.
- `outputs/calibration/` isotonic regression models MUST be committed alongside checkpoints.

**Rationale**: A portfolio ML project that cannot be independently reproduced cannot be
evaluated or trusted. Reproducibility is the minimum bar for scientific credibility.

### II. Config-Driven Architecture

No model name, data path, training hyperparameter, or LoRA setting MAY be hardcoded in
source code.

- All configurable values MUST live in `configs/model_config.yaml`,
  `configs/training_config.yaml`, or `configs/data_config.yaml`.
- The model backbone MUST be resolved at runtime from `config.model.backbone`; no model
  family name or checkpoint path may appear as a string literal in `src/`.
- LoRA target modules MUST be resolved per backbone family in `lora_wrapper.py`, not
  hardcoded per model checkpoint.
- Any new configurable value introduced during development MUST be added to the relevant
  config file and documented in the Configuration Contract (project.md §8).

**Rationale**: Config-driven design allows swapping DistilBERT for DeBERTa or any other
HuggingFace backbone without touching source code — a central architectural claim of this
project. Violations break backbone swappability.

### III. Temporal Integrity

Data splits MUST follow temporal ordering. Random splitting on time-ordered review data
is prohibited.

- The Furniture dataset spans **2000-03-17 (minimum) to 2015-08-31 (maximum)** as
  confirmed by inspection of the actual TSV file. All split boundary dates MUST fall
  within this range.
- Split boundaries MUST be declared in `configs/data_config.yaml` and MUST NOT be
  overridden in source code. The calibrated boundaries based on the actual dataset are:
  - Train: up to 2014-12-31 (~406k rows, 70% of filtered data)
  - Validation: 2015-01-01 to 2015-04-30 (~88k rows)
  - Test: 2015-05-01 to 2015-08-31 (~82k rows)
- Data outside the configured split boundaries is intentionally excluded from the analysis
  scope. Any change to these boundaries MUST be documented with a rationale and
  accompanied by a constitution amendment.
- Cross-validation for hyperparameter search MUST operate only within the training period.
  It MUST NOT access validation or test data.

**Rationale**: Temporal splitting simulates real deployment conditions and prevents
information leakage from future reviews into training. Violating the no-random-split
rule invalidates all evaluation claims on the test set.

### IV. Explainability as First-Class Concern

Every inference path MUST return token-level attribution scores alongside the risk score.

- Integrated Gradients (via `captum.attr.IntegratedGradients`) is the primary
  explainability method and MUST be implemented in `integrated_gradients.py`.
- SHAP (via `shap.Explainer` on merged LoRA weights) is the secondary method and MUST be
  implemented in `shap_explainer.py` and compared in `notebooks/04_explainability.ipynb`.
- `DissatisfactionPredictor` MUST return `explanation` (tokens + IG attribution scores)
  as part of its standard output contract.
- No new inference capability MAY be added without a corresponding explainability path.

**Rationale**: The core thesis is that text drives predictions that are human-interpretable.
A model that cannot explain its outputs defeats this thesis and is unsuitable for
operational use or portfolio demonstration.

### V. Test-Driven Data Quality

Data validation MUST execute before any training or evaluation run. Pipeline failures on
invalid data MUST halt execution — silent errors are prohibited.

- `validate_reviews_df()` MUST be called in the ETL pipeline before any downstream
  processing.
- `data.validation.fail_on_invalid: true` is the required default in
  `configs/data_config.yaml`. Changing this to `false` requires explicit justification.
- Unit tests MUST cover all four core modules: `DissatisfactionDataset`, backbone loading
  via `load_backbone()`, both explainability modules, and `DissatisfactionPredictor`.
- All tests MUST be runnable via `pytest` from the repository root with no additional
  setup beyond the declared `environment.yml`.

**Rationale**: Silent data quality failures produce misleading evaluation results and
undetectable model degradation. Hard failures force upstream fixes at the correct
abstraction layer.

## Scientific Integrity & Ethics

- The dataset MUST be public, real, and cited with full provenance (source URL, category,
  version) in project.md §3.1.
- Only verified purchases (`verified_purchase == 'Y'`) MAY be used to exclude astroturfed
  or incentivized signals.
- Natural class imbalance (~15–20% positive) MUST be preserved. Oversampling or
  undersampling to artificially balance the dataset is prohibited. Imbalance is addressed
  exclusively via class weights in the loss function.
- All evaluation claims MUST include a baseline comparison table: majority class,
  TF-IDF + Logistic Regression, frozen transformer (no fine-tuning), and full fine-tune
  (no LoRA). Claims without baselines MUST NOT be presented as conclusions.
- 3★ reviews (ambiguous satisfaction signal) MUST be excluded from the dataset. This
  exclusion MUST be documented and justified in `notebooks/01_eda.ipynb`.

## Portfolio & Documentation Standards

- All four notebooks MUST include a "Key question" section that frames the narrative for
  a technical reviewer reading the portfolio cold.
- Methodology decisions (temporal split rationale, LoRA rank, threshold selection
  criterion, explainability method comparison) MUST be documented inline in the relevant
  notebook — not left implicit in code comments.
- The repository is public (GitHub portfolio). All committed artifacts MUST be
  presentable: no debugging output, no placeholder text, no broken notebook cells in
  committed state.
- `outputs/reports/` metrics artifacts MUST be committed so the portfolio is self-contained
  without requiring a GPU rerun by the reviewer.

## Governance

This constitution supersedes all other informal practices for this project. It governs all
development, evaluation, and documentation decisions.

**Amendment procedure**:
1. Identify the principle or section requiring change.
2. Update this file with the revised text.
3. Increment `CONSTITUTION_VERSION` per the versioning policy below.
4. Update `LAST_AMENDED_DATE` to the amendment date (ISO format).
5. Add a Sync Impact Report entry at the top of this file.
6. Commit with message: `docs: amend constitution to vX.Y.Z (<change summary>)`.

**Versioning policy**:
- MAJOR: Principle removed, renamed, or fundamentally redefined.
- MINOR: New principle or section added; existing principle materially expanded.
- PATCH: Wording clarification, typo fix, non-semantic refinement.

**Compliance**: All implementation tasks generated by `/speckit-tasks` MUST be verified
against this constitution before execution. The Constitution Check gate in `plan.md` MUST
reference Principles I–V explicitly before any Phase 0 research begins.

**Version**: 1.0.0 | **Ratified**: 2026-05-17 | **Last Amended**: 2026-05-17
