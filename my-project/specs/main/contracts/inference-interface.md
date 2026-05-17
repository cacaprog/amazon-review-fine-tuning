# Contract: Inference Interface

**Module**: `src/dissatisfaction_classifier/inference/predictor.py`

---

## `DissatisfactionPredictor`

```python
class DissatisfactionPredictor:
    def __init__(
        self,
        checkpoint_dir: str,         # path to best checkpoint (outputs/checkpoints/...)
        config_path: str,            # path to model_config.yaml
        calibration_path: str | None = None,  # path to outputs/calibration/isotonic.pkl
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        """
        Loads backbone + LoRA checkpoint + tokenizer + optional calibration model.
        Model MUST be set to eval() mode on initialization.
        """

    def predict(self, text: str) -> dict:
        """
        Runs end-to-end inference: tokenize → forward pass → calibrate → explain.

        Returns:
            {
                "risk_score":  float,          # calibrated probability in [0, 1]
                "label":       int,            # 0 or 1 at self.threshold
                "explanation": {
                    "tokens": list[str],       # token strings
                    "scores": list[float],     # IG attribution scores [-1, 1]
                    "html":   str,             # HTML heatmap
                }
            }

        Invariants:
        - risk_score MUST use calibrated probability if calibration_path is provided
          and the loaded calibrator is not None; raw sigmoid score otherwise.
        - explanation MUST always be populated — no prediction is returned without it.
        - text MUST be a non-empty string; raises ValueError otherwise.
        - Model MUST remain in eval() mode; no gradient computation during predict().
        """

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Batch inference. Each element follows the same contract as predict().
        SHAP is NOT called in batch mode — IG only.
        """
```

**Loading contract**:
- Backbone name resolved from `config.model.backbone` in the loaded `model_config.yaml`.
- LoRA adapters loaded from `checkpoint_dir` via `PeftModel.from_pretrained()`.
- Calibration model loaded via `joblib.load(calibration_path)` if path exists and is not None.
- If calibration file is missing, log a warning and proceed with uncalibrated scores.

**Threshold behavior**:
```python
label = 1 if risk_score >= self.threshold else 0
```
Default threshold 0.5; configurable at instantiation for business cost optimization
(see `03_evaluation.ipynb` threshold analysis).
