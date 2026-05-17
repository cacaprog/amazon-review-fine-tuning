import logging
from pathlib import Path

import joblib
import torch
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dissatisfaction_classifier.explainability.integrated_gradients import get_integrated_gradients
from dissatisfaction_classifier.models.backbone import load_backbone

logger = logging.getLogger(__name__)


class DissatisfactionPredictor:
    """End-to-end inference: raw text → calibrated risk score + token explanation.

    Args:
        checkpoint_dir: Path to best LoRA checkpoint (outputs/checkpoints/...).
        config_path: Path to model_config.yaml.
        calibration_path: Optional path to outputs/calibration/isotonic.pkl.
        threshold: Decision threshold for label assignment (default 0.5).
        device: Torch device string.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config_path: str,
        calibration_path: str | None = None,
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._threshold = threshold
        self._calibrator = None

        config = OmegaConf.load(config_path)
        self._config = config

        base = load_backbone(config)
        self._model = PeftModel.from_pretrained(base, checkpoint_dir)
        self._model.to(self._device).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        if calibration_path and Path(calibration_path).exists():
            payload = joblib.load(calibration_path)
            self._calibrator = payload.get("calibrator")
            logger.info("Loaded calibration model from %s", calibration_path)
        elif calibration_path:
            logger.warning("Calibration file not found at %s — using raw scores", calibration_path)

    def _raw_prob(self, text: str) -> float:
        encoding = self._tokenizer(
            text,
            max_length=self._config.model.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return prob

    def _calibrate(self, prob: float) -> float:
        if self._calibrator is None:
            return prob
        import numpy as np

        return float(self._calibrator.transform(np.array([prob]))[0])

    def predict(self, text: str) -> dict:
        """Run end-to-end inference on a single text.

        Returns:
            {
                "risk_score": float in [0, 1] (calibrated),
                "label": int (0 or 1),
                "explanation": {"tokens": [...], "scores": [...], "html": str}
            }
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        raw_prob = self._raw_prob(text)
        risk_score = self._calibrate(raw_prob)
        label = int(risk_score >= self._threshold)

        explanation = get_integrated_gradients(
            self._model,
            self._tokenizer,
            text,
            target_label=1,
            device=str(self._device),
        )

        return {
            "risk_score": risk_score,
            "label": label,
            "explanation": {
                "tokens": explanation["tokens"],
                "scores": explanation["scores"],
                "html": explanation["html"],
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batch inference — IG attribution for each text (no SHAP)."""
        return [self.predict(text) for text in texts]
