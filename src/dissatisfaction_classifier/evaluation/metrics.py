import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    total = bin_sizes.sum()
    if total == 0:
        return 0.0
    ece = np.sum(bin_sizes * np.abs(fraction_of_positives - mean_predicted)) / total
    return float(ece)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute classification metrics for HuggingFace Trainer.

    Returns AUC-ROC, F1, precision, recall, Brier score, and ECE.
    """
    logits, labels = eval_pred
    probs = _softmax(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "auc_roc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "ece": _expected_calibration_error(labels, probs),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_calibration(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    output_path: str,
    ece_threshold: float = 0.05,
) -> IsotonicRegression | None:
    """Fit isotonic regression calibration on validation probabilities.

    Returns None (identity) if ECE is already below ece_threshold.
    Saves calibrator to output_path via joblib.
    """
    ece_before = _expected_calibration_error(val_labels, val_probs)
    logger.info("Pre-calibration ECE: %.4f (threshold=%.2f)", ece_before, ece_threshold)

    if ece_before <= ece_threshold:
        logger.info("ECE within threshold — skipping calibration")
        return None

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, val_labels)

    calibrated_probs = iso.transform(val_probs)
    ece_after = _expected_calibration_error(val_labels, calibrated_probs)
    logger.info("Post-calibration ECE: %.4f", ece_after)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"calibrator": iso, "fit_ece": ece_before, "post_ece": ece_after}, output_path)
    logger.info("Calibration model saved to %s", output_path)

    return iso


def export_metrics_json(metrics: dict, output_path: str) -> None:
    """Write metrics dictionary to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics exported to %s", output_path)
