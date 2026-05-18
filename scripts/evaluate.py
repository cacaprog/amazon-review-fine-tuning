"""Evaluate the best checkpoint on the temporal test set and export metrics + figures."""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import OmegaConf
from peft import PeftModel
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dissatisfaction_classifier.evaluation.metrics import export_metrics_json, fit_calibration
from dissatisfaction_classifier.models.backbone import load_backbone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

CHECKPOINT = Path("outputs/checkpoints/best")
FIGURES = Path("outputs/figures")
FIGURES.mkdir(parents=True, exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)
Path("outputs/calibration").mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_cfg = OmegaConf.load("configs/model_config.yaml")


def get_probs(model, tokenizer, df, batch_size=128):
    all_probs = []
    for i in range(0, len(df), batch_size):
        batch = df["text"].iloc[i : i + batch_size].tolist()
        enc = tokenizer(
            batch,
            max_length=model_cfg.model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        if i % 10000 == 0:
            logger.info("  %d / %d", i, len(df))
    return np.array(all_probs)


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    train_df = pd.read_parquet("data/processed/train.parquet")
    logger.info("Val: %d | Test: %d | Train: %d", len(val_df), len(test_df), len(train_df))

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info("Loading checkpoint from %s", CHECKPOINT)
    base = load_backbone(model_cfg)
    model = PeftModel.from_pretrained(base, str(CHECKPOINT)).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))

    # ── Generate predictions ─────────────────────────────────────────────────
    logger.info("Running inference on val set (%d rows)...", len(val_df))
    val_probs = get_probs(model, tokenizer, val_df)

    logger.info("Running inference on test set (%d rows)...", len(test_df))
    test_probs = get_probs(model, tokenizer, test_df)

    # ── Calibration ──────────────────────────────────────────────────────────
    calibrator = fit_calibration(
        val_probs,
        val_df["label"].values,
        output_path="outputs/calibration/isotonic.pkl",
    )
    test_probs_cal = calibrator.transform(test_probs) if calibrator else test_probs

    # Reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, probs, label in zip(axes, [test_probs, test_probs_cal], ["Raw", "Calibrated"]):
        frac, mean_pred = calibration_curve(test_df["label"], probs, n_bins=10)
        ax.plot(mean_pred, frac, "o-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_title(f"Reliability Diagram ({label})")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend()
    fig.savefig(FIGURES / "reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Test metrics ─────────────────────────────────────────────────────────
    y_true = test_df["label"].values
    y_pred_default = (test_probs_cal >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_true, test_probs_cal)),
        "f1": float(f1_score(y_true, y_pred_default, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_default, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_default, zero_division=0)),
    }
    logger.info("Test metrics (threshold=0.5): %s", json.dumps(metrics, indent=2))

    # ── Threshold analysis ───────────────────────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_true, test_probs_cal)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = float(thresholds[optimal_idx])
    logger.info("Optimal threshold (max F1): %.3f", optimal_threshold)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(recalls, precisions, linewidth=2)
    ax.scatter(
        recalls[optimal_idx], precisions[optimal_idx],
        s=100, zorder=5, color="red",
        label=f"Optimal (t={optimal_threshold:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.savefig(FIGURES / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Confusion matrices ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, threshold, label in zip(
        axes, [0.5, optimal_threshold],
        ["Default (0.5)", f"Optimal ({optimal_threshold:.2f})"],
    ):
        y_pred = (test_probs_cal >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(label)
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Baseline comparison ──────────────────────────────────────────────────
    logger.info("Fitting TF-IDF + LR baseline...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df["text"])
    X_test = tfidf.transform(test_df["text"])
    lr = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
    lr.fit(X_train, train_df["label"])
    lr_pred = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    baselines = {
        "Majority class": {"AUC-ROC": 0.5, "F1": f1_score(y_true, np.zeros(len(y_true), int), zero_division=0)},
        "TF-IDF + LR":    {"AUC-ROC": roc_auc_score(y_true, lr_probs), "F1": f1_score(y_true, lr_pred, zero_division=0)},
        "DistilBERT+LoRA":{"AUC-ROC": metrics["auc_roc"], "F1": metrics["f1"]},
    }
    logger.info("Baseline comparison:\n%s", pd.DataFrame(baselines).T.to_string())

    # ── Export ───────────────────────────────────────────────────────────────
    full_metrics = {
        **metrics,
        "optimal_threshold": optimal_threshold,
        "n_test": int(len(test_df)),
        "positive_rate_test": float(y_true.mean()),
        "baselines": baselines,
    }
    export_metrics_json(full_metrics, "outputs/reports/metrics.json")
    logger.info("Done. Figures saved to %s/", FIGURES)
    print(json.dumps({k: v for k, v in full_metrics.items() if k != "baselines"}, indent=2))


if __name__ == "__main__":
    main()
