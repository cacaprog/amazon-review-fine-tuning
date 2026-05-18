"""Explainability analysis: Integrated Gradients + SHAP on representative examples."""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoTokenizer, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dissatisfaction_classifier.explainability.integrated_gradients import get_integrated_gradients
from dissatisfaction_classifier.explainability.shap_explainer import explain_with_shap
from dissatisfaction_classifier.models.backbone import load_backbone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT = Path("outputs/checkpoints/best")
OUT = Path("outputs/explainability")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_cfg = OmegaConf.load("configs/model_config.yaml")


def select_examples(test_df, model, tokenizer):
    """Pick high-confidence 1★, borderline 2★, and a false negative from the test set."""
    logger.info("Selecting representative examples via model inference...")

    def score(texts, batch_size=64):
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, max_length=model_cfg.model.max_length,
                            padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        return np.array(probs)

    ones = test_df[test_df["star_rating"] == 1].reset_index(drop=True)
    twos = test_df[test_df["star_rating"] == 2].reset_index(drop=True)

    ones_probs = score(ones["text"].tolist())
    twos_probs = score(twos["text"].tolist())

    # High-confidence 1★: highest probability of dissatisfaction
    high_conf_idx = ones_probs.argmax()
    high_conf_text = ones["text"].iloc[high_conf_idx]
    high_conf_prob = ones_probs[high_conf_idx]

    # Borderline 2★: probability closest to 0.5
    borderline_idx = np.abs(twos_probs - 0.5).argmin()
    borderline_text = twos["text"].iloc[borderline_idx]
    borderline_prob = twos_probs[borderline_idx]

    # False negative: 1★ review with lowest dissatisfaction probability
    fn_idx = ones_probs.argmin()
    fn_text = ones["text"].iloc[fn_idx]
    fn_prob = ones_probs[fn_idx]

    examples = {
        "high_confidence_1star": (high_conf_text, high_conf_prob),
        "borderline_2star":      (borderline_text, borderline_prob),
        "false_negative":        (fn_text, fn_prob),
    }

    for name, (text, prob) in examples.items():
        logger.info("%s (p=%.3f): %s…", name, prob, text[:120])

    return {name: text for name, (text, prob) in examples.items()}


def main():
    # ── Load data and model ───────────────────────────────────────────────────
    test_df = pd.read_parquet("data/processed/test.parquet")
    logger.info("Loaded test set: %d rows", len(test_df))

    logger.info("Loading checkpoint from %s", CHECKPOINT)
    base = load_backbone(model_cfg)
    lora_model = PeftModel.from_pretrained(base, str(CHECKPOINT)).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))

    # ── Select examples ───────────────────────────────────────────────────────
    examples = select_examples(test_df, lora_model, tokenizer)

    # ── Integrated Gradients ──────────────────────────────────────────────────
    logger.info("Running Integrated Gradients (n_steps=50)...")
    ig_results = {}
    html_parts = []

    for name, text in examples.items():
        result = get_integrated_gradients(lora_model, tokenizer, text, n_steps=50, device=DEVICE)
        ig_results[name] = result
        logger.info("%s — convergence delta: %.4f", name, result["delta"])

        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(f"<p><b>{text[:300]}</b></p>")
        html_parts.append(result["html"])
        html_parts.append("<hr>")

    ig_html_path = OUT / "ig_heatmaps.html"
    ig_html_path.write_text(
        "<html><body style='font-family:monospace'>" + "\n".join(html_parts) + "</body></html>"
    )
    logger.info("IG heatmaps saved to %s", ig_html_path)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    logger.info("Merging LoRA weights for SHAP compatibility...")
    merged_model = lora_model.merge_and_unload()

    shap_pipeline = pipeline(
        "text-classification",
        model=merged_model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1,
    )

    texts_for_shap = list(examples.values())
    logger.info("Running SHAP on %d texts...", len(texts_for_shap))
    shap_explanation = explain_with_shap(shap_pipeline, texts_for_shap)

    shap_fig = shap.plots.text(shap_explanation, display=False)
    shap_plot_path = OUT / "shap_text_plot.html"
    if hasattr(shap_fig, "html"):
        shap_plot_path.write_text(shap_fig.html())
    else:
        import io
        buf = io.StringIO()
        shap.plots.text(shap_explanation, display=False)
        plt.savefig(OUT / "shap_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
    logger.info("SHAP output saved to %s", OUT)

    # ── IG vs SHAP token comparison (high-confidence example) ─────────────────
    logger.info("Building IG vs SHAP token comparison...")
    name = "high_confidence_1star"
    ig = ig_results[name]
    shap_vals = shap_explanation[0].values   # (n_tokens, n_classes)
    shap_tokens = shap_explanation[0].data

    ig_df = pd.DataFrame({"token": ig["tokens"], "ig_score": ig["scores"]})
    shap_df = pd.DataFrame({"token": shap_tokens, "shap_score": shap_vals[:, 1]})

    comparison = (
        ig_df.merge(shap_df, on="token", how="outer")
        .sort_values("ig_score", ascending=False)
        .reset_index(drop=True)
    )

    comparison_path = OUT / "ig_shap_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info("Token comparison saved to %s", comparison_path)

    print("\nTop 20 tokens by IG score (high_confidence_1star):")
    print(comparison.head(20).to_string(index=False))

    # ── Bar chart: top tokens ─────────────────────────────────────────────────
    top = comparison.dropna().head(15)
    x = range(len(top))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - 0.2 for i in x], top["ig_score"], width=0.4, label="IG", color="steelblue")
    ax.bar([i + 0.2 for i in x], top["shap_score"], width=0.4, label="SHAP", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(top["token"].tolist(), rotation=45, ha="right")
    ax.set_title("Top token attributions: IG vs SHAP (high-confidence 1★)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ig_shap_bar.png", dpi=150)
    plt.close()

    logger.info("All explainability outputs saved to %s/", OUT)


if __name__ == "__main__":
    main()
