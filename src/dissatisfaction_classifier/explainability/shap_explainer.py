import logging

import shap

logger = logging.getLogger(__name__)

_MAX_TEXTS = 10


def explain_with_shap(pipeline, texts: list[str]) -> shap.Explanation:
    """Compute SHAP token attributions using shap.Explainer on a merged LoRA pipeline.

    Args:
        pipeline: A transformers text-classification pipeline wrapping a MERGED model
                  (call model.merge_and_unload() before creating the pipeline).
                  Must NOT receive a PeftModel directly.
        texts: Raw text strings. MUST have len(texts) <= 10 for acceptable runtime.

    Returns:
        shap.Explanation suitable for shap.plots.text().
    """
    if len(texts) > _MAX_TEXTS:
        raise ValueError(
            f"explain_with_shap accepts at most {_MAX_TEXTS} texts "
            f"(received {len(texts)}). SHAP is O(n_tokens²) — use Integrated Gradients "
            f"for larger batches."
        )

    explainer = shap.Explainer(pipeline, masker=shap.maskers.Text(r"\W"))
    shap_values = explainer(texts)
    return shap_values
