# Contract: Explainability Interface

**Module**: `src/dissatisfaction_classifier/explainability/`

---

## `get_integrated_gradients`

```python
def get_integrated_gradients(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    target_label: int = 1,          # 1 = dissatisfaction class
    n_steps: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Computes Integrated Gradients attribution over the word embedding layer.

    Returns:
        {
            "tokens": list[str],          # decoded token strings
            "scores": list[float],        # IG scores normalized to [-1, 1]
            "html":   str,               # inline HTML heatmap for notebook rendering
            "delta":  float,             # convergence delta (lower is better; target < 0.05)
        }

    Invariants:
    - Baseline MUST be zero embeddings (all-zeros tensor, same shape as input embeddings).
    - Embedding layer resolved backbone-agnostically (see research.md §4).
    - Attribution MUST be computed over the word embedding layer output, not input_ids.
    - Scores MUST be L2-normalized per token across the embedding dimension, then returned
      as a 1D list of length equal to number of tokens.
    - Convergence delta MUST be logged; warn if abs(delta) > 0.05.
    - Model MUST be in eval() mode during attribution.
    """
```

---

## `explain_with_shap`

```python
def explain_with_shap(
    pipeline: transformers.Pipeline,    # text-classification pipeline on MERGED model
    texts: list[str],
) -> shap.Explanation:
    """
    Computes SHAP text attributions using shap.Explainer.

    Args:
        pipeline: A transformers text-classification pipeline wrapping the MERGED
                  (adapter weights merged via model.merge_and_unload()) model.
                  MUST NOT receive a PeftModel directly.
        texts:    List of raw text strings. MUST have len(texts) <= 10 for
                  acceptable runtime.

    Returns:
        shap.Explanation object suitable for shap.plots.text() visualization.

    Invariants:
    - Caller MUST merge LoRA weights before passing the pipeline:
          merged = lora_model.merge_and_unload()
          pipe = pipeline("text-classification", model=merged, tokenizer=tokenizer,
                          return_all_scores=True)
    - Masker MUST be shap.maskers.Text(r"\\W") for word-level attribution.
    - len(texts) > 10 MUST raise ValueError with explanation of the computational cost.
    - Function is NOT suitable for production inference — notebook use only.
    """
```

**Comparison summary** (for notebook narrative):

| Criterion | Integrated Gradients | SHAP |
|-----------|---------------------|------|
| Theoretical grounding | Axiomatic (completeness, sensitivity) | Shapley values |
| Computational cost | Low — gradient computation | High — perturbation-based |
| Faithfulness | Gradient-native to model | Approximation |
| Batch limit | No practical limit | ≤10 texts |
| Notebook | `04_explainability.ipynb` | `04_explainability.ipynb` |
