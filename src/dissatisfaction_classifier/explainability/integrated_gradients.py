import logging

import torch
from captum.attr import IntegratedGradients
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

_RED = "#ff4444"
_BLUE = "#4444ff"


def _resolve_embedding_layer(model) -> torch.nn.Module:
    """Resolve word embedding layer backbone-agnostically."""
    candidates = [
        "base_model.embeddings.word_embeddings",
        "base_model.distilbert.embeddings.word_embeddings",
        "base_model.bert.embeddings.word_embeddings",
        "base_model.roberta.embeddings.word_embeddings",
        "base_model.deberta.embeddings.word_embeddings",
    ]
    module_dict = dict(model.named_modules())
    for path in candidates:
        if path in module_dict:
            return module_dict[path]
    raise ValueError(
        f"Cannot resolve word embedding layer for backbone. Modules: {list(module_dict.keys())[:20]}"
    )


def _token_score_to_html(tokens: list[str], scores: list[float]) -> str:
    parts = ["<span style='font-family:monospace;line-height:2'>"]
    for token, score in zip(tokens, scores):
        intensity = min(abs(score), 1.0)
        color = _RED if score > 0 else _BLUE
        alpha = int(30 + intensity * 200)
        bg = f"{color}{alpha:02x}"
        parts.append(
            f"<span style='background:{bg};padding:2px 4px;margin:1px;border-radius:3px'>"
            f"{token}</span>"
        )
    parts.append("</span>")
    return "".join(parts)


def get_integrated_gradients(
    model,
    tokenizer: PreTrainedTokenizer,
    text: str,
    target_label: int = 1,
    n_steps: int = 50,
    device: str = "cuda",
) -> dict:
    """Compute Integrated Gradients token attribution over the word embedding layer.

    Returns dict with keys: tokens, scores (normalized to [-1,1]), html, delta.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=False,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    embedding_layer = _resolve_embedding_layer(model)
    input_embeddings = embedding_layer(input_ids)
    baseline = torch.zeros_like(input_embeddings)

    def forward_func(embeddings):
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.logits[:, target_label]

    ig = IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(
        input_embeddings,
        baseline,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum over embedding dim, normalize to [-1, 1]
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu()
    norm = token_scores.norm()
    if norm > 0:
        token_scores = token_scores / norm

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    scores = token_scores.tolist()
    delta_val = float(delta.mean().abs())

    if delta_val > 0.05:
        logger.warning("IG convergence delta %.4f > 0.05 — consider increasing n_steps", delta_val)

    return {
        "tokens": tokens,
        "scores": scores,
        "html": _token_score_to_html(tokens, scores),
        "delta": delta_val,
    }
