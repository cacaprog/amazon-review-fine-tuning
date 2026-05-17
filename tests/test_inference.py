import pytest


class TestDissatisfactionPredictorContract:
    """Contract tests for DissatisfactionPredictor.predict() output shape.

    These tests use monkeypatching to avoid requiring a trained checkpoint.
    """

    @pytest.fixture
    def mock_predictor(self, monkeypatch, tmp_path):
        """Return a predictor with all I/O mocked out."""
        import torch
        from omegaconf import OmegaConf

        from dissatisfaction_classifier.inference import predictor as pred_module

        # Stub out model/tokenizer loading
        monkeypatch.setattr(pred_module, "load_backbone", lambda cfg: None)
        monkeypatch.setattr(pred_module.PeftModel, "from_pretrained", lambda base, path: _FakeModel())
        monkeypatch.setattr(pred_module.AutoTokenizer, "from_pretrained", lambda path: _FakeTokenizer())

        # Stub IG
        monkeypatch.setattr(
            pred_module,
            "get_integrated_gradients",
            lambda model, tok, text, **kw: {
                "tokens": text.split(),
                "scores": [0.1] * len(text.split()),
                "html": "<span/>",
                "delta": 0.01,
            },
        )

        config_path = tmp_path / "model_config.yaml"
        OmegaConf.save(
            OmegaConf.create({"model": {"backbone": "distilbert-base-uncased", "num_labels": 2, "max_length": 256}}),
            str(config_path),
        )

        predictor = pred_module.DissatisfactionPredictor.__new__(pred_module.DissatisfactionPredictor)
        predictor._device = torch.device("cpu")
        predictor._threshold = 0.5
        predictor._calibrator = None
        predictor._config = OmegaConf.load(str(config_path))
        predictor._model = _FakeModel()
        predictor._tokenizer = _FakeTokenizer()

        return predictor

    def test_predict_output_keys(self, mock_predictor):
        result = mock_predictor.predict("This furniture broke after one week and is terrible")
        assert {"risk_score", "label", "explanation"}.issubset(result.keys())

    def test_risk_score_in_range(self, mock_predictor):
        result = mock_predictor.predict("Great product loved it very much excellent quality")
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_label_binary(self, mock_predictor):
        result = mock_predictor.predict("This is a perfectly acceptable furniture review")
        assert result["label"] in {0, 1}

    def test_explanation_keys(self, mock_predictor):
        result = mock_predictor.predict("Broke immediately very disappointed with this product")
        assert {"tokens", "scores", "html"}.issubset(result["explanation"].keys())

    def test_raises_on_empty_text(self, mock_predictor):
        with pytest.raises(ValueError):
            mock_predictor.predict("")

    def test_raises_on_whitespace_text(self, mock_predictor):
        with pytest.raises(ValueError):
            mock_predictor.predict("   ")

    def test_predict_batch_returns_list(self, mock_predictor, monkeypatch):
        monkeypatch.setattr(
            mock_predictor,
            "predict",
            lambda text: {"risk_score": 0.3, "label": 0, "explanation": {"tokens": [], "scores": [], "html": ""}},
        )
        results = mock_predictor.predict_batch(["text one", "text two"])
        assert isinstance(results, list)
        assert len(results) == 2


# ── Stubs ────────────────────────────────────────────────────────────────────

class _FakeModel:
    training = False

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        import torch

        class _Out:
            logits = torch.tensor([[1.0, 2.0]])

        return _Out()

    def named_modules(self):
        import torch.nn as nn

        return [("base_model.embeddings.word_embeddings", nn.Embedding(100, 32))]


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        import torch

        ml = kwargs.get("max_length", 256)
        return {
            "input_ids": torch.zeros(1, ml, dtype=torch.long),
            "attention_mask": torch.ones(1, ml, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        return [f"tok{i}" for i in ids]
