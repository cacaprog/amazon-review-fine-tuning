import pytest


class TestGetIntegratedGradients:
    def test_output_keys(self):
        pytest.importorskip("captum")
        from dissatisfaction_classifier.explainability.integrated_gradients import (
            get_integrated_gradients,
        )

        assert callable(get_integrated_gradients)

    def test_scores_length_matches_tokens(self, monkeypatch):
        """Unit-test the shape contract without loading a real model."""
        from dissatisfaction_classifier.explainability import integrated_gradients as ig_module

        # Stub get_integrated_gradients to return a controlled dict
        def _fake_ig(model, tokenizer, text, **kwargs):
            tokens = text.split()
            scores = [0.1] * len(tokens)
            return {"tokens": tokens, "scores": scores, "html": "<span/>", "delta": 0.01}

        monkeypatch.setattr(ig_module, "get_integrated_gradients", _fake_ig)
        result = ig_module.get_integrated_gradients(None, None, "hello world test")
        assert len(result["scores"]) == len(result["tokens"])

    def test_scores_in_range(self, monkeypatch):
        from dissatisfaction_classifier.explainability import integrated_gradients as ig_module

        def _fake_ig(model, tokenizer, text, **kwargs):
            tokens = ["tok1", "tok2"]
            scores = [0.9, -0.8]
            return {"tokens": tokens, "scores": scores, "html": "", "delta": 0.02}

        monkeypatch.setattr(ig_module, "get_integrated_gradients", _fake_ig)
        result = ig_module.get_integrated_gradients(None, None, "hello world")
        assert all(-1.01 <= s <= 1.01 for s in result["scores"])

    def test_output_has_required_keys(self, monkeypatch):
        from dissatisfaction_classifier.explainability import integrated_gradients as ig_module

        def _fake_ig(model, tokenizer, text, **kwargs):
            return {"tokens": ["a"], "scores": [0.1], "html": "", "delta": 0.0}

        monkeypatch.setattr(ig_module, "get_integrated_gradients", _fake_ig)
        result = ig_module.get_integrated_gradients(None, None, "a")
        assert {"tokens", "scores", "html", "delta"}.issubset(result.keys())


class TestExplainWithShap:
    def test_raises_on_too_many_texts(self):
        from dissatisfaction_classifier.explainability.shap_explainer import explain_with_shap

        with pytest.raises(ValueError, match="10"):
            explain_with_shap(None, ["text"] * 11)

    def test_accepts_up_to_10_texts(self, monkeypatch):
        import shap
        from dissatisfaction_classifier.explainability import shap_explainer as shap_module

        class _FakeExplainer:
            def __init__(self, *a, **kw):
                pass

            def __call__(self, texts):
                return shap.Explanation(
                    values=[[0.1] * 3] * len(texts),
                    base_values=[0.0] * len(texts),
                    data=texts,
                )

        monkeypatch.setattr(shap_module.shap, "Explainer", _FakeExplainer)
        result = shap_module.explain_with_shap(None, ["test text"] * 5)
        assert isinstance(result, shap.Explanation)
