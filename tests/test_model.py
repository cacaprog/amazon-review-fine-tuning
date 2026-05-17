import pytest

from dissatisfaction_classifier.models.lora_wrapper import LORA_TARGET_MODULES


class TestLoadBackbone:
    def test_returns_sequence_classification_model(self):
        pytest.importorskip("transformers")
        from omegaconf import OmegaConf
        from transformers import PreTrainedModel

        from dissatisfaction_classifier.models.backbone import load_backbone

        config = OmegaConf.create({"model": {"backbone": "distilbert-base-uncased", "num_labels": 2}})
        model = load_backbone(config)
        assert isinstance(model, PreTrainedModel)

    def test_num_labels_matches_config(self):
        pytest.importorskip("transformers")
        from omegaconf import OmegaConf

        from dissatisfaction_classifier.models.backbone import load_backbone

        config = OmegaConf.create({"model": {"backbone": "distilbert-base-uncased", "num_labels": 2}})
        model = load_backbone(config)
        assert model.config.num_labels == 2


class TestApplyLora:
    def test_returns_peft_model(self):
        pytest.importorskip("peft")
        from omegaconf import OmegaConf
        from peft import PeftModel

        from dissatisfaction_classifier.models.backbone import load_backbone
        from dissatisfaction_classifier.models.lora_wrapper import apply_lora

        config = OmegaConf.create(
            {
                "model": {"backbone": "distilbert-base-uncased", "num_labels": 2},
                "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "bias": "none"},
            }
        )
        base = load_backbone(config)
        peft_model = apply_lora(base, config)
        assert isinstance(peft_model, PeftModel)

    def test_trainable_params_below_5_percent(self):
        pytest.importorskip("peft")
        from omegaconf import OmegaConf

        from dissatisfaction_classifier.models.backbone import load_backbone
        from dissatisfaction_classifier.models.lora_wrapper import apply_lora

        config = OmegaConf.create(
            {
                "model": {"backbone": "distilbert-base-uncased", "num_labels": 2},
                "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "bias": "none"},
            }
        )
        base = load_backbone(config)
        peft_model = apply_lora(base, config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        assert trainable / total < 0.05


class TestLoraTargetModules:
    def test_distilbert_family_covered(self):
        assert "distilbert" in LORA_TARGET_MODULES

    def test_bert_family_covered(self):
        assert "bert" in LORA_TARGET_MODULES

    def test_roberta_family_covered(self):
        assert "roberta" in LORA_TARGET_MODULES

    def test_deberta_family_covered(self):
        assert "deberta" in LORA_TARGET_MODULES
