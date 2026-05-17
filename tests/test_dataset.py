import pandas as pd
import pytest

from dissatisfaction_classifier.data.dataset import DissatisfactionDataset
from dissatisfaction_classifier.data.validation import validate_reviews_df


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "star_rating": [1, 5, 2],
            "verified_purchase": ["Y", "Y", "Y"],
            "text": [
                "This product broke after one day and is terrible quality overall",
                "Great sturdy table arrived quickly and looks beautiful in my office",
                "Falls apart immediately very disappointed with this furniture purchase",
            ],
            "label": [1, 0, 1],
            "split": ["train", "train", "val"],
        }
    )


@pytest.fixture
def mock_tokenizer():
    """Minimal tokenizer stub that returns fixed-length tensors."""
    import torch

    class _Tok:
        model_max_length = 256

        def __call__(self, text, max_length, padding, truncation, return_tensors):
            import torch

            input_ids = torch.zeros(1, max_length, dtype=torch.long)
            attention_mask = torch.ones(1, max_length, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    return _Tok()


# ── DissatisfactionDataset tests ─────────────────────────────────────────────

class TestDissatisfactionDataset:
    def test_len(self, valid_df, mock_tokenizer):
        ds = DissatisfactionDataset(valid_df, mock_tokenizer, max_length=256)
        assert len(ds) == 3

    def test_getitem_keys(self, valid_df, mock_tokenizer):
        ds = DissatisfactionDataset(valid_df, mock_tokenizer, max_length=256)
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_getitem_input_ids_shape(self, valid_df, mock_tokenizer):
        ds = DissatisfactionDataset(valid_df, mock_tokenizer, max_length=256)
        item = ds[0]
        assert item["input_ids"].shape == (256,)

    def test_getitem_attention_mask_shape(self, valid_df, mock_tokenizer):
        ds = DissatisfactionDataset(valid_df, mock_tokenizer, max_length=256)
        item = ds[0]
        assert item["attention_mask"].shape == (256,)

    def test_getitem_label_scalar(self, valid_df, mock_tokenizer):
        ds = DissatisfactionDataset(valid_df, mock_tokenizer, max_length=256)
        item = ds[0]
        assert item["labels"].ndim == 0
        assert item["labels"].item() in {0, 1}


# ── validate_reviews_df tests ─────────────────────────────────────────────────

class TestValidateReviewsDf:
    def test_valid_df_passes(self, valid_df):
        result = validate_reviews_df(valid_df, fail_on_invalid=True)
        assert len(result) == len(valid_df)

    def test_raises_on_invalid_when_strict(self, valid_df):
        bad = valid_df.copy()
        bad.loc[0, "verified_purchase"] = "N"
        with pytest.raises(ValueError, match="validation failed"):
            validate_reviews_df(bad, fail_on_invalid=True)

    def test_drops_invalid_rows_when_lenient(self, valid_df):
        bad = valid_df.copy()
        bad.loc[0, "verified_purchase"] = "N"
        result = validate_reviews_df(bad, fail_on_invalid=False)
        assert len(result) == 2
        assert (result["verified_purchase"] == "Y").all()

    def test_raises_on_invalid_label(self):
        df = pd.DataFrame(
            {
                "review_id": ["r1"],
                "star_rating": [1],
                "verified_purchase": ["Y"],
                "text": ["This is a sufficiently long text for validation purposes here"],
                "label": [99],
                "split": ["train"],
            }
        )
        with pytest.raises(ValueError):
            validate_reviews_df(df, fail_on_invalid=True)

    def test_raises_on_short_text(self):
        df = pd.DataFrame(
            {
                "review_id": ["r1"],
                "star_rating": [1],
                "verified_purchase": ["Y"],
                "text": ["Too short"],
                "label": [1],
                "split": ["train"],
            }
        )
        with pytest.raises(ValueError):
            validate_reviews_df(df, fail_on_invalid=True)
