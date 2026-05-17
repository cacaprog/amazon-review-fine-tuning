import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DissatisfactionDataset(Dataset):
    """PyTorch Dataset for verified Amazon review dissatisfaction classification.

    Args:
        df: DataFrame with columns [text, label] (post-validation).
        tokenizer: HuggingFace tokenizer matched to the model backbone.
        max_length: Maximum token sequence length (from config.model.max_length).
    """

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        self._texts = df["text"].tolist()
        self._labels = df["label"].tolist()
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self._tokenizer(
            self._texts[idx],
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self._labels[idx], dtype=torch.long),
        }
