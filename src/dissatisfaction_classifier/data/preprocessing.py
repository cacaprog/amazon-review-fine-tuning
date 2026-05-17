import pandas as pd
from omegaconf import DictConfig


def build_text(row: pd.Series) -> str:
    headline = str(row.get("review_headline", "") or "")
    body = str(row.get("review_body", "") or "")
    return f"{headline}. {body}".strip(". ").strip()


def assign_label(star_rating: int) -> int:
    return 1 if star_rating <= 2 else 0


def assign_split(review_date: pd.Timestamp, config: DictConfig) -> str | None:
    train_end = pd.Timestamp(config.data.train_end_date)
    val_end = pd.Timestamp(config.data.val_end_date)
    test_end = pd.Timestamp(config.data.test_end_date)

    if review_date <= train_end:
        return "train"
    elif review_date <= val_end:
        return "val"
    elif review_date <= test_end:
        return "test"
    return None


def filter_reviews(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    min_tokens = config.data.min_tokens

    mask = (
        (df["verified_purchase"] == "Y")
        & (df["star_rating"] != 3)
        & (df["text"].str.split().str.len() >= min_tokens)
    )
    return df[mask].copy()
