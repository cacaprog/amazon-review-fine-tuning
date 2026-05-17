"""ETL pipeline: raw Amazon Furniture TSV → validated, temporally-split Parquet files."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

# Allow running as a script before the package is installed
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dissatisfaction_classifier.data.preprocessing import (
    assign_label,
    assign_split,
    build_text,
    filter_reviews,
)
from dissatisfaction_classifier.data.validation import validate_reviews_df

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_etl(config_path: str) -> None:
    config = OmegaConf.load(config_path)

    raw_path = Path(config.data.raw_path)
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(
        raw_path,
        sep="\t",
        on_bad_lines="skip",
        dtype={"star_rating": "Int64"},
        parse_dates=["review_date"],
        low_memory=False,
    )
    logger.info("Loaded %d rows", len(df))

    # Build text and label before filtering so we can apply token-length filter
    df["text"] = df.apply(build_text, axis=1)
    df["label"] = df["star_rating"].apply(assign_label)

    df = filter_reviews(df, config)
    logger.info("After filtering: %d rows", len(df))

    # Assign temporal splits; drop rows outside analysis window
    df["split"] = df["review_date"].apply(lambda d: assign_split(d, config))
    df = df[df["split"].notna()].copy()
    logger.info("After temporal windowing: %d rows", len(df))

    # Keep only columns needed downstream
    df = df[
        ["review_id", "star_rating", "verified_purchase", "text", "label", "split",
         "review_date", "helpful_votes"]
    ].copy()
    df["star_rating"] = df["star_rating"].astype(int)

    fail_on_invalid = config.data.validation.fail_on_invalid
    df = validate_reviews_df(df, fail_on_invalid=fail_on_invalid)
    logger.info("Validation passed — %d rows remaining", len(df))

    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        out_path = processed_dir / f"{split_name}.parquet"
        split_df.to_parquet(out_path, index=False)

        n_pos = (split_df["label"] == 1).sum()
        n_total = len(split_df)
        logger.info(
            "%-5s → %d rows | positive (dissatisfied): %d (%.1f%%)",
            split_name,
            n_total,
            n_pos,
            100 * n_pos / n_total if n_total else 0,
        )

    logger.info("✓ ETL complete. Parquet files written to %s/", processed_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare Amazon review dataset")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()
    run_etl(args.config)


if __name__ == "__main__":
    main()
