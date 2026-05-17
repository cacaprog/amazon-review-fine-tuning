import logging

import pandas as pd
import pandera as pa

logger = logging.getLogger(__name__)

_processed_schema = pa.DataFrameSchema(
    {
        "review_id": pa.Column(str, nullable=False, unique=True),
        "star_rating": pa.Column(int, pa.Check.isin([1, 2, 4, 5])),
        "verified_purchase": pa.Column(str, pa.Check.eq("Y")),
        "text": pa.Column(
            str,
            pa.Check(lambda s: s.str.split().str.len() >= 10, element_wise=False),
        ),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
        "split": pa.Column(str, pa.Check.isin(["train", "val", "test"])),
    },
    strict="filter",
)


def validate_reviews_df(df: pd.DataFrame, fail_on_invalid: bool = True) -> pd.DataFrame:
    """Validate processed reviews DataFrame against the ProcessedReview pandera schema.

    If fail_on_invalid=True: raises ValueError with full failure case report on any violation.
    If fail_on_invalid=False: logs violations and returns df with invalid rows removed.
    """
    try:
        return _processed_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        n_failures = len(failure_cases)
        if fail_on_invalid:
            raise ValueError(
                f"Data validation failed with {n_failures} violations:\n{failure_cases}"
            ) from exc
        invalid_idx = failure_cases["index"].dropna().astype(int).unique()
        logger.warning(
            "Dropping %d invalid rows due to schema violations:\n%s",
            len(invalid_idx),
            failure_cases,
        )
        return df.drop(index=invalid_idx).reset_index(drop=True)
