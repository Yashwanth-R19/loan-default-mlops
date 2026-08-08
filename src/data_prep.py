"""Loads the raw loan default dataset, draws a stratified subsample, and writes
a stratified train/test split to data/processed/. Run as: python -m src.data_prep
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    FEATURE_COLUMNS,
    ID_COL,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    load_params,
)


def main() -> None:
    params = load_params()
    random_state = params["random_state"]
    subsample_size = params["data"]["subsample_size"]
    test_size = params["data"]["test_size"]

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    df = df[[ID_COL] + FEATURE_COLUMNS + [TARGET_COL]]

    if subsample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=subsample_size,
            stratify=df[TARGET_COL],
            random_state=random_state,
        )
    print(f"Subsampled to {len(df)} rows (default rate: {df[TARGET_COL].mean():.4f})")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COL],
        random_state=random_state,
    )
    print(
        f"Train: {len(train_df)} rows (default rate: {train_df[TARGET_COL].mean():.4f}) | "
        f"Test: {len(test_df)} rows (default rate: {test_df[TARGET_COL].mean():.4f})"
    )

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Wrote {TRAIN_DATA_PATH} and {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
