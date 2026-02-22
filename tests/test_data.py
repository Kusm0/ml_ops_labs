"""Pre-train tests: data validation and schema (Lab 4)."""

import os
import pandas as pd

from conftest import get_data_path, MIN_ROWS, REQUIRED_COLUMNS


def test_data_schema_basic() -> None:
    """Check that train data exists and has required columns and minimum rows."""
    data_path = get_data_path()
    assert os.path.exists(data_path), f"Data not found: {data_path}"
    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert df["popularity"].notna().all(), "popularity contains NaNs"
    assert (
        df.shape[0] >= MIN_ROWS
    ), f"Too few rows for a learning experiment (got {df.shape[0]}, need >= {MIN_ROWS})"


def test_data_no_critical_missing() -> None:
    """Check no critical missing values in feature and target columns."""
    data_path = get_data_path()
    if not os.path.exists(data_path):
        return
    df = pd.read_csv(data_path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        assert null_count == 0, f"Column {col} has {null_count} missing values"
