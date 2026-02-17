"""
Data preprocessing for Spotify popularity dataset.

Reads data/raw/dataset.csv, applies:
- Numeric encoding for track_genre (LabelEncoder)
- Outlier removal (IQR method on numerical columns)
- Normalization of numerical features (StandardScaler)

Saves result to data/processed/dataset.csv and genre mapping to data/processed/genre_mapping.json.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"

# Numerical columns to normalize and use for outlier detection (excluding target)
NUMERIC_FEATURE_COLUMNS = [
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "explicit_numeric",
]
TARGET_COLUMN = "popularity"
GENRE_COLUMN = "track_genre"
OUTPUT_FILENAME = "dataset.csv"
MAPPING_FILENAME = "genre_mapping.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess Spotify dataset")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_RAW_PATH),
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for processed CSV and metadata",
    )
    parser.add_argument(
        "--iqr_mult",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier bounds (default 1.5)",
    )
    return parser.parse_args()


def load_raw(input_path: str) -> pd.DataFrame:
    """Load raw CSV and add explicit_numeric."""
    logger.info("Loading %s", input_path)
    df = pd.read_csv(input_path)
    if "explicit" in df.columns:
        df["explicit_numeric"] = (
            df["explicit"].astype(str).str.lower() == "true"
        ).astype(int)
    required = set(NUMERIC_FEATURE_COLUMNS) | {TARGET_COLUMN, GENRE_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=list(required))
    logger.info("Loaded %d rows", len(df))
    return df


def encode_genre(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode track_genre numerically; return df with track_genre_encoded and label mapping."""
    le = LabelEncoder()
    df = df.copy()
    df["track_genre_encoded"] = le.fit_transform(df[GENRE_COLUMN].astype(str))
    mapping = {int(i): label for i, label in enumerate(le.classes_)}
    return df, mapping


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    mult: float = 1.5,
) -> pd.DataFrame:
    """Remove rows where any column value is outside [Q1 - mult*IQR, Q3 + mult*IQR]."""
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - mult * iqr
        high = q3 + mult * iqr
        mask &= (df[col] >= low) & (df[col] <= high)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info("Outliers removed: %d rows (IQR mult=%.1f)", n_removed, mult)
    return df[mask].copy()


def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fit StandardScaler on columns and transform in place (copy)."""
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.input)

    df, genre_mapping = encode_genre(df)
    logger.info("Encoded track_genre -> %d labels", len(genre_mapping))

    outlier_cols = NUMERIC_FEATURE_COLUMNS + [TARGET_COLUMN]
    df = remove_outliers_iqr(df, outlier_cols, mult=args.iqr_mult)

    df = normalize_numeric(df, NUMERIC_FEATURE_COLUMNS)

    out_columns = NUMERIC_FEATURE_COLUMNS + ["track_genre_encoded", TARGET_COLUMN]
    out_path = out_dir / OUTPUT_FILENAME
    df[out_columns].to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)

    mapping_path = out_dir / MAPPING_FILENAME
    with open(mapping_path, "w") as f:
        json.dump(genre_mapping, f, indent=2)
    logger.info("Saved genre mapping to %s", mapping_path)


if __name__ == "__main__":
    main()
