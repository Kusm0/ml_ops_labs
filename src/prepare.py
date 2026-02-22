"""
Data preparation stage for DVC pipeline: raw CSV → train.csv + test.csv.

Reads raw data (data/raw/dataset.csv), reuses preprocessing from preprocess.py
(encoding, IQR outliers, normalization), then splits into train/test and writes
to output_dir (e.g. data/prepared/). Used by dvc.yaml stage 'prepare'.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow importing preprocess when running as script from project root (e.g. dvc repro)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import (
    GENRE_COLUMN,
    MAPPING_FILENAME,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    encode_genre,
    load_raw,
    normalize_numeric,
    remove_outliers_iqr,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "prepared"

OUT_COLUMNS = NUMERIC_FEATURE_COLUMNS + ["track_genre_encoded", TARGET_COLUMN]


def parse_args() -> argparse.Namespace:
    """Parse CLI: input CSV and output directory (e.g. for dvc.yaml)."""
    parser = argparse.ArgumentParser(
        description="Prepare data: raw CSV → train.csv, test.csv"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=str(DEFAULT_RAW_PATH),
        help="Path to raw CSV",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for train.csv and test.csv",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction for test set",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for split",
    )
    parser.add_argument(
        "--iqr_mult",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier removal",
    )
    return parser.parse_args()


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

    train_df, test_df = train_test_split(
        df[OUT_COLUMNS],
        test_size=args.test_size,
        random_state=args.random_state,
    )
    logger.info("Split: train=%d test=%d", len(train_df), len(test_df))

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved %s and %s", train_path, test_path)

    mapping_path = out_dir / MAPPING_FILENAME
    with open(mapping_path, "w") as f:
        json.dump(genre_mapping, f, indent=2)
    logger.info("Saved genre mapping to %s", mapping_path)


if __name__ == "__main__":
    main()
