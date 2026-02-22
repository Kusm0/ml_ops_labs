"""Pytest configuration and shared constants for Lab 4 tests."""

import os
from pathlib import Path

# Project root (tests/ is under project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data path: env DATA_PATH or default train.csv path (prepared data)
def get_data_path() -> str:
    return os.environ.get(
        "DATA_PATH", str(PROJECT_ROOT / "data" / "prepared" / "train.csv")
    )


# Artifacts produced by train_ci.py (in project root)
def get_artifact_dir() -> Path:
    return PROJECT_ROOT


REQUIRED_COLUMNS = {
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
    "track_genre_encoded",
    "popularity",
}
MIN_ROWS = 50
