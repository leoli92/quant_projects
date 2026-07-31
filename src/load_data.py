"""Load Home Credit application tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    COLUMNS_DESC_FILE,
    DATA_RAW,
    ID_COL,
    TEST_FILE,
    TRAIN_FILE,
)


def _read_csv(filename: str, data_dir: Path = DATA_RAW) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python src/download_data.py` first."
        )
    return pd.read_csv(path)


def load_train(data_dir: Path = DATA_RAW) -> pd.DataFrame:
    return _read_csv(TRAIN_FILE, data_dir)


def load_test(data_dir: Path = DATA_RAW) -> pd.DataFrame:
    return _read_csv(TEST_FILE, data_dir)


def load_column_descriptions(data_dir: Path = DATA_RAW) -> pd.DataFrame:
    return _read_csv(COLUMNS_DESC_FILE, data_dir)


def train_test_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "train_rows": len(train),
        "train_cols": train.shape[1],
        "test_rows": len(test),
        "test_cols": test.shape[1],
        "default_rate": float(train["TARGET"].mean()) if "TARGET" in train else None,
        "id_col": ID_COL,
    }
