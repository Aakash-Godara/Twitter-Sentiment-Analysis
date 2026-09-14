"""
Downloads the Sentiment140 dataset and copies it into data/raw/, verifying
its shape against the published schema before use.
"""
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

EXPECTED_ROWS = 1_600_000
EXPECTED_COLS = 6
COLUMN_NAMES = ["target", "id", "date", "query", "user", "text"]

RAW_DIR = Path(__file__).parent / "data" / "raw"
RAW_PATH = RAW_DIR / "sentiment140.csv"


def fetch_and_verify() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        cache_dir = Path(kagglehub.dataset_download("kazanova/sentiment140"))
        source_csv = next(cache_dir.glob("*.csv"))
        shutil.copy(source_csv, RAW_PATH)
        print(f"Copied {source_csv.name} -> {RAW_PATH}")
    else:
        print(f"{RAW_PATH} already exists, skipping download.")

    df = pd.read_csv(RAW_PATH, encoding="ISO-8859-1", header=None, names=COLUMN_NAMES)

    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLS), (
        f"Expected shape ({EXPECTED_ROWS}, {EXPECTED_COLS}), got {df.shape}. "
        "The dataset may have changed upstream."
    )
    assert set(df["target"].unique()) == {0, 4}, (
        f"Expected target values {{0, 4}} (negative/positive), "
        f"got {set(df['target'].unique())}."
    )

    print(f"Verified: {df.shape[0]:,} rows, {df.shape[1]} columns, "
          f"target values {sorted(df['target'].unique())}.")
    return RAW_PATH


if __name__ == "__main__":
    fetch_and_verify()
