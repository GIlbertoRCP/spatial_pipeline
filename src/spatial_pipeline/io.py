"""
Loader for generic labeled 2D points (`x, y, label`), the input of the engine-level binning,
graph and out-of-core functions. Photo tables have their own loaders in `photos`.
"""

from __future__ import annotations

import os

import pandas as pd


def load_coordinates(file_path: str) -> pd.DataFrame:
    """
    Loads labeled 2D points from a CSV or Parquet file with 'x', 'y' and 'label' columns.

    Args:
        file_path: Path to the CSV/Parquet file.

    Returns:
        pd.DataFrame: DataFrame with columns ['x', 'y', 'label'] (float64, float64, category).

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    columns = ["x", "y", "label"]
    if file_path.endswith(".parquet"):
        pdf = pd.read_parquet(file_path, columns=columns)
    else:
        pdf = pd.read_csv(file_path, usecols=columns)

    pdf["x"] = pdf["x"].astype("float64")
    pdf["y"] = pdf["y"].astype("float64")
    pdf["label"] = pdf["label"].astype("category")
    return pdf
