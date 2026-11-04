"""Synthetic labeled points (`x, y, label`) for engine-level tests (binning, tiling, Dask)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def write_synthetic_points(file_path: str, num_points: int = 1000, seed: int = 42) -> None:
    """
    Writes a CSV of `x, y, label` points in a 10..500 square: half uniform background, half in
    five Gaussian clusters, with labels drawn from five names.

    Args:
        file_path: Target CSV path.
        num_points: Number of points to write.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    n_back = num_points // 2
    n_clust = num_points - n_back

    background = rng.uniform(10.0, 500.0, (n_back, 2))
    centers = rng.uniform(50.0, 450.0, (5, 2))
    clustered = centers[rng.integers(0, 5, n_clust)] + rng.normal(0.0, 15.0, (n_clust, 2))
    xy = np.clip(np.vstack([background, clustered]), 10.0, 500.0)

    pd.DataFrame({
        "x": xy[:, 0].astype(np.float32),
        "y": xy[:, 1].astype(np.float32),
        "label": rng.choice(["elk", "mallard", "columbine", "aspen", "bighorn"], size=num_points),
    }).to_csv(file_path, index=False)
