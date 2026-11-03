"""
Spatial binning algorithms: Square (Spatial Hashing), Hexagonal (Axial), KD-Tree, Quadtree, and Voronoi.

This module provides high-speed 2D continuous molecule discretization and sparse count
matrix aggregation for spatial transcriptomics datasets.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.spatial import KDTree


def bin_square(x: np.ndarray, y: np.ndarray, resolution: float) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Bins 2D continuous points into a regular square grid of given resolution via spatial hashing.

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        resolution: Side length of each square bin.

    Returns:
        tuple[np.ndarray, tuple[float, float]]:
            - bin_coords: 2D array of shape (N, 2) containing integer grid indices (bin_x, bin_y).
            - offset: The (x_min, y_min) minimum bounds used to anchor the grid origin.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0:
        return np.empty((0, 2), dtype=int), (0.0, 0.0)

    x_min = float(np.min(x))
    y_min = float(np.min(y))

    # Use float64 arithmetic to preserve precision on extreme ranges and sub-nanometer values
    bin_x = np.floor((x.astype(np.float64) - x_min) / float(resolution)).astype(int)
    bin_y = np.floor((y.astype(np.float64) - y_min) / float(resolution)).astype(int)

    return np.column_stack((bin_x, bin_y)), (x_min, y_min)


def bin_hexagonal(x: np.ndarray, y: np.ndarray, resolution: float) -> np.ndarray:
    """
    Bins 2D points into a hexagonal grid where resolution defines the center-to-center distance.

    Uses pointy-topped hexagons with axial coordinate quantization:
      - R (outer radius) = resolution / sqrt(3)
      - Spacing between adjacent columns = resolution
      - Spacing between adjacent rows = 1.5 * R

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        resolution: Center-to-center distance between adjacent hexagons.

    Returns:
        np.ndarray: 2D array of shape (N, 2) containing integer axial coordinates (q, r).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0:
        return np.empty((0, 2), dtype=int)

    R = resolution / np.sqrt(3.0)

    # Transform Cartesian to fractional axial coordinates (q_frac, r_frac)
    q_frac = (np.sqrt(3.0) * x - y) / (3.0 * R)
    r_frac = (2.0 / 3.0) * y / R
    s_frac = -q_frac - r_frac

    # Round to nearest integer coordinates
    q_round = np.round(q_frac).astype(int)
    r_round = np.round(r_frac).astype(int)
    s_round = np.round(s_frac).astype(int)

    # Correct rounding to enforce q + r + s = 0 constraint
    dq = np.abs(q_round - q_frac)
    dr = np.abs(r_round - r_frac)
    ds = np.abs(s_round - s_frac)

    mask_q = (dq >= dr) & (dq >= ds)
    mask_r = (~mask_q) & (dr >= ds)

    q_round = np.where(mask_q, -r_round - s_round, q_round)
    r_round = np.where(mask_r, -q_round - s_round, r_round)

    return np.column_stack((q_round, r_round))


def bin_kdtree(x: np.ndarray, y: np.ndarray, resolution: float) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Bins 2D points into a grid by searching nearest active spatial centers using a KD-Tree.

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        resolution: Side length of each square bin.

    Returns:
        tuple[np.ndarray, tuple[float, float]]: (bin_coords, (x_min, y_min)).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return np.empty((0, 2), dtype=int), (0.0, 0.0)

    x_min = float(np.min(x))
    y_min = float(np.min(y))

    bin_x_approx = np.floor((x - x_min) / resolution).astype(int)
    bin_y_approx = np.floor((y - y_min) / resolution).astype(int)

    active_grid = np.unique(np.column_stack((bin_x_approx, bin_y_approx)), axis=0)

    centers = np.column_stack((
        x_min + (active_grid[:, 0] + 0.5) * resolution,
        y_min + (active_grid[:, 1] + 0.5) * resolution
    ))

    tree = KDTree(centers)
    points = np.column_stack((x, y))
    _, indices = tree.query(points)

    return active_grid[indices], (x_min, y_min)


def bin_quadtree(
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
    max_depth: int = 10
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Bins 2D points into a spatial grid using recursive Quadtree space partitioning.

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        resolution: Target cell size threshold for leaf nodes.
        max_depth: Maximum recursion depth of the Quadtree.

    Returns:
        tuple[np.ndarray, tuple[float, float]]: (bin_coords, (x_min, y_min)).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return np.empty((0, 2), dtype=int), (0.0, 0.0)

    x_min = float(np.min(x))
    y_min = float(np.min(y))
    x_max = float(np.max(x)) + 1e-6
    y_max = float(np.max(y)) + 1e-6

    points = np.column_stack((x, y))
    bin_coords = np.zeros((len(x), 2), dtype=int)

    def _split_node(indices: np.ndarray, bbox: tuple[float, float, float, float], depth: int) -> None:
        if len(indices) == 0:
            return
        bx_min, bx_max, by_min, by_max = bbox
        width = bx_max - bx_min
        height = by_max - by_min

        if width <= resolution or height <= resolution or depth >= max_depth:
            bx = np.floor((points[indices, 0] - x_min) / resolution).astype(int)
            by = np.floor((points[indices, 1] - y_min) / resolution).astype(int)
            bin_coords[indices] = np.column_stack((bx, by))
            return

        mid_x = (bx_min + bx_max) / 2.0
        mid_y = (by_min + by_max) / 2.0

        p = points[indices]
        q0 = indices[(p[:, 0] < mid_x) & (p[:, 1] < mid_y)]
        q1 = indices[(p[:, 0] >= mid_x) & (p[:, 1] < mid_y)]
        q2 = indices[(p[:, 0] < mid_x) & (p[:, 1] >= mid_y)]
        q3 = indices[(p[:, 0] >= mid_x) & (p[:, 1] >= mid_y)]

        _split_node(q0, (bx_min, mid_x, by_min, mid_y), depth + 1)
        _split_node(q1, (mid_x, bx_max, by_min, mid_y), depth + 1)
        _split_node(q2, (bx_min, mid_x, mid_y, by_max), depth + 1)
        _split_node(q3, (mid_x, bx_max, mid_y, by_max), depth + 1)

    _split_node(np.arange(len(x)), (x_min, x_max, y_min, y_max), 0)
    return bin_coords, (x_min, y_min)


def bin_square_memmap(
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
    chunk_size: int = 100000
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Bins 2D points into a square grid using chunked execution for memory-constrained workflows.

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        resolution: Side length of each square bin.
        chunk_size: Maximum chunk size for memory-bounded processing.

    Returns:
        tuple[np.ndarray, tuple[float, float]]: (bin_coords, (x_min, y_min)).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n == 0:
        return np.empty((0, 2), dtype=int), (0.0, 0.0)

    x_min = float(np.min(x))
    y_min = float(np.min(y))

    bin_coords = np.empty((n, 2), dtype=int)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        bx = np.floor((x[start:end] - x_min) / resolution).astype(int)
        by = np.floor((y[start:end] - y_min) / resolution).astype(int)
        bin_coords[start:end] = np.column_stack((bx, by))

    return bin_coords, (x_min, y_min)


def benchmark_binning_algorithms(
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
    runs: int = 3
) -> pd.DataFrame:
    """
    Compares speed and memory tradeoffs between Spatial Hashing, KD-Tree, Quadtree, and Memmap Grid binning.

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        resolution: Bin resolution size.
        runs: Number of benchmark repetitions for averaging.

    Returns:
        pd.DataFrame: Table summarizing execution time (ms) and peak memory usage (MB) for each algorithm.
    """
    algorithms = {
        "Spatial Hashing (Grid)": lambda: bin_square(x, y, resolution),
        "KD-Tree Query": lambda: bin_kdtree(x, y, resolution),
        "Quadtree Decomposition": lambda: bin_quadtree(x, y, resolution),
        "Memmap Chunked Grid": lambda: bin_square_memmap(x, y, resolution)
    }

    results = []
    for name, func in algorithms.items():
        times = []
        peak_mems = []
        for _ in range(runs):
            tracemalloc.start()
            t0 = time.perf_counter()
            _ = func()
            t1 = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            times.append((t1 - t0) * 1000.0)  # ms
            peak_mems.append(peak / (1024.0 * 1024.0))  # MB

        avg_time = float(np.mean(times))
        avg_peak_mem = float(np.mean(peak_mems))
        results.append({
            "Algorithm": name,
            "Execution Time (ms)": round(avg_time, 3),
            "Peak Memory (MB)": round(avg_peak_mem, 4)
        })

    df_res = pd.DataFrame(results)
    baseline_time = df_res.loc[0, "Execution Time (ms)"]
    if baseline_time > 0:
        df_res["Relative Speedup"] = (baseline_time / df_res["Execution Time (ms)"]).round(2)
    else:
        df_res["Relative Speedup"] = 1.0

    return df_res


def get_bin_centers(
    unique_bins: np.ndarray,
    bin_type: Literal["square", "hex"],
    resolution: float,
    offset: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Computes Cartesian center coordinates (x, y) for unique active grid bins.

    Args:
        unique_bins: 2D array of shape (M, 2) of integer bin coordinates.
        bin_type: Type of grid: 'square' or 'hex'.
        resolution: Physical resolution (grid spacing).
        offset: The (x_min, y_min) offset for square bins. Required if bin_type is 'square'.

    Returns:
        np.ndarray: 2D array of shape (M, 2) holding Cartesian coordinates of bin centers.

    Raises:
        ValueError: If offset is missing for square bins or invalid bin_type is supplied.
    """
    if bin_type == "square":
        if offset is None:
            raise ValueError("Offset (x_min, y_min) is required for square bin centers.")
        x_min, y_min = offset
        x_centers = (unique_bins[:, 0] + 0.5) * resolution + x_min
        y_centers = (unique_bins[:, 1] + 0.5) * resolution + y_min
        return np.column_stack((x_centers, y_centers))

    elif bin_type == "hex":
        R = resolution / np.sqrt(3.0)
        q = unique_bins[:, 0]
        r = unique_bins[:, 1]
        x_centers = R * (np.sqrt(3.0) * q + (np.sqrt(3.0) / 2.0) * r)
        y_centers = R * (1.5 * r)
        return np.column_stack((x_centers, y_centers))

    else:
        raise ValueError(f"Unknown bin type: {bin_type}. Choose 'square' or 'hex'.")


def build_sparse_matrix(
    bin_coords: np.ndarray,
    labels: np.ndarray,
    matrix_format: Literal["csr", "csc"] = "csr"
) -> tuple[csr_matrix | csc_matrix, np.ndarray, np.ndarray]:
    """
    Aggregates molecule counts within active bins and exports to a SciPy sparse matrix.

    Args:
        bin_coords: 2D array of shape (N, 2) containing integer bin coordinates for each molecule.
        labels: 1D array of label names (strings or categories) for each molecule.
        matrix_format: Format of sparse matrix: 'csr' or 'csc'.

    Returns:
        tuple[csr_matrix | csc_matrix, np.ndarray, np.ndarray]:
            - sparse_matrix: Matrix of shape (num_active_bins, num_unique_labels) containing count data.
            - unique_bins: 2D array of shape (num_active_bins, 2) of active bin coordinates.
            - unique_labels: 1D array of shape (num_unique_labels,) of sorted label names.
    """
    unique_bins, inverse_bins = np.unique(bin_coords, axis=0, return_inverse=True)
    unique_labels, inverse_labels = np.unique(labels, return_inverse=True)

    counts = np.ones(len(labels), dtype=int)
    coo = coo_matrix(
        (counts, (inverse_bins, inverse_labels)),
        shape=(len(unique_bins), len(unique_labels)),
        dtype=int
    )

    if matrix_format.lower() == "csr":
        sparse_matrix = coo.tocsr()
    elif matrix_format.lower() == "csc":
        sparse_matrix = coo.tocsc()
    else:
        raise ValueError(f"Unsupported sparse matrix format: {matrix_format}. Use 'csr' or 'csc'.")

    return sparse_matrix, unique_bins, unique_labels


def bin_voronoi(
    x: np.ndarray,
    y: np.ndarray,
    num_bins: int = 50
) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """
    Spatially bins 2D points into adaptive Voronoi regions adapting dynamically to local molecule density.

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        num_bins: Target number of adaptive Voronoi region seeds.

    Returns:
        tuple[np.ndarray, tuple[float, float], np.ndarray]:
            - bin_indices: 1D array of shape (N,) with assigned Voronoi seed region index for each point.
            - offset: (x_min, y_min) minimum bounds.
            - seed_centers: 2D array of shape (num_bins, 2) holding Voronoi seed center coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return np.empty(0, dtype=int), (0.0, 0.0), np.empty((0, 2))

    x_min = float(np.min(x))
    y_min = float(np.min(y))

    points = np.column_stack((x, y))
    k = min(num_bins, len(points))

    from scipy.cluster.vq import kmeans2
    seed_centers, _ = kmeans2(points.astype(np.float64), k=k, minit="points")

    tree = KDTree(seed_centers)
    _, bin_indices = tree.query(points)

    return bin_indices, (x_min, y_min), seed_centers


def compute_square_bins(
    df: pd.DataFrame,
    resolution: float,
    min_counts: int = 1
) -> tuple[csr_matrix, np.ndarray]:
    """
    Spatially bins continuous coordinate points into discrete grid blocks and constructs sparse count matrix.

    Args:
        df: DataFrame containing 'x', 'y', and 'label' columns.
        resolution: Physical width of each square bin.
        min_counts: Minimum transcript count threshold per bin to eliminate background noise.

    Returns:
        tuple[csr_matrix, np.ndarray]:
            - sparse_counts: Compressed Sparse Row matrix of shape (num_bins, num_labels).
            - bin_coords: Array of shape (num_bins, 2) holding center (x, y) coordinates for each valid bin.
    """
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    if len(x) == 0:
        num_labels = len(df["label"].cat.categories) if hasattr(df["label"], "cat") else 0
        return csr_matrix((0, num_labels), dtype=np.int32), np.empty((0, 2))

    label_codes = df["label"].cat.codes.to_numpy()
    num_labels = len(df["label"].cat.categories)

    x_min, y_min = float(x.min()), float(y.min())

    grid_x = np.floor((x - x_min) / resolution).astype(np.int32)
    grid_y = np.floor((y - y_min) / resolution).astype(np.int32)

    max_grid_x = int(grid_x.max()) + 1
    bin_ids = grid_y.astype(np.int64) * max_grid_x + grid_x.astype(np.int64)

    unique_bins, inverse_indices = np.unique(bin_ids, return_inverse=True)
    num_unique_bins = len(unique_bins)

    unique_grid_y = unique_bins // max_grid_x
    unique_grid_x = unique_bins % max_grid_x

    bin_x_centers = x_min + (unique_grid_x + 0.5) * resolution
    bin_y_centers = y_min + (unique_grid_y + 0.5) * resolution
    bin_coords = np.column_stack((bin_x_centers, bin_y_centers))

    sparse_counts = csr_matrix(
        (np.ones_like(inverse_indices), (inverse_indices, label_codes)),
        shape=(num_unique_bins, num_labels),
        dtype=np.int32
    )

    if min_counts > 1:
        bin_totals = np.array(sparse_counts.sum(axis=1)).flatten()
        keep_mask = bin_totals >= min_counts
        sparse_counts = sparse_counts[keep_mask]
        bin_coords = bin_coords[keep_mask]

    return sparse_counts, bin_coords
