"""
Chunked Out-of-Core Processing and Memory-Constrained Spatial Pipelines.

This module provides spatial tiling, boundary halo exchange, Dask distributed DataFrame binning,
and Zarr streaming matrix computations for spatial transcriptomics datasets exceeding host RAM.
"""

from __future__ import annotations

import os
import shutil
import time
import tracemalloc
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import KDTree

try:
    import resource
except ImportError:
    resource = None  # type: ignore

try:
    import dask.array as da  # type: ignore
    import dask.dataframe as dd  # type: ignore
    HAS_DASK = True
except ImportError:
    da = None  # type: ignore
    dd = None  # type: ignore
    HAS_DASK = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    zarr = None  # type: ignore
    HAS_ZARR = False


def check_memory_limit(limit_mb: float | None = None) -> None:
    """
    Checks current traced RAM usage against specified limit_mb (in Megabytes).

    Args:
        limit_mb: Memory ceiling in Megabytes.

    Raises:
        MemoryError: If peak allocated memory exceeds limit_mb.
    """
    if limit_mb is None or limit_mb <= 0:
        return
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    if peak_mb > limit_mb:
        raise MemoryError(
            f"Strict memory limit exceeded: Peak allocated RAM is {peak_mb:.2f} MB, "
            f"which exceeds artificial limit of {limit_mb:.2f} MB."
        )


@contextmanager
def enforce_memory_limit(limit_mb: float | None = None) -> Generator[None, None, None]:
    """
    Context manager enforcing an artificial memory limit on Linux (via resource.setrlimit)
    and cross-platform via tracemalloc monitoring.

    Args:
        limit_mb: Maximum memory threshold in Megabytes.
    """
    old_limit = None
    if limit_mb is not None and limit_mb > 0:
        bytes_limit = int(limit_mb * 1024 * 1024)
        if resource is not None and hasattr(resource, "RLIMIT_AS"):
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                old_limit = (soft, hard)
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, hard))
            except Exception:
                pass
    try:
        yield
    finally:
        if old_limit is not None and resource is not None and hasattr(resource, "RLIMIT_AS"):
            try:
                resource.setrlimit(resource.RLIMIT_AS, old_limit)
            except Exception:
                pass
        if limit_mb is not None:
            check_memory_limit(limit_mb)


@contextmanager
def profile_memory() -> Generator[dict[str, float], None, None]:
    """
    Context manager to track exact peak RAM usage (in bytes and MB) during an execution block.

    Yields:
        dict[str, float]: Dictionary with metrics: 'current_bytes', 'peak_bytes', 'peak_mb', 'elapsed_sec'.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    metrics = {"current_bytes": 0.0, "peak_bytes": 0.0, "peak_mb": 0.0, "elapsed_sec": 0.0}
    try:
        yield metrics
    finally:
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["current_bytes"] = float(current)
        metrics["peak_bytes"] = float(peak)
        metrics["peak_mb"] = float(peak / (1024.0 * 1024.0))
        metrics["elapsed_sec"] = float(t1 - t0)


def partition_dataset_to_tiles(file_path: str, tile_size: float, partition_dir: str) -> None:
    """
    Partitions a large spatial transcriptomics CSV file into spatial tiles on disk.

    Args:
        file_path: Path to the raw transcript coordinates CSV.
        tile_size: The physical dimensions (width/height) of each spatial tile.
        partition_dir: Directory where tile partition CSV files will be stored.
    """
    if os.path.exists(partition_dir):
        # Clean existing CSV partitions to prevent appending to stale data
        for f in os.listdir(partition_dir):
            if f.endswith((".csv", ".npy")):
                try:
                    os.remove(os.path.join(partition_dir, f))
                except OSError:
                    pass
    else:
        os.makedirs(partition_dir, exist_ok=True)

    print(f"[*] Partitioning dataset {file_path} into {tile_size} um spatial tiles...")

    global_labels_set: set[str] = set()
    x_min = float("inf")
    y_min = float("inf")

    # Process in row-chunks to minimize memory consumption
    for chunk in pd.read_csv(file_path, chunksize=100000):
        # Update offsets globally
        x_min = min(x_min, chunk["x"].min())
        y_min = min(y_min, chunk["y"].min())

        # Collect all unique labels in the dataset
        global_labels_set.update(chunk["label"].unique())

        # Ensure coordinates are float and calculate tile grid coordinates
        tx = np.floor(chunk["x"].to_numpy() / tile_size).astype(int)
        ty = np.floor(chunk["y"].to_numpy() / tile_size).astype(int)

        # Group and append each sub-chunk to its corresponding tile file
        for (tile_x, tile_y), group in chunk.groupby([tx, ty]):
            tile_file = os.path.join(partition_dir, f"tile_{tile_x}_{tile_y}.csv")
            group.to_csv(
                tile_file,
                mode="a",
                index=False,
                header=not os.path.exists(tile_file)
            )

    # Save global offset and label list to the partition directory
    global_labels = np.array(sorted(global_labels_set))
    np.save(os.path.join(partition_dir, "labels.npy"), global_labels)
    np.save(os.path.join(partition_dir, "offset.npy"), np.array([x_min, y_min]))

    print(f"[+] Dataset partitioning complete. Saved in directory: {partition_dir}")


def load_tile_with_halo(
    tile_x: int,
    tile_y: int,
    tile_size: float,
    halo_width: float,
    partition_dir: str
) -> pd.DataFrame:
    """
    Loads coordinate data for a target tile and its 8 neighboring tiles,
    filtering to the expanded bounding box (core + halo).

    Args:
        tile_x: Grid column coordinate of the target tile.
        tile_y: Grid row coordinate of the target tile.
        tile_size: Physical width/height of a tile.
        halo_width: Width of the boundary exchange halo (typically max_dist).
        partition_dir: Directory containing tile partition CSV files.

    Returns:
        pd.DataFrame: DataFrame of transcripts inside core + halo boundary with source tile columns.
    """
    dfs = []
    # Search the 3x3 grid of tiles centered at (tile_x, tile_y)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = tile_x + dx
            ny = tile_y + dy
            tile_file = os.path.join(partition_dir, f"tile_{nx}_{ny}.csv")
            if os.path.exists(tile_file):
                sub_df = pd.read_csv(tile_file)
                sub_df["src_tile_x"] = nx
                sub_df["src_tile_y"] = ny
                dfs.append(sub_df)

    if not dfs:
        return pd.DataFrame(columns=["x", "y", "label", "src_tile_x", "src_tile_y"])

    df_combined = pd.concat(dfs, ignore_index=True)

    # Establish expanded spatial bounds (core tile box + halo border)
    x_min = tile_x * tile_size - halo_width
    x_max = (tile_x + 1) * tile_size + halo_width
    y_min = tile_y * tile_size - halo_width
    y_max = (tile_y + 1) * tile_size + halo_width

    # Filter molecules to bounds
    in_bounds = (
        (df_combined["x"] >= x_min) & (df_combined["x"] <= x_max) &
        (df_combined["y"] >= y_min) & (df_combined["y"] <= y_max)
    )

    df_filtered = df_combined[in_bounds].copy()
    return df_filtered


def process_tile_out_of_core(
    tile_x: int,
    tile_y: int,
    tile_size: float,
    resolution: float,
    max_dist: float,
    partition_dir: str,
    tmp_out_dir: str
) -> None:
    """
    Bins and generates sub-graphs for a single spatial tile out-of-core.
    Active bins are classified into 'owned' or 'halo' to avoid duplicate node and edge artifacts.

    Args:
        tile_x: Tile grid column index.
        tile_y: Tile grid row index.
        tile_size: Physical tile side length.
        resolution: Spatial bin width.
        max_dist: Maximum neighbor radius distance.
        partition_dir: Directory containing tile partition CSVs.
        tmp_out_dir: Output scratch directory for intermediate binned matrices.
    """
    labels_path = os.path.join(partition_dir, "labels.npy")
    offset_path = os.path.join(partition_dir, "offset.npy")
    if not os.path.exists(labels_path) or not os.path.exists(offset_path):
        raise FileNotFoundError("Global partition metadata not found. Run partitioning first.")
    unique_labels = np.load(labels_path, allow_pickle=True)
    x_min, y_min = np.load(offset_path)

    # 1. Ingest core coordinates + halo
    df = load_tile_with_halo(tile_x, tile_y, tile_size, max_dist, partition_dir)
    if len(df) == 0:
        return

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # 2. Bin molecules relative to global offset
    bin_x = np.floor((x - x_min) / resolution).astype(np.int32)
    bin_y = np.floor((y - y_min) / resolution).astype(np.int32)
    bin_coords = np.column_stack((bin_x, bin_y))

    unique_bins, inverse_indices = np.unique(bin_coords, axis=0, return_inverse=True)

    bin_centers = np.column_stack((
        x_min + (unique_bins[:, 0] + 0.5) * resolution,
        y_min + (unique_bins[:, 1] + 0.5) * resolution
    ))

    # 3. Classify owned bins
    cx = bin_centers[:, 0]
    cy = bin_centers[:, 1]

    home_tile_x = np.floor(cx / tile_size).astype(np.int32)
    home_tile_y = np.floor(cy / tile_size).astype(np.int32)

    home_active = np.array([
        os.path.exists(os.path.join(partition_dir, f"tile_{htx}_{hty}.csv"))
        for htx, hty in zip(home_tile_x, home_tile_y)
    ])

    src_tile_x = df["src_tile_x"].to_numpy()
    src_tile_y = df["src_tile_y"].to_numpy()
    src_keys = src_tile_x.astype(np.int64) * 1000000 + src_tile_y.astype(np.int64)

    min_src_keys = np.full(len(unique_bins), 999999999, dtype=np.int64)
    np.minimum.at(min_src_keys, inverse_indices, src_keys)

    u_key = int(tile_x) * 1000000 + int(tile_y)

    is_owned = (
        (home_active & (home_tile_x == tile_x) & (home_tile_y == tile_y)) |
        (~home_active & (min_src_keys == u_key))
    )

    num_owned = np.sum(is_owned)
    if num_owned == 0:
        return

    owned_bins = unique_bins[is_owned]
    owned_transcripts_mask = is_owned[inverse_indices]
    df_owned = df[owned_transcripts_mask]

    _, inverse_owned = np.unique(bin_coords[owned_transcripts_mask], axis=0, return_inverse=True)

    label_series = pd.Categorical(df_owned["label"], categories=unique_labels)
    label_codes = label_series.codes

    counts = np.ones(len(df_owned), dtype=np.int32)
    sparse_counts = coo_matrix(
        (counts, (inverse_owned, label_codes)),
        shape=(len(owned_bins), len(unique_labels)),
        dtype=np.int32
    ).tocsr()

    # 4. Local adjacency graph with KDTree
    tree = KDTree(bin_centers)
    neighbor_pairs = tree.query_pairs(r=max_dist, output_type="ndarray")

    if len(neighbor_pairs) > 0:
        i_owned = is_owned[neighbor_pairs[:, 0]]
        j_owned = is_owned[neighbor_pairs[:, 1]]
        valid_edges_mask = i_owned | j_owned
        tile_edges = neighbor_pairs[valid_edges_mask]

        global_edges = np.column_stack((
            unique_bins[tile_edges[:, 0]],
            unique_bins[tile_edges[:, 1]]
        ))
    else:
        global_edges = np.empty((0, 4), dtype=np.int32)

    os.makedirs(tmp_out_dir, exist_ok=True)
    prefix = f"{tile_x}_{tile_y}"

    scipy.sparse.save_npz(os.path.join(tmp_out_dir, f"counts_{prefix}.npz"), sparse_counts)
    np.save(os.path.join(tmp_out_dir, f"bins_{prefix}.npy"), owned_bins)
    np.save(os.path.join(tmp_out_dir, f"labels_{prefix}.npy"), unique_labels)
    np.save(os.path.join(tmp_out_dir, f"edges_{prefix}.npy"), global_edges)
    np.save(os.path.join(tmp_out_dir, "offset.npy"), np.array([x_min, y_min]))


def stitch_out_of_core_results(
    tmp_out_dir: str,
    resolution: float
) -> tuple[csr_matrix, np.ndarray, np.ndarray, csr_matrix, csr_matrix]:
    """
    Stitches disjoint count matrices, coordinates, and boundary edges from disk,
    resolving cross-boundary neighbors and returning globally unified sparse matrices.

    Args:
        tmp_out_dir: Directory containing intermediate tile npz/npy artifacts.
        resolution: Spatial bin size.

    Returns:
        tuple[csr_matrix, np.ndarray, np.ndarray, csr_matrix, csr_matrix]:
            (global_expression, global_centers, global_labels, adjacency_matrix, laplacian_matrix).
    """
    all_bins = []
    all_counts = []
    global_labels = None

    for f in os.listdir(tmp_out_dir):
        if f.startswith("bins_") and f.endswith(".npy"):
            prefix = f[5:-4]
            bins = np.load(os.path.join(tmp_out_dir, f))
            counts = scipy.sparse.load_npz(os.path.join(tmp_out_dir, f"counts_{prefix}.npz"))
            labels = np.load(os.path.join(tmp_out_dir, f"labels_{prefix}.npy"), allow_pickle=True)

            all_bins.append(bins)
            all_counts.append(counts)
            if global_labels is None:
                global_labels = labels

    if not all_bins:
        raise ValueError("No intermediate results found to stitch.")

    global_bins = np.concatenate(all_bins, axis=0)
    global_expression = scipy.sparse.vstack(all_counts).tocsr()

    offset_path = os.path.join(tmp_out_dir, "offset.npy")
    if os.path.exists(offset_path):
        x_min, y_min = np.load(offset_path)
        global_centers = np.column_stack((
            x_min + (global_bins[:, 0] + 0.5) * resolution,
            y_min + (global_bins[:, 1] + 0.5) * resolution
        ))
    else:
        global_centers = (global_bins + 0.5) * resolution

    all_edges = []
    for f in os.listdir(tmp_out_dir):
        if f.startswith("edges_") and f.endswith(".npy"):
            edges = np.load(os.path.join(tmp_out_dir, f))
            if len(edges) > 0:
                all_edges.append(edges)

    if all_edges:
        global_edges = np.concatenate(all_edges, axis=0)
        mask = (global_edges[:, 0] > global_edges[:, 2]) | (
            (global_edges[:, 0] == global_edges[:, 2]) & (global_edges[:, 1] > global_edges[:, 3])
        )
        temp = global_edges[mask, 0:2].copy()
        global_edges[mask, 0:2] = global_edges[mask, 2:4]
        global_edges[mask, 2:4] = temp
        unique_edges = np.unique(global_edges, axis=0)
    else:
        unique_edges = np.empty((0, 4), dtype=np.int32)

    num_bins = len(global_bins)

    if len(unique_edges) > 0:
        gx_min = np.min(global_bins[:, 0])
        gy_min = np.min(global_bins[:, 1])
        stride = int(np.max(global_bins[:, 1]) - gy_min + 2)

        bin_keys = (global_bins[:, 0] - gx_min) * stride + (global_bins[:, 1] - gy_min)
        sort_idx = np.argsort(bin_keys)
        sorted_keys = bin_keys[sort_idx]

        edge_A_keys = (unique_edges[:, 0] - gx_min) * stride + (unique_edges[:, 1] - gy_min)
        edge_B_keys = (unique_edges[:, 2] - gx_min) * stride + (unique_edges[:, 3] - gy_min)

        idx_A = np.searchsorted(sorted_keys, edge_A_keys)
        idx_B = np.searchsorted(sorted_keys, edge_B_keys)

        row_indices = sort_idx[idx_A]
        col_indices = sort_idx[idx_B]

        rows = np.concatenate([row_indices, col_indices])
        cols = np.concatenate([col_indices, row_indices])
        data = np.ones_like(rows, dtype=np.float32)

        adjacency_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(num_bins, num_bins),
            dtype=np.float32
        )
    else:
        adjacency_matrix = csr_matrix((num_bins, num_bins), dtype=np.float32)

    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degree_matrix = csr_matrix(
        (degrees, (np.arange(num_bins), np.arange(num_bins))),
        shape=(num_bins, num_bins),
        dtype=np.float32
    )
    laplacian_matrix = degree_matrix - adjacency_matrix

    return global_expression, global_centers, global_labels, adjacency_matrix, laplacian_matrix


def _process_single_tile_wrapper(args: tuple[int, int, float, float, float, str, str]) -> None:
    """Helper worker for multi-processing spatial tile out-of-core pipeline."""
    tx, ty, tile_size, resolution, max_dist, partition_dir, tmp_out_dir = args
    process_tile_out_of_core(tx, ty, tile_size, resolution, max_dist, partition_dir, tmp_out_dir)


def process_tiles_parallel_out_of_core(
    tiles: list[tuple[int, int]],
    tile_size: float,
    resolution: float,
    max_dist: float,
    partition_dir: str,
    tmp_out_dir: str,
    n_jobs: int = -1
) -> None:
    """
    Processes spatial tiles concurrently using Python worker pools,
    bypassing GIL limitations for CPU-bound spatial sub-graph construction.

    Args:
        tiles: List of (tile_x, tile_y) integer tuples.
        tile_size: Tile width/height.
        resolution: Spatial bin width.
        max_dist: Maximum radius distance.
        partition_dir: Directory containing tile CSV files.
        tmp_out_dir: Output scratch directory.
        n_jobs: Parallel worker count (-1 uses all available cores).
    """
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, int(n_jobs))

    if n_jobs == 1 or len(tiles) <= 1:
        for tx, ty in tiles:
            process_tile_out_of_core(tx, ty, tile_size, resolution, max_dist, partition_dir, tmp_out_dir)
    else:
        tasks = [
            (tx, ty, tile_size, resolution, max_dist, partition_dir, tmp_out_dir)
            for tx, ty in tiles
        ]
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                list(executor.map(_process_single_tile_wrapper, tasks))
        except (RuntimeError, OSError, TypeError):
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                list(executor.map(_process_single_tile_wrapper, tasks))


def profile_out_of_core_execution(
    file_path: str,
    tile_size: float,
    resolution: float,
    max_dist: float,
    partition_dir: str = "tmp_partition_profile",
    tmp_out_dir: str = "tmp_out_profile",
    n_jobs: int = -1,
    limit_mb: float | None = None
) -> dict[str, Any]:
    """
    Profiles exact peak RAM usage (bytes and MB) and runtime across partition,
    tile processing, and stitching phases in out-of-core operations.

    Args:
        file_path: Path to coordinate CSV.
        tile_size: Spatial tile size.
        resolution: Spatial binning resolution.
        max_dist: Maximum neighbor radius.
        partition_dir: Directory for tile CSV partitions.
        tmp_out_dir: Directory for intermediate matrices.
        n_jobs: Worker pool size.
        limit_mb: Optional memory limit threshold.

    Returns:
        dict[str, Any]: Detailed execution profile report.
    """
    report: dict[str, Any] = {}
    partition_dir = os.path.abspath(partition_dir)
    tmp_out_dir = os.path.abspath(tmp_out_dir)

    with enforce_memory_limit(limit_mb):
        with profile_memory() as p_stats:
            partition_dataset_to_tiles(file_path, tile_size, partition_dir)
        report["partitioning"] = p_stats

        tiles = []
        if os.path.exists(partition_dir):
            for f in os.listdir(partition_dir):
                if f.startswith("tile_") and f.endswith(".csv"):
                    parts = f[5:-4].split("_")
                    tiles.append((int(parts[0]), int(parts[1])))

        with profile_memory() as proc_stats:
            process_tiles_parallel_out_of_core(
                tiles, tile_size, resolution, max_dist, partition_dir, tmp_out_dir, n_jobs=n_jobs
            )
        report["tile_processing"] = proc_stats

        with profile_memory() as stitch_stats:
            _ = stitch_out_of_core_results(tmp_out_dir, resolution)
        report["stitching"] = stitch_stats

    report["summary"] = {
        "num_tiles": float(len(tiles)),
        "max_stage_peak_mb": max(
            p_stats["peak_mb"],
            proc_stats["peak_mb"],
            stitch_stats["peak_mb"]
        ),
        "total_elapsed_sec": (
            p_stats["elapsed_sec"] +
            proc_stats["elapsed_sec"] +
            stitch_stats["elapsed_sec"]
        )
    }

    try:
        if os.path.exists(partition_dir):
            shutil.rmtree(partition_dir)
        if os.path.exists(tmp_out_dir):
            shutil.rmtree(tmp_out_dir)
    except OSError:
        pass

    return report


def dask_bin_transcripts(
    file_or_df: str | pd.DataFrame,
    resolution: float,
    min_counts: int = 1,
    blocksize: str | int = "64MB"
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    Performs chunked, out-of-core spatial binning using Dask Distributed DataFrames.

    Ideal for massive CSV datasets that cannot fit in available RAM.

    Args:
        file_or_df: Path to CSV coordinate file or pandas DataFrame.
        resolution: Spatial bin grid width.
        min_counts: Minimum transcript count threshold per bin.
        blocksize: Partition chunk size for Dask CSV reading (e.g. '64MB').

    Returns:
        tuple[csr_matrix, np.ndarray, np.ndarray]: (sparse_counts, bin_centers, unique_labels).
    """
    if not HAS_DASK or dd is None:
        raise ImportError("Dask is required. Install with `pip install 'dask[dataframe]'`.")

    if isinstance(file_or_df, str):
        ddf = dd.read_csv(file_or_df, usecols=["x", "y", "label"], blocksize=blocksize)
    else:
        ddf = dd.from_pandas(file_or_df, npartitions=max(1, len(file_or_df) // 50000))

    # Compute global minimum coordinate offset
    x_min = float(ddf["x"].min().compute())
    y_min = float(ddf["y"].min().compute())

    # Map coordinates to integer bin IDs
    ddf["bin_x"] = ((ddf["x"] - x_min) / resolution).astype(int)
    ddf["bin_y"] = ((ddf["y"] - y_min) / resolution).astype(int)

    # Distributed groupby aggregation of counts
    grouped = ddf.groupby(["bin_x", "bin_y", "label"]).size().reset_index()
    grouped.columns = ["bin_x", "bin_y", "label", "count"]
    agg_df = grouped.compute()

    if len(agg_df) == 0:
        return csr_matrix((0, 0), dtype=np.int32), np.empty((0, 2), dtype=np.float32), np.array([])

    unique_bins_2d = agg_df[["bin_x", "bin_y"]].drop_duplicates().to_numpy()
    unique_labels = np.array(sorted(agg_df["label"].unique()))

    # Build coordinate and label maps
    bin_map = {tuple(b): i for i, b in enumerate(unique_bins_2d)}
    label_map = {g: j for j, g in enumerate(unique_labels)}

    row_indices = np.array([bin_map[(r.bin_x, r.bin_y)] for r in agg_df.itertuples()], dtype=np.int32)
    col_indices = np.array([label_map[r.label] for r in agg_df.itertuples()], dtype=np.int32)
    counts = agg_df["count"].to_numpy(dtype=np.int32)

    sparse_matrix = coo_matrix(
        (counts, (row_indices, col_indices)),
        shape=(len(unique_bins_2d), len(unique_labels)),
        dtype=np.int32
    ).tocsr()

    bin_centers = np.column_stack((
        x_min + (unique_bins_2d[:, 0] + 0.5) * resolution,
        y_min + (unique_bins_2d[:, 1] + 0.5) * resolution
    )).astype(np.float32)

    if min_counts > 1:
        total_counts = np.array(sparse_matrix.sum(axis=1)).flatten()
        keep_mask = total_counts >= min_counts
        sparse_matrix = sparse_matrix[keep_mask]
        bin_centers = bin_centers[keep_mask]

    return sparse_matrix, bin_centers, unique_labels


def dask_compute_spatial_metrics(
    zarr_store_path: str,
    chunk_size: int = 10000
) -> dict[str, np.ndarray]:
    """
    Computes out-of-core label counts statistics (mean, variance, sparsity)
    across a chunked Zarr hierarchy using Dask Arrays.

    Args:
        zarr_store_path: Path to .zarr dataset store on disk.
        chunk_size: Observation chunk size for Dask chunking.

    Returns:
        dict[str, np.ndarray]: Dictionary containing:
            - 'mean': 1D array of mean expression per label.
            - 'var': 1D array of variance per label.
            - 'sparsity': 1D array of fractional non-zero bins per label.
    """
    if not HAS_DASK or da is None:
        raise ImportError("Dask is required. Install with `pip install dask`.")

    if not HAS_ZARR or zarr is None:
        raise ImportError("Zarr is required. Install with `pip install zarr`.")

    root = zarr.open_group(zarr_store_path, mode="r")
    x_node = root.get("X")
    if x_node is None:
        x_node = root.get("raw/X") or root

    darr = da.from_zarr(x_node, chunks=(chunk_size, x_node.shape[1]))

    mean_da = darr.mean(axis=0)
    var_da = darr.var(axis=0)
    sparsity_da = (darr > 0).mean(axis=0)

    mean_res, var_res, sparsity_res = da.compute(mean_da, var_da, sparsity_da)

    return {
        "mean": np.asarray(mean_res),
        "var": np.asarray(var_res),
        "sparsity": np.asarray(sparsity_res)
    }
