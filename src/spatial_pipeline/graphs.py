"""
Spatial graph generation, GPU acceleration (RAPIDS cuGraph / cuML), and spatial statistics.

This module provides high-performance graph construction routines (k-NN, radius graphs,
Graph Laplacians, spectral decomposition, Moran's I, Geary's C, and Spatially Variable Label
detection) supporting both multi-core CPU parallel execution and RAPIDS GPU acceleration.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# GPU RAPIDS Ecosystem Detection
try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore
    HAS_CUPY = False

try:
    import cuml  # type: ignore
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors  # type: ignore
    HAS_CUML = True
except ImportError:
    cuml = None  # type: ignore
    cuNearestNeighbors = None  # type: ignore
    HAS_CUML = False

try:
    import cudf  # type: ignore
    import cugraph  # type: ignore
    HAS_CUGRAPH = True
except ImportError:
    cugraph = None  # type: ignore
    cudf = None  # type: ignore
    HAS_CUGRAPH = False


def is_gpu_available() -> bool:
    """
    Checks if a compatible NVIDIA GPU with RAPIDS cuML / cuGraph is available.

    Returns:
        bool: True if CUDA GPU acceleration libraries are importable and functional.
    """
    if not (HAS_CUPY and (HAS_CUML or HAS_CUGRAPH)):
        return False
    try:
        if cp is not None:
            _ = cp.cuda.runtime.getDeviceCount()
            return True
    except Exception:  # noqa: BLE001 - any CUDA runtime/driver failure means no usable GPU
        return False
    return False


def _query_batch_neighbors(args: tuple[np.ndarray, np.ndarray, float, int]) -> list[tuple[int, int]]:
    """
    Helper worker for multi-processing batch spatial query.

    Args:
        args: Tuple containing (batch_coords, all_coords, max_dist, start_idx).

    Returns:
        List of (i, j) node index pairs where distance <= max_dist and i < j.
    """
    batch_coords, all_coords, max_dist, start_idx = args
    tree = KDTree(all_coords)
    neighbors_list = tree.query_ball_point(batch_coords, r=max_dist)
    pairs: list[tuple[int, int]] = []
    for idx_in_batch, neighbors in enumerate(neighbors_list):
        i = start_idx + idx_in_batch
        for j in neighbors:
            if i < j:  # upper triangle pair
                pairs.append((i, j))
    return pairs


def compute_gpu_knn_graph_cuml(
    bin_coords: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean"
) -> csr_matrix:
    """
    Constructs a k-NN spatial graph using RAPIDS cuML on GPU.

    Args:
        bin_coords: Array of shape (N, 2) or (N, D) holding spatial coordinates.
        n_neighbors: Number of nearest neighbors per node.
        metric: Distance metric ('euclidean', 'sqeuclidean', 'minkowski').

    Returns:
        scipy.sparse.csr_matrix: Sparse adjacency matrix on host CPU memory.
    """
    if not HAS_CUML or cp is None:
        raise RuntimeError("RAPIDS cuML is required for GPU k-NN construction.")

    num_bins = bin_coords.shape[0]
    if num_bins == 0:
        return csr_matrix((0, 0), dtype=np.float32)

    k = min(n_neighbors + 1, num_bins)  # +1 to include self-loop, removed later
    d_coords = cp.asarray(bin_coords, dtype=cp.float32)
    nn_model = cuNearestNeighbors(n_neighbors=k, metric=metric, output_type="cupy")
    nn_model.fit(d_coords)
    _, indices = nn_model.kneighbors(d_coords)

    # Convert GPU indices to COO host arrays
    h_indices = cp.asnumpy(indices)
    row_indices = np.repeat(np.arange(num_bins, dtype=np.int32), k)
    col_indices = h_indices.flatten().astype(np.int32)

    # Exclude self loops
    mask = row_indices != col_indices
    rows = row_indices[mask]
    cols = col_indices[mask]

    # Symmetrize graph (undirected)
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    data = np.ones_like(sym_rows, dtype=np.float32)

    adj = csr_matrix(
        (data, (sym_rows, sym_cols)),
        shape=(num_bins, num_bins),
        dtype=np.float32
    )
    # Binary edge weights (0 or 1)
    adj.data = np.clip(adj.data, 0.0, 1.0)
    return adj


def build_knn_graph(
    bin_coords: np.ndarray,
    n_neighbors: int = 15,
    backend: Literal["auto", "cuml", "cugraph", "sklearn", "scipy"] = "auto",
    metric: str = "euclidean",
    compute_laplacian: bool = True
) -> tuple[csr_matrix, csr_matrix | None]:
    """
    Constructs a k-Nearest Neighbors (k-NN) spatial adjacency graph and Graph Laplacian.

    Supports automatic GPU acceleration via RAPIDS cuML / cuGraph with fallback to CPU.

    Args:
        bin_coords: Array of shape (N, 2) holding Cartesian (x, y) coordinates of spatial bins.
        n_neighbors: Number of nearest neighbors per node. Default is 15.
        backend: Execution backend ('auto', 'cuml', 'cugraph', 'sklearn', 'scipy').
        metric: Distance metric to use. Default is 'euclidean'.
        compute_laplacian: If True, computes and returns Graph Laplacian L = D - A.

    Returns:
        tuple[csr_matrix, csr_matrix | None]: (adjacency_matrix, laplacian_matrix).
    """
    bin_coords = np.asarray(bin_coords, dtype=np.float32)
    num_bins = bin_coords.shape[0]

    if num_bins == 0:
        empty_csr = csr_matrix((0, 0), dtype=np.float32)
        return empty_csr, (empty_csr if compute_laplacian else None)

    # Dispatch to GPU if available or explicitly requested
    use_gpu = False
    if backend in ("cuml", "cugraph"):
        if is_gpu_available():
            use_gpu = True
        else:
            logger.warning("GPU backend requested but RAPIDS cuML/cuGraph not available. Falling back to CPU.")
    elif backend == "auto" and is_gpu_available() and num_bins >= 5000:
        use_gpu = True

    if use_gpu:
        try:
            adj = compute_gpu_knn_graph_cuml(bin_coords, n_neighbors=n_neighbors, metric=metric)
        except Exception as e:  # noqa: BLE001 - GPU errors vary by RAPIDS version; always fall back to CPU
            logger.warning(f"GPU k-NN graph construction failed: {e}. Falling back to CPU.")
            use_gpu = False

    if not use_gpu:
        # High-performance CPU k-NN path using KDTree (fast, stable, no OpenMP fork issues)
        tree = KDTree(bin_coords)
        k = min(n_neighbors + 1, num_bins)
        _, indices = tree.query(bin_coords, k=k)
        
        # Ensure 2D array if k=1
        if indices.ndim == 1:
            indices = indices[:, None]

        row_indices = np.repeat(np.arange(num_bins, dtype=np.int32), k)
        col_indices = indices.flatten().astype(np.int32)
        mask = row_indices != col_indices
        rows = row_indices[mask]
        cols = col_indices[mask]

        sym_rows = np.concatenate([rows, cols])
        sym_cols = np.concatenate([cols, rows])
        data = np.ones_like(sym_rows, dtype=np.float32)

        adj = csr_matrix(
            (data, (sym_rows, sym_cols)),
            shape=(num_bins, num_bins),
            dtype=np.float32
        )
        adj.data = np.clip(adj.data, 0.0, 1.0)

    laplacian: csr_matrix | None = None
    if compute_laplacian:
        degrees = np.array(adj.sum(axis=1)).flatten()
        degree_matrix = csr_matrix(
            (degrees, (np.arange(num_bins), np.arange(num_bins))),
            shape=(num_bins, num_bins),
            dtype=np.float32
        )
        laplacian = degree_matrix - adj

    return adj, laplacian


def build_spatial_graph(
    bin_coords: np.ndarray,
    max_distance: float,
    n_jobs: int = 1,
    backend: Literal["auto", "cugraph", "cuml", "cuda", "multiprocessing", "threading", "cpu"] = "multiprocessing"
) -> tuple[csr_matrix, csr_matrix]:
    """
    Constructs a spatial radius adjacency graph and its corresponding Graph Laplacian.

    Supports single-threaded CPU, multi-threading, multi-processing, and RAPIDS GPU acceleration.

    Parameters:
        bin_coords: Array of shape (N, 2) holding the (x, y) center coordinates of each bin.
        max_distance: The maximum radius to consider two distinct bins as spatial neighbors.
        n_jobs: Number of parallel jobs to run (-1 uses all available CPU cores).
        backend: Parallel backend ('auto', 'cugraph', 'cuml', 'cuda', 'multiprocessing', 'threading', 'cpu').

    Returns:
        tuple[csr_matrix, csr_matrix]:
            - adjacency_matrix: Sparse binary matrix where matrix[i, j] = 1 if bins are neighbors.
            - laplacian_matrix: Sparse Graph Laplacian matrix (L = D - A).
    """
    bin_coords = np.asarray(bin_coords, dtype=np.float32)
    num_bins = bin_coords.shape[0]

    if num_bins == 0:
        empty_csr = csr_matrix((0, 0), dtype=np.float32)
        return empty_csr, empty_csr

    # GPU Dispatch
    wants_gpu = backend in ("cugraph", "cuml", "cuda") or (backend == "auto" and num_bins >= 5000)
    if wants_gpu and is_gpu_available() and HAS_CUML and cp is not None:
        try:
            d_coords = cp.asarray(bin_coords, dtype=cp.float32)
            # cuML NearestNeighbors radius query
            nn_model = cuNearestNeighbors(radius=max_distance, metric="euclidean", output_type="cupy")
            nn_model.fit(d_coords)
            # Query radius neighbors
            _, indices = nn_model.radius_neighbors(d_coords)
            # Symmetrize and build adjacency
            # Note: radius_neighbors in cuML returns RaggedArray or list of cupy arrays
            all_rows = []
            all_cols = []
            for i in range(num_bins):
                nbrs = cp.asnumpy(indices[i])
                nbrs = nbrs[nbrs != i]
                if len(nbrs) > 0:
                    all_rows.append(np.full(len(nbrs), i, dtype=np.int32))
                    all_cols.append(nbrs.astype(np.int32))
            if all_rows:
                row_indices = np.concatenate(all_rows)
                col_indices = np.concatenate(all_cols)
                rows = np.concatenate([row_indices, col_indices])
                cols = np.concatenate([col_indices, row_indices])
                data = np.ones_like(rows, dtype=np.float32)
                adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(num_bins, num_bins), dtype=np.float32)
                adjacency_matrix.data = np.clip(adjacency_matrix.data, 0.0, 1.0)
            else:
                adjacency_matrix = csr_matrix((num_bins, num_bins), dtype=np.float32)

            degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
            degree_matrix = csr_matrix(
                (degrees, (np.arange(num_bins), np.arange(num_bins))),
                shape=(num_bins, num_bins),
                dtype=np.float32
            )
            laplacian_matrix = degree_matrix - adjacency_matrix
            return adjacency_matrix, laplacian_matrix
        except Exception as e:  # noqa: BLE001 - GPU errors vary by RAPIDS version; always fall back to CPU
            logger.warning(f"GPU radius graph building encountered error: {e}. Falling back to CPU.")

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, int(n_jobs))

    if n_jobs == 1 or num_bins < 50 or backend == "cpu":
        # Fast single-threaded path using KDTree.query_pairs
        tree = KDTree(bin_coords)
        neighbor_pairs = tree.query_pairs(r=max_distance, output_type="ndarray")
        if len(neighbor_pairs) == 0:
            row_indices = np.array([], dtype=np.int32)
            col_indices = np.array([], dtype=np.int32)
        else:
            row_indices = neighbor_pairs[:, 0]
            col_indices = neighbor_pairs[:, 1]
    else:
        # Multi-worker parallel processing path
        chunk_size = int(np.ceil(num_bins / n_jobs))
        tasks = []
        for k in range(0, num_bins, chunk_size):
            batch_coords = bin_coords[k:k + chunk_size]
            tasks.append((batch_coords, bin_coords, max_distance, k))

        all_pairs: list[tuple[int, int]] = []
        ExecutorClass = ProcessPoolExecutor if backend in ("multiprocessing", "auto") else ThreadPoolExecutor
        try:
            with ExecutorClass(max_workers=n_jobs) as executor:
                results = executor.map(_query_batch_neighbors, tasks)
                for res in results:
                    all_pairs.extend(res)
        except (RuntimeError, OSError, TypeError):
            # Fallback to ThreadPoolExecutor if process spawning encounters issues in sub-environments
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = executor.map(_query_batch_neighbors, tasks)
                for res in results:
                    all_pairs.extend(res)

        if len(all_pairs) == 0:
            row_indices = np.array([], dtype=np.int32)
            col_indices = np.array([], dtype=np.int32)
        else:
            pairs_arr = np.array(all_pairs, dtype=np.int32)
            row_indices = pairs_arr[:, 0]
            col_indices = pairs_arr[:, 1]

    # Symmetric / undirected graph
    rows = np.concatenate([row_indices, col_indices])
    cols = np.concatenate([col_indices, row_indices])
    data = np.ones_like(rows, dtype=np.float32)

    # Construct Sparse Adjacency Matrix (A)
    adjacency_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(num_bins, num_bins),
        dtype=np.float32
    )

    # Degree matrix D and Graph Laplacian L = D - A
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degree_matrix = csr_matrix(
        (degrees, (np.arange(num_bins), np.arange(num_bins))),
        shape=(num_bins, num_bins),
        dtype=np.float32
    )
    laplacian_matrix = degree_matrix - adjacency_matrix

    return adjacency_matrix, laplacian_matrix


def to_pyg_data(
    sparse_counts: Any,
    bin_coords: np.ndarray,
    adjacency_matrix: csr_matrix
) -> Any:
    """
    Converts binned label counts, spatial coordinates, and spatial adjacency matrix
    into PyTorch Geometric (PyG) `Data` object format for GNN downstream spatial modeling.

    Parameters:
        sparse_counts: Node feature expression matrix of shape (num_bins, num_labels).
        bin_coords: Spatial node positions of shape (num_bins, 2).
        adjacency_matrix: Adjacency graph matrix of shape (num_bins, num_bins).

    Returns:
        PyG Data object or Dictionary:
            If `torch` and `torch_geometric` are installed, returns `torch_geometric.data.Data(x=..., pos=..., edge_index=...)`.
            Otherwise, returns a standardized PyG-compatible dictionary.
    """
    coo = adjacency_matrix.tocoo()
    edge_index = np.vstack((coo.row, coo.col)).astype(np.int64)

    x_features = sparse_counts.toarray() if hasattr(sparse_counts, "toarray") else np.asarray(sparse_counts)
    pos_coords = np.asarray(bin_coords, dtype=np.float32)

    try:
        import torch
        from torch_geometric.data import Data

        pyg_data = Data(
            x=torch.tensor(x_features, dtype=torch.float32),
            pos=torch.tensor(pos_coords, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long)
        )
        return pyg_data
    except ImportError:
        return {
            "x": x_features,
            "pos": pos_coords,
            "edge_index": edge_index
        }


def compute_laplacian_spectrum(
    laplacian_matrix: csr_matrix,
    k: int = 10
) -> np.ndarray:
    """
    Computes the top K smallest non-zero eigenvalues of the Graph Laplacian matrix
    for graph spectral analysis and spatial connectivity profiling.

    Parameters:
        laplacian_matrix: Graph Laplacian matrix L = D - A of shape (N, N).
        k: Number of smallest eigenvalues to compute.

    Returns:
        np.ndarray: 1D array of sorted eigenvalues.
    """
    num_nodes = laplacian_matrix.shape[0]
    if num_nodes == 0:
        return np.array([], dtype=np.float64)

    target_k = min(k, num_nodes - 1)
    if target_k <= 0:
        return np.array([0.0], dtype=np.float64)

    try:
        from scipy.sparse.linalg import eigsh
        eigenvalues, _ = eigsh(laplacian_matrix.astype(np.float64), k=target_k, which="SM")
        return np.sort(eigenvalues)
    except (ValueError, RuntimeError, ArithmeticError):
        dense_L = laplacian_matrix.toarray()
        evals = np.linalg.eigvalsh(dense_L)
        return np.sort(evals)[:target_k]


def compute_morans_i(
    sparse_counts: Any,
    adjacency_matrix: csr_matrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of Moran's I spatial autocorrelation statistic,
    z-scores, and p-values for all labels across a spatial graph.

    Parameters:
        sparse_counts: Matrix of shape (N_bins, N_labels) containing binned label counts.
        adjacency_matrix: Binary spatial adjacency matrix of shape (N_bins, N_bins).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - morans_i: 1D array of shape (N_labels,) with Moran's I statistic values.
            - z_scores: 1D array of shape (N_labels,) with z-scores.
            - p_values: 1D array of shape (N_labels,) with two-sided p-values.
    """
    from scipy.stats import norm

    if hasattr(sparse_counts, "toarray"):
        X = sparse_counts.toarray().astype(np.float64)
    else:
        X = np.asarray(sparse_counts, dtype=np.float64)

    N, G = X.shape
    if N <= 1 or G == 0:
        return np.zeros(G, dtype=np.float64), np.zeros(G, dtype=np.float64), np.ones(G, dtype=np.float64)

    mean_X = np.mean(X, axis=0, keepdims=True)
    Z = X - mean_X
    denom = np.sum(Z**2, axis=0)

    S0 = float(adjacency_matrix.sum())
    if S0 == 0:
        return np.zeros(G, dtype=np.float64), np.zeros(G, dtype=np.float64), np.ones(G, dtype=np.float64)

    WZ = adjacency_matrix.dot(Z)
    numer = np.sum(Z * WZ, axis=0)

    valid_mask = denom > 1e-12
    morans_i = np.zeros(G, dtype=np.float64)
    morans_i[valid_mask] = (N / S0) * (numer[valid_mask] / denom[valid_mask])

    e_i = -1.0 / (N - 1.0)

    W_coo = adjacency_matrix.tocoo()
    S1 = float(2.0 * W_coo.nnz)
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    S2 = float(4.0 * np.sum(degrees**2))

    if N > 3:
        v_num = (N * ((N**2 - 3 * N + 3) * S1 - N * S2 + 3 * (S0**2))) / ((N - 1.0) * (N - 2.0) * (N - 3.0) * (S0**2))
        var_i = max(v_num - (e_i**2), 1e-12)
    else:
        var_i = 1.0

    std_i = np.sqrt(var_i)
    z_scores = np.zeros(G, dtype=np.float64)
    z_scores[valid_mask] = (morans_i[valid_mask] - e_i) / std_i

    p_values = np.ones(G, dtype=np.float64)
    p_values[valid_mask] = 2.0 * (1.0 - norm.cdf(np.abs(z_scores[valid_mask])))

    return morans_i, z_scores, p_values


def compute_gearys_c(
    sparse_counts: Any,
    adjacency_matrix: csr_matrix
) -> np.ndarray:
    """
    Vectorized computation of Geary's C spatial autocorrelation statistic for all labels.

    Parameters:
        sparse_counts: Matrix of shape (N_bins, N_labels).
        adjacency_matrix: Binary spatial adjacency matrix of shape (N_bins, N_bins).

    Returns:
        np.ndarray: 1D array of shape (N_labels,) with Geary's C statistic values.
    """
    if hasattr(sparse_counts, "toarray"):
        X = sparse_counts.toarray().astype(np.float64)
    else:
        X = np.asarray(sparse_counts, dtype=np.float64)

    N, G = X.shape
    if N <= 1 or G == 0:
        return np.ones(G, dtype=np.float64)

    mean_X = np.mean(X, axis=0, keepdims=True)
    Z = X - mean_X
    denom = np.sum(Z**2, axis=0)

    S0 = float(adjacency_matrix.sum())
    if S0 == 0:
        return np.ones(G, dtype=np.float64)

    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    term1 = 2.0 * np.sum(degrees[:, None] * (X**2), axis=0)
    term2 = 2.0 * np.sum(X * (adjacency_matrix.dot(X)), axis=0)
    numer = term1 - term2

    valid_mask = denom > 1e-12
    gearys_c = np.ones(G, dtype=np.float64)
    gearys_c[valid_mask] = ((N - 1.0) / (2.0 * S0)) * (numer[valid_mask] / denom[valid_mask])
    return gearys_c


def compute_spatially_variable_labels(
    sparse_counts: Any,
    label_names: Any,
    adjacency_matrix: csr_matrix,
    top_n: int | None = None
) -> pd.DataFrame:
    """
    Computes Moran's I and Geary's C statistics for all labels and returns a ranked DataFrame
    with adjusted p-values (FDR Benjamini-Hochberg correction).

    Args:
        sparse_counts: Matrix of shape (N_bins, N_labels).
        label_names: 1D list or array of label identifiers.
        adjacency_matrix: Binary spatial adjacency matrix of shape (N_bins, N_bins).
        top_n: Optional number of top ranked spatially variable labels to return.

    Returns:
        pd.DataFrame: Table sorted by Moran's I descending with columns:
            ['label', 'morans_i', 'gearys_c', 'z_score', 'p_value', 'fdr_p_value'].
    """
    label_names_arr = np.asarray(label_names)
    morans_i, z_scores, p_values = compute_morans_i(sparse_counts, adjacency_matrix)
    gearys_c = compute_gearys_c(sparse_counts, adjacency_matrix)

    G = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    fdr = np.zeros(G, dtype=np.float64)

    cummin = 1.0
    for i in range(G - 1, -1, -1):
        rank = i + 1
        adj_p = (sorted_p[i] * G) / rank
        cummin = min(cummin, adj_p)
        fdr[sorted_indices[i]] = min(1.0, cummin)

    df = pd.DataFrame({
        "label": label_names_arr,
        "morans_i": morans_i,
        "gearys_c": gearys_c,
        "z_score": z_scores,
        "p_value": p_values,
        "fdr_p_value": fdr
    })

    df = df.sort_values(by="morans_i", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)
    return df