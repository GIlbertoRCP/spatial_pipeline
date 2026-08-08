import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree


def _query_batch_neighbors(args: tuple[np.ndarray, np.ndarray, float, int]) -> list[tuple[int, int]]:
    """Helper worker for multi-processing batch spatial query."""
    batch_coords, all_coords, max_dist, start_idx = args
    tree = KDTree(all_coords)
    neighbors_list = tree.query_ball_point(batch_coords, r=max_dist)
    pairs = []
    for idx_in_batch, neighbors in enumerate(neighbors_list):
        i = start_idx + idx_in_batch
        for j in neighbors:
            if i < j:  # upper triangle pair
                pairs.append((i, j))
    return pairs

def build_spatial_graph(
    bin_coords: np.ndarray, 
    max_distance: float,
    n_jobs: int = 1,
    backend: str = "multiprocessing"
) -> tuple[csr_matrix, csr_matrix]:
    """
    Constructs a spatial adjacency graph and its corresponding Graph Laplacian.
    Supports multi-threading and multi-processing for parallel construction.
    
    Parameters:
        bin_coords: Array of shape (N, 2) holding the (x, y) center coordinates of each bin.
        max_distance: The maximum radius to consider two distinct bins as spatial neighbors.
        n_jobs: Number of parallel jobs to run (-1 uses all available CPU cores).
        backend: Parallel backend - "multiprocessing" or "threading".
        
    Returns:
        adjacency_matrix: Sparse binary matrix where matrix[i, j] = 1 if bins are neighbors.
        laplacian_matrix: Sparse Graph Laplacian matrix (L = D - A).
    """
    bin_coords = np.asarray(bin_coords)
    num_bins = bin_coords.shape[0]
    
    if num_bins == 0:
        empty_csr = csr_matrix((0, 0), dtype=np.float32)
        return empty_csr, empty_csr

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
        
    n_jobs = max(1, int(n_jobs))

    if n_jobs == 1 or num_bins < 50:
        # Fast single-threaded path using KDTree.query_pairs
        tree = KDTree(bin_coords)
        neighbor_pairs = tree.query_pairs(r=max_distance, output_type='ndarray')
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

        all_pairs = []
        ExecutorClass = ProcessPoolExecutor if backend.lower() == "multiprocessing" else ThreadPoolExecutor
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
    Converts binned gene expression counts, spatial coordinates, and spatial adjacency matrix
    into PyTorch Geometric (PyG) `Data` object format for GNN downstream spatial modeling.
    
    Parameters
    ----------
    sparse_counts : scipy.sparse matrix
        Node feature expression matrix of shape (num_bins, num_genes).
    bin_coords : np.ndarray
        Spatial node positions of shape (num_bins, 2).
    adjacency_matrix : csr_matrix
        Adjacency graph matrix of shape (num_bins, num_bins).
        
    Returns
    -------
    PyG Data object or Dictionary
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
    
    Parameters
    ----------
    laplacian_matrix : csr_matrix
        Graph Laplacian matrix L = D - A of shape (N, N).
    k : int, default 10
        Number of smallest eigenvalues to compute.
        
    Returns
    -------
    np.ndarray
        1D array of sorted eigenvalues.
    """
    num_nodes = laplacian_matrix.shape[0]
    if num_nodes == 0:
        return np.array([])
        
    target_k = min(k, num_nodes - 1)
    if target_k <= 0:
        return np.array([0.0])
        
    try:
        from scipy.sparse.linalg import eigsh
        eigenvalues, _ = eigsh(laplacian_matrix.astype(np.float64), k=target_k, which='SM')
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
    z-scores, and p-values for all genes across a spatial graph.
    
    Parameters
    ----------
    sparse_counts : scipy.sparse matrix or np.ndarray
        Matrix of shape (N_bins, N_genes) containing binned gene expression counts.
    adjacency_matrix : csr_matrix
        Binary spatial adjacency matrix of shape (N_bins, N_bins).
        
    Returns
    -------
    morans_i : np.ndarray
        1D array of shape (N_genes,) with Moran's I statistic values.
    z_scores : np.ndarray
        1D array of shape (N_genes,) with z-scores.
    p_values : np.ndarray
        1D array of shape (N_genes,) with two-sided p-values.
    """
    from scipy.stats import norm

    if hasattr(sparse_counts, "toarray"):
        X = sparse_counts.toarray().astype(np.float64)
    else:
        X = np.asarray(sparse_counts, dtype=np.float64)
        
    N, G = X.shape
    if N <= 1 or G == 0:
        return np.zeros(G), np.zeros(G), np.ones(G)
        
    mean_X = np.mean(X, axis=0, keepdims=True)
    Z = X - mean_X
    denom = np.sum(Z**2, axis=0)
    
    S0 = float(adjacency_matrix.sum())
    if S0 == 0:
        return np.zeros(G), np.zeros(G), np.ones(G)
        
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
        v_num = (N * ((N**2 - 3*N + 3)*S1 - N*S2 + 3*(S0**2))) / ((N - 1.0)*(N - 2.0)*(N - 3.0)*(S0**2))
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
    Vectorized computation of Geary's C spatial autocorrelation statistic for all genes.
    
    Parameters
    ----------
    sparse_counts : scipy.sparse matrix or np.ndarray
        Matrix of shape (N_bins, N_genes).
    adjacency_matrix : csr_matrix
        Binary spatial adjacency matrix of shape (N_bins, N_bins).
        
    Returns
    -------
    gearys_c : np.ndarray
        1D array of shape (N_genes,) with Geary's C statistic values.
    """
    if hasattr(sparse_counts, "toarray"):
        X = sparse_counts.toarray().astype(np.float64)
    else:
        X = np.asarray(sparse_counts, dtype=np.float64)
        
    N, G = X.shape
    if N <= 1 or G == 0:
        return np.ones(G)
        
    mean_X = np.mean(X, axis=0, keepdims=True)
    Z = X - mean_X
    denom = np.sum(Z**2, axis=0)
    
    S0 = float(adjacency_matrix.sum())
    if S0 == 0:
        return np.ones(G)
        
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    term1 = 2.0 * np.sum(degrees[:, None] * (X**2), axis=0)
    term2 = 2.0 * np.sum(X * (adjacency_matrix.dot(X)), axis=0)
    numer = term1 - term2
    
    valid_mask = denom > 1e-12
    gearys_c = np.ones(G, dtype=np.float64)
    gearys_c[valid_mask] = ((N - 1.0) / (2.0 * S0)) * (numer[valid_mask] / denom[valid_mask])
    return gearys_c


def compute_spatially_variable_genes(
    sparse_counts: Any,
    gene_names: Any,
    adjacency_matrix: csr_matrix,
    top_n: int | None = None
) -> Any:
    """
    Computes Moran's I and Geary's C statistics for all genes and returns a ranked pandas DataFrame
    with adjusted p-values (FDR Benjamini-Hochberg correction).
    """
    import pandas as pd
    gene_names_arr = np.asarray(gene_names)
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
        "gene": gene_names_arr,
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


