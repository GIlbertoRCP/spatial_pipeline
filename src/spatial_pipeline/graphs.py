import os
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def _query_batch_neighbors(args: Tuple[np.ndarray, np.ndarray, float, int]) -> List[Tuple[int, int]]:
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
) -> Tuple[csr_matrix, csr_matrix]:
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
        except Exception:
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