import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from typing import Tuple

def build_spatial_graph(
    bin_coords: np.ndarray, 
    max_distance: float
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Constructs a spatial adjacency graph and its corresponding Graph Laplacian.
    
    Parameters:
        bin_coords: Array of shape (N, 2) holding the (x, y) center coordinates of each bin.
        max_distance: The maximum radius to consider two distinct bins as spatial neighbors.
        
    Returns:
        adjacency_matrix: Sparse binary matrix where matrix[i, j] = 1 if bins are neighbors.
        laplacian_matrix: Sparse Graph Laplacian matrix (L = D - A).
    """
    num_bins = bin_coords.shape[0]
    
    # 1. Build a fast KDTree for spatial query lookup
    tree = KDTree(bin_coords)
    
    # 2. Query pairs of bins that sit within the neighbor distance threshold
    # avoid_self=True drops the self-distance diagonal entries
    neighbor_pairs = tree.query_pairs(r=max_distance, output_type='ndarray')
    
    if len(neighbor_pairs) == 0:
        # Fallback handle case for zero spatial overlaps
        row_indices = np.array([], dtype=np.int32)
        col_indices = np.array([], dtype=np.int32)
    else:
        row_indices = neighbor_pairs[:, 0]
        col_indices = neighbor_pairs[:, 1]
        
    # Since an adjacency graph is symmetric/undirected, mirror the connection pairs
    rows = np.concatenate([row_indices, col_indices])
    cols = np.concatenate([col_indices, row_indices])
    data = np.ones_like(rows, dtype=np.float32)
    
    # Construct Sparse Adjacency Matrix (A)
    adjacency_matrix = csr_matrix(
        (data, (rows, cols)), 
        shape=(num_bins, num_bins), 
        dtype=np.float32
    )
    
    # 3. Calculate Degree values (sum of rows) to form Diagonal Matrix D
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    
    # Construct the sparse diagonal Degree Matrix
    # We use a custom manual CSR structure to cleanly subtract them next
    degree_matrix = csr_matrix(
        (degrees, (np.arange(num_bins), np.arange(num_bins))),
        shape=(num_bins, num_bins),
        dtype=np.float32
    )
    
    # Math: Graph Laplacian is defined as L = D - A
    laplacian_matrix = degree_matrix - adjacency_matrix
    
    return adjacency_matrix, laplacian_matrix