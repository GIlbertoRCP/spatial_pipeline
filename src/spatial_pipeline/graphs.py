import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix
from typing import Union, Set, Tuple

def build_adjacency_graph(
    bin_centers: np.ndarray, 
    bin_type: str, 
    resolution: float, 
    connectivity: int = 8,
    include_self: bool = False
) -> csr_matrix:
    """
    Generates a spatial adjacency graph representing neighbor bins using scipy.spatial.KDTree.
    
    Parameters
    ----------
    bin_centers : np.ndarray
        2D array of shape (M, 2) containing Cartesian center coordinates for each active bin.
    bin_type : str
        Type of grid: "square" or "hex".
    resolution : float
        Grid resolution (spacing between centers or side length).
    connectivity : int
        Connectivity for square grid neighbors: 4 (edge sharing) or 8 (edge and corner sharing).
        Ignored for hexagonal grids (always 6-connectivity).
    include_self : bool
        If True, self-loops are added to the adjacency matrix.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        A symmetric sparse adjacency matrix of shape (M, M) with 1s at neighbor positions.
    """
    if len(bin_centers) == 0:
        return csr_matrix((0, 0), dtype=int)
        
    # Build KDTree on the bin center coordinates
    tree = KDTree(bin_centers)
    
    # Define distance threshold based on grid type and connectivity
    if bin_type == "square":
        if connectivity == 4:
            # 4-connected: only immediate horizontal and vertical neighbors (distance = resolution)
            threshold = 1.01 * resolution
        elif connectivity == 8:
            # 8-connected: includes diagonal neighbors (distance = resolution * sqrt(2))
            threshold = 1.01 * resolution * np.sqrt(2.0)
        else:
            raise ValueError(f"Unsupported connectivity for square bins: {connectivity}. Use 4 or 8.")
            
    elif bin_type == "hex":
        # Hexagonal grid: distance between center and all 6 neighbors is exactly resolution
        threshold = 1.01 * resolution
        
    else:
        raise ValueError(f"Unknown bin type: {bin_type}")
        
    # Query all pairs within threshold distance (excluding self-pairs)
    pairs_set = tree.query_pairs(r=threshold)
    
    if len(pairs_set) == 0:
        row = np.array([], dtype=int)
        col = np.array([], dtype=int)
        data = np.array([], dtype=int)
    else:
        pairs = np.array(list(pairs_set))
        # Since the graph is undirected, make the adjacency matrix symmetric
        row = np.concatenate([pairs[:, 0], pairs[:, 1]])
        col = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(row), dtype=int)
        
    if include_self:
        n_nodes = len(bin_centers)
        self_rows = np.arange(n_nodes)
        row = np.concatenate([row, self_rows])
        col = np.concatenate([col, self_rows])
        data = np.concatenate([data, np.ones(n_nodes, dtype=int)])
        
    # Build sparse COO matrix and convert to CSR for efficiency
    n_nodes = len(bin_centers)
    adj_matrix = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=int)
    
    return adj_matrix.tocsr()
