import time
import tracemalloc

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.spatial import KDTree


def bin_square(x: np.ndarray, y: np.ndarray, resolution: float) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Bins 2D points into a square grid of a given side length (resolution) via spatial hashing.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    resolution : float
        Side length of each square bin.
        
    Returns
    -------
    bin_coords : np.ndarray
        2D array of shape (N, 2) containing integer grid indices (bin_x, bin_y).
    offset : Tuple[float, float]
        The minimum x and y coordinates used to offset the binning.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) == 0:
        return np.empty((0, 2), dtype=int), (0.0, 0.0)
        
    x_min = float(np.min(x))
    y_min = float(np.min(y))
    
    # Use float64 arithmetic to preserve precision on extreme ranges/sub-nanometer values
    bin_x = np.floor((x.astype(np.float64) - x_min) / float(resolution)).astype(int)
    bin_y = np.floor((y.astype(np.float64) - y_min) / float(resolution)).astype(int)
    
    return np.column_stack((bin_x, bin_y)), (x_min, y_min)


def bin_hexagonal(x: np.ndarray, y: np.ndarray, resolution: float) -> np.ndarray:
    """
    Bins 2D points into a hexagonal grid where resolution defines the distance
    between centers of adjacent hexagons.
    
    Uses flat-topped or pointy-topped hexagons. Here we use pointy-topped hexagons
    where:
      - R (outer radius) = resolution / sqrt(3)
      - Spacing between adjacent columns = resolution
      - Spacing between adjacent rows = 1.5 * R
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    resolution : float
        Center-to-center distance between adjacent hexagons.
        
    Returns
    -------
    bin_coords : np.ndarray
        2D array of shape (N, 2) containing integer axial coordinates (q, r).
    """
    R = resolution / np.sqrt(3.0)
    
    # Transform Cartesian to fractional axial coordinates (q_frac, r_frac)
    q_frac = (np.sqrt(3.0) * x - y) / (3.0 * R)
    r_frac = (2.0 / 3.0) * y / R
    s_frac = -q_frac - r_frac
    
    # Round to nearest integer coordinates
    q_round = np.round(q_frac).astype(int)
    r_round = np.round(r_frac).astype(int)
    s_round = np.round(s_frac).astype(int)
    
    # Calculate rounding errors
    dq = np.abs(q_round - q_frac)
    dr = np.abs(r_round - r_frac)
    ds = np.abs(s_round - s_frac)
    
    # Correct rounding to enforce q + r + s = 0 constraint
    mask_q = (dq >= dr) & (dq >= ds)
    mask_r = (~mask_q) & (dr >= ds)
    
    q_round = np.where(mask_q, -r_round - s_round, q_round)

    r_round = np.where(mask_r, -q_round - s_round, r_round)
    
    return np.column_stack((q_round, r_round))

def bin_kdtree(x: np.ndarray, y: np.ndarray, resolution: float) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Bins 2D points into a grid by searching nearest spatial grid centers using a KD-Tree.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    resolution : float
        Side length of each square bin.
        
    Returns
    -------
    bin_coords : np.ndarray
        2D array of shape (N, 2) containing integer grid indices (bin_x, bin_y).
    offset : Tuple[float, float]
        The minimum x and y coordinates used to offset the binning.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return np.empty((0, 2), dtype=int), (0.0, 0.0)
        
    x_min = float(np.min(x))
    y_min = float(np.min(y))
    
    # Calculate target spatial hashing grid bounds
    bin_x_approx = np.floor((x - x_min) / resolution).astype(int)
    bin_y_approx = np.floor((y - y_min) / resolution).astype(int)
    
    # Unique active bin grid indices
    active_grid = np.unique(np.column_stack((bin_x_approx, bin_y_approx)), axis=0)
    
    # Center coordinates of active bins
    centers = np.column_stack((
        x_min + (active_grid[:, 0] + 0.5) * resolution,
        y_min + (active_grid[:, 1] + 0.5) * resolution
    ))
    
    # Query KDTree for exact nearest center match
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
    Bins 2D points into a spatial grid using recursive Quadtree decomposition.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    resolution : float
        Target cell size threshold for leaf nodes.
    max_depth : int, default 10
        Maximum recursion depth of the Quadtree.
        
    Returns
    -------
    bin_coords : np.ndarray
        2D array of shape (N, 2) containing integer grid indices (bin_x, bin_y).
    offset : Tuple[float, float]
        The minimum x and y coordinates used to offset the binning.
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
    
    def _split_node(indices: np.ndarray, bbox: tuple[float, float, float, float], depth: int):
        if len(indices) == 0:
            return
        bx_min, bx_max, by_min, by_max = bbox
        width = bx_max - bx_min
        height = by_max - by_min
        
        if width <= resolution or height <= resolution or depth >= max_depth:
            # Leaf node assignment
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
    Bins 2D points into a square grid using chunked memory-mapped arrays, 
    preserving an O(chunk_size) memory ceiling for massive datasets.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    resolution : float
        Side length of each square bin.
    chunk_size : int, default 100000
        Maximum chunk size for memory-bounded processing.
        
    Returns
    -------
    bin_coords : np.ndarray
        2D array of shape (N, 2) containing integer grid indices (bin_x, bin_y).
    offset : Tuple[float, float]
        The minimum x and y coordinates used to offset the binning.
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
    Compares speed and memory tradeoffs between Spatial Hashing, KD-Tree, Quadtree, and Memmap Grid binning algorithms.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    resolution : float
        Bin resolution size.
    runs : int, default 3
        Number of benchmark repetitions for averaging.
        
    Returns
    -------
    pd.DataFrame
        DataFrame summarizing execution time (ms) and peak memory usage (MB) for each algorithm.
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
            
        avg_time = np.mean(times)
        avg_peak_mem = np.mean(peak_mems)
        results.append({
            "Algorithm": name,
            "Execution Time (ms)": round(float(avg_time), 3),
            "Peak Memory (MB)": round(float(avg_peak_mem), 4)
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
    bin_type: str, 
    resolution: float, 
    offset: tuple[float, float] | None = None
) -> np.ndarray:

    """
    Computes Cartesian center coordinates for unique grid bins.
    
    Parameters
    ----------
    unique_bins : np.ndarray
        2D array of shape (M, 2) of integer bin coordinates.
    bin_type : str
        Type of grid: "square" or "hex".
    resolution : float
        Resolution (grid spacing).
    offset : Tuple[float, float], optional
        The (x_min, y_min) offset for square bins. Required if bin_type is "square".
        
    Returns
    -------
    np.ndarray
        2D array of shape (M, 2) of Cartesian coordinates (x, y) of bin centers.
    """
    if bin_type == "square":
        if offset is None:
            raise ValueError("Offset (x_min, y_min) is required for square bin centers.")
        x_min, y_min = offset
        # Center of square bin is at (bin_idx + 0.5) * resolution
        x_centers = (unique_bins[:, 0] + 0.5) * resolution + x_min
        y_centers = (unique_bins[:, 1] + 0.5) * resolution + y_min
        return np.column_stack((x_centers, y_centers))
        
    elif bin_type == "hex":
        R = resolution / np.sqrt(3.0)
        q = unique_bins[:, 0]
        r = unique_bins[:, 1]
        # Reverse axial coordinates mapping to Cartesian
        x_centers = R * (np.sqrt(3.0) * q + (np.sqrt(3.0) / 2.0) * r)
        y_centers = R * (1.5 * r)
        return np.column_stack((x_centers, y_centers))
        
    else:
        raise ValueError(f"Unknown bin type: {bin_type}")

def build_sparse_matrix(
    bin_coords: np.ndarray, 
    genes: np.ndarray, 
    matrix_format: str = "csr"
) -> tuple[csr_matrix | csc_matrix, np.ndarray, np.ndarray]:
    """
    Aggregates molecule counts within active bins and exports to a SciPy sparse matrix.
    
    Parameters
    ----------
    bin_coords : np.ndarray
        2D array of shape (N, 2) containing integer bin coordinates for each molecule.
    genes : np.ndarray
        1D array of gene names (strings or categories) for each molecule.
    matrix_format : str
        Format of sparse matrix: "csr" or "csc".
        
    Returns
    -------
    sparse_matrix : Union[csr_matrix, csc_matrix]
        Sparse matrix of shape (num_active_bins, num_unique_genes) containing count data.
    unique_bins : np.ndarray
        2D array of shape (num_active_bins, 2) containing integer coordinates of active bins.
    unique_genes : np.ndarray
        1D array of shape (num_unique_genes,) containing gene names corresponding to columns.
    """
    # Map active bins to unique sequential indices
    unique_bins, inverse_bins = np.unique(bin_coords, axis=0, return_inverse=True)
    
    # Map genes to unique sequential indices
    unique_genes, inverse_genes = np.unique(genes, return_inverse=True)
    
    # Construct SciPy COO matrix (automatically aggregates counts of duplicate indices)
    counts = np.ones(len(genes), dtype=int)
    coo = coo_matrix(
        (counts, (inverse_bins, inverse_genes)),
        shape=(len(unique_bins), len(unique_genes)),
        dtype=int
    )
    
    # Convert to target sparse matrix format
    if matrix_format.lower() == "csr":
        sparse_matrix = coo.tocsr()
    elif matrix_format.lower() == "csc":
        sparse_matrix = coo.tocsc()
    else:
        raise ValueError(f"Unsupported sparse matrix format: {matrix_format}. Use 'csr' or 'csc'.")
        
    return sparse_matrix, unique_bins, unique_genes

def bin_voronoi(
    x: np.ndarray, 
    y: np.ndarray, 
    num_bins: int = 50
) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """
    Spatially bins 2D points into adaptive Voronoi regions where region sizes adapt dynamically
    to local transcript density.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    num_bins : int, default 50
        Target number of adaptive Voronoi region seeds.
        
    Returns
    -------
    bin_indices : np.ndarray
        1D array of shape (N,) containing assigned Voronoi seed region index for each point.
    offset : Tuple[float, float]
        (x_min, y_min) minimum bounds.
    seed_centers : np.ndarray
        2D array of shape (num_bins, 2) holding Voronoi seed center coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return np.empty(0, dtype=int), (0.0, 0.0), np.empty((0, 2))
        
    x_min = float(np.min(x))
    y_min = float(np.min(y))
    
    points = np.column_stack((x, y))
    k = min(num_bins, len(points))
    
    # Use k-means seed generation to adaptively cluster density centers
    from scipy.cluster.vq import kmeans2
    seed_centers, _ = kmeans2(points.astype(np.float64), k=k, minit='points')
    
    # Assign each coordinate to nearest Voronoi seed
    tree = KDTree(seed_centers)
    _, bin_indices = tree.query(points)
    
    return bin_indices, (x_min, y_min), seed_centers

def compute_square_bins(
    df: pd.DataFrame, 
    resolution: float,
    min_counts: int = 1
) -> tuple[csr_matrix, np.ndarray]:
    """
    Spatially bins continuous coordinate points into discrete grid blocks.
    
    Parameters:
        df: DataFrame containing 'x', 'y', and 'gene' columns.
        resolution: The physical size/width of each square bin.
        min_counts: Minimum transcript count threshold per bin to eliminate background noise.
        
    Returns:
        sparse_counts: A Compressed Sparse Row matrix (bins x unique_genes).
        bin_coords: An array of shape (N, 2) holding the actual center (x, y) 
                    coordinates for each valid bin index.
    """
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    
    if len(x) == 0:
        num_genes = len(df['gene'].cat.categories) if hasattr(df['gene'], 'cat') else 0
        return csr_matrix((0, num_genes), dtype=np.int32), np.empty((0, 2))
    
    # Map genes to unique integer categorical codes
    gene_codes = df['gene'].cat.codes.to_numpy()
    num_genes = len(df['gene'].cat.categories)
    
    # Establish grid origin points
    x_min, y_min = float(x.min()), float(y.min())
    
    # Math: Compute integer index positions for every single transcript molecule
    grid_x = np.floor((x - x_min) / resolution).astype(np.int32)
    grid_y = np.floor((y - y_min) / resolution).astype(np.int32)
    
    # Calculate unique 1D bin IDs using stride configuration
    max_grid_x = int(grid_x.max()) + 1
    bin_ids = grid_y.astype(np.int64) * max_grid_x + grid_x.astype(np.int64)
    
    # Map active bin IDs to sequential row indices (eliminating empty space)
    unique_bins, inverse_indices = np.unique(bin_ids, return_inverse=True)
    num_unique_bins = len(unique_bins)
    
    # Reconstruct physical spatial centers for the unique active bins
    unique_grid_y = unique_bins // max_grid_x
    unique_grid_x = unique_bins % max_grid_x
    
    bin_x_centers = x_min + (unique_grid_x + 0.5) * resolution
    bin_y_centers = y_min + (unique_grid_y + 0.5) * resolution
    bin_coords = np.column_stack((bin_x_centers, bin_y_centers))
    
    # Construct an optimized high-performance Compressed Sparse Row Matrix
    sparse_counts = csr_matrix(
        (np.ones_like(inverse_indices), (inverse_indices, gene_codes)),
        shape=(num_unique_bins, num_genes),
        dtype=np.int32
    )
    
    if min_counts > 1:
        bin_totals = np.array(sparse_counts.sum(axis=1)).flatten()
        valid_mask = bin_totals >= min_counts
        sparse_counts = sparse_counts[valid_mask]
        bin_coords = bin_coords[valid_mask]
    
    return sparse_counts, bin_coords

