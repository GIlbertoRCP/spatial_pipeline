import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from typing import Tuple, Dict, Any, Union

def bin_square(x: np.ndarray, y: np.ndarray, resolution: float) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Bins 2D points into a square grid of a given side length (resolution).
    
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
    x_min = float(np.min(x))
    y_min = float(np.min(y))
    
    bin_x = np.floor((x - x_min) / resolution).astype(int)
    bin_y = np.floor((y - y_min) / resolution).astype(int)
    
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
    mask_s = (~mask_q) & (~mask_r)
    
    q_round = np.where(mask_q, -r_round - s_round, q_round)
    r_round = np.where(mask_r, -q_round - s_round, r_round)
    
    return np.column_stack((q_round, r_round))

def get_bin_centers(
    unique_bins: np.ndarray, 
    bin_type: str, 
    resolution: float, 
    offset: Tuple[float, float] = None
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
) -> Tuple[Union[csr_matrix, csc_matrix], np.ndarray, np.ndarray]:
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
