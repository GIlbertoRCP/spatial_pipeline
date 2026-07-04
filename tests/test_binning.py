import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from src.spatial_pipeline.binning import (
    bin_square, 
    bin_hexagonal, 
    get_bin_centers, 
    build_sparse_matrix,
    compute_square_bins
)
from src.spatial_pipeline.graphs import build_adjacency_graph

def test_bin_square():
    x = np.array([0.0, 5.0, 10.0, 14.9, 15.0])
    y = np.array([0.0, 5.0, 10.0, 14.9, 15.0])
    resolution = 10.0
    
    bin_coords, offset = bin_square(x, y, resolution)
    
    # Minimum offset should be 0.0
    assert offset == (0.0, 0.0)
    
    # Expected bins:
    # 0.0 -> bin 0
    # 5.0 -> bin 0
    # 10.0 -> bin 1
    # 14.9 -> bin 1
    # 15.0 -> bin 1
    expected_x = np.array([0, 0, 1, 1, 1])
    expected_y = np.array([0, 0, 1, 1, 1])
    
    np.testing.assert_array_equal(bin_coords[:, 0], expected_x)
    np.testing.assert_array_equal(bin_coords[:, 1], expected_y)

def test_bin_hexagonal():
    resolution = 10.0
    R = resolution / np.sqrt(3.0)
    
    # Test origin
    x = np.array([0.0, resolution, 0.5 * resolution])
    y = np.array([0.0, 0.0, 1.5 * R])
    
    bin_coords = bin_hexagonal(x, y, resolution)
    
    # Origin (0,0) -> (0,0)
    assert np.all(bin_coords[0] == [0, 0])
    # Adjacent right center (resolution, 0) -> (1,0)
    assert np.all(bin_coords[1] == [1, 0])
    # Adjacent top right (0.5 * resolution, 1.5 * R) -> (0,1)
    assert np.all(bin_coords[2] == [0, 1])

def test_get_bin_centers():
    unique_bins = np.array([[0, 0], [1, 2]])
    resolution = 10.0
    
    # Square
    offset = (100.0, 200.0)
    centers_square = get_bin_centers(unique_bins, "square", resolution, offset)
    
    # bin [0,0] -> center = ((0 + 0.5)*10 + 100, (0 + 0.5)*10 + 200) = (105, 205)
    # bin [1,2] -> center = ((1 + 0.5)*10 + 100, (2 + 0.5)*10 + 200) = (115, 225)
    expected_square = np.array([[105.0, 205.0], [115.0, 225.0]])
    np.testing.assert_allclose(centers_square, expected_square)
    
    # Hex
    centers_hex = get_bin_centers(unique_bins, "hex", resolution)
    R = resolution / np.sqrt(3.0)
    # bin [0,0] -> center = (0, 0)
    # bin [1,2] -> q=1, r=2
    # x = R * (sqrt(3)*1 + sqrt(3)/2 * 2) = R * 2 * sqrt(3) = R * sqrt(12) = 2 * resolution
    # y = R * 1.5 * 2 = 3 * R = sqrt(3) * resolution
    expected_hex = np.array([
        [0.0, 0.0],
        [R * (np.sqrt(3.0) + np.sqrt(3.0)), R * 3.0]
    ])
    np.testing.assert_allclose(centers_hex, expected_hex)

def test_build_sparse_matrix():
    bin_coords = np.array([
        [0, 0],  # bin 0
        [0, 0],  # bin 0
        [0, 1],  # bin 1
        [0, 0],  # bin 0
        [1, 1],  # bin 2
    ])
    genes = np.array(["ACTB", "ACTB", "CD3D", "GAPDH", "CD3D"])
    
    matrix, unique_bins, unique_genes = build_sparse_matrix(bin_coords, genes, "csr")
    
    assert len(unique_bins) == 3
    assert set(unique_genes) == {"ACTB", "CD3D", "GAPDH"}
    
    # Find column indices
    actb_col = np.where(unique_genes == "ACTB")[0][0]
    cd3d_col = np.where(unique_genes == "CD3D")[0][0]
    gapdh_col = np.where(unique_genes == "GAPDH")[0][0]
    
    # Find row indices of the unique bins
    bin_0_0 = np.where((unique_bins == [0, 0]).all(axis=1))[0][0]
    bin_0_1 = np.where((unique_bins == [0, 1]).all(axis=1))[0][0]
    bin_1_1 = np.where((unique_bins == [1, 1]).all(axis=1))[0][0]
    
    # Assert counts
    # bin [0,0] has 2 ACTB, 1 GAPDH
    assert matrix[bin_0_0, actb_col] == 2
    assert matrix[bin_0_0, gapdh_col] == 1
    assert matrix[bin_0_0, cd3d_col] == 0
    
    # bin [0,1] has 1 CD3D
    assert matrix[bin_0_1, cd3d_col] == 1
    
    # bin [1,1] has 1 CD3D
    assert matrix[bin_1_1, cd3d_col] == 1

def test_build_adjacency_graph_square():
    resolution = 10.0
    # 3x3 square grid: coordinates (0,0) to (2,2)
    x, y = np.meshgrid([5, 15, 25], [5, 15, 25])
    bin_centers = np.column_stack((x.flatten(), y.flatten()))
    
    # Center node is at index 4 (15, 15)
    # With 4-connectivity: neighbors are indices 1 (15, 5), 3 (5, 15), 5 (25, 15), 7 (15, 25)
    adj_4 = build_adjacency_graph(bin_centers, "square", resolution, connectivity=4)
    assert adj_4[4, 4] == 0  # No self loop
    assert adj_4[4, 1] == 1
    assert adj_4[4, 3] == 1
    assert adj_4[4, 5] == 1
    assert adj_4[4, 7] == 1
    assert adj_4[4, 0] == 0  # Diagonal is not neighbor
    assert adj_4.sum(axis=1)[4] == 4
    
    # With 8-connectivity: all 8 neighbors are connected
    adj_8 = build_adjacency_graph(bin_centers, "square", resolution, connectivity=8)
    assert adj_8.sum(axis=1)[4] == 8
    assert adj_8[4, 0] == 1  # Diagonal neighbor connected

def test_build_adjacency_graph_hex():
    resolution = 10.0
    
    # Let's create a central hex and its 6 neighbors
    unique_bins = np.array([
        [0, 0],   # Center
        [1, 0],   # Right
        [0, 1],   # Top-Right
        [-1, 1],  # Top-Left
        [-1, 0],  # Left
        [0, -1],  # Bottom-Left
        [1, -1]   # Bottom-Right
    ])
    
    bin_centers = get_bin_centers(unique_bins, "hex", resolution)
    adj = build_adjacency_graph(bin_centers, "hex", resolution)
    
    # The center node (index 0) should be connected to all 6 surrounding nodes (indices 1 to 6)
    assert adj.sum(axis=1)[0] == 6
    for i in range(1, 7):
        assert adj[0, i] == 1

def test_square_binning_aggregation():
    # Create mock data: 3 molecules sitting in the exact same spatial bin area
    mock_data = pd.DataFrame({
        'x': [10.2, 10.5, 10.8],
        'y': [20.1, 20.4, 20.3],
        'gene': pd.Series(['GeneA', 'GeneB', 'GeneA'], dtype='category')
    })
    
    # Run binning with a resolution wide enough to hold all 3 molecules
    sparse_counts, bin_coords = compute_square_bins(mock_data, resolution=5.0)
    
    # Assertions to verify our array tracking dimensions match expectations
    assert sparse_counts.shape[0] == 1  # All coordinates must collapse to exactly 1 unique bin
    assert sparse_counts[0, 0] == 2    # Count for GeneA should equal 2
    assert sparse_counts[0, 1] == 1    # Count for GeneB should equal 1
