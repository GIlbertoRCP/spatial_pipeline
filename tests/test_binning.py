import os
import shutil
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.spatial_pipeline.binning import (
    bin_square, 
    bin_hexagonal, 
    bin_kdtree,
    bin_quadtree,
    bin_voronoi,
    benchmark_binning_algorithms,
    get_bin_centers, 
    build_sparse_matrix,
    compute_square_bins
)
from src.spatial_pipeline.graphs import (
    build_spatial_graph,
    to_pyg_data,
    compute_laplacian_spectrum
)
from src.spatial_pipeline.out_of_core import (
    partition_dataset_to_tiles,
    process_tile_out_of_core,
    stitch_out_of_core_results,
    profile_out_of_core_execution
)
from tests.synthetic import write_synthetic_points


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
    labels = np.array(["columbine", "columbine", "CD3D", "elk", "CD3D"])
    
    matrix, unique_bins, unique_labels = build_sparse_matrix(bin_coords, labels, "csr")
    
    assert len(unique_bins) == 3
    assert set(unique_labels) == {"columbine", "CD3D", "elk"}
    
    # Find column indices
    actb_col = np.where(unique_labels == "columbine")[0][0]
    cd3d_col = np.where(unique_labels == "CD3D")[0][0]
    gapdh_col = np.where(unique_labels == "elk")[0][0]
    
    # Find row indices of the unique bins
    bin_0_0 = np.where((unique_bins == [0, 0]).all(axis=1))[0][0]
    bin_0_1 = np.where((unique_bins == [0, 1]).all(axis=1))[0][0]
    bin_1_1 = np.where((unique_bins == [1, 1]).all(axis=1))[0][0]
    
    # Assert counts
    # bin [0,0] has 2 columbine, 1 elk
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
    
    # 4-connectivity: neighbors are indices 1 (15, 5), 3 (5, 15), 5 (25, 15), 7 (15, 25)
    adj_4, lap_4 = build_spatial_graph(bin_centers, max_distance=10.1)
    assert adj_4[4, 4] == 0  # No self loop
    assert adj_4[4, 1] == 1
    assert adj_4[4, 3] == 1
    assert adj_4[4, 5] == 1
    assert adj_4[4, 7] == 1
    assert adj_4[4, 0] == 0  # Diagonal is not neighbor
    assert adj_4.sum(axis=1)[4] == 4
    
    # 8-connectivity: all 8 neighbors are connected
    adj_8, lap_8 = build_spatial_graph(bin_centers, max_distance=14.2)
    assert adj_8.sum(axis=1)[4] == 8
    assert adj_8[4, 0] == 1  # Diagonal neighbor connected
    
    # Laplacian matrix checks (L = D - A)
    assert lap_8[4, 4] == 8
    assert lap_8[4, 0] == -1

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
    adj, lap = build_spatial_graph(bin_centers, max_distance=resolution * 1.01)
    
    # The center node (index 0) should be connected to all 6 surrounding nodes (indices 1 to 6)
    assert adj.sum(axis=1)[0] == 6
    for i in range(1, 7):
        assert adj[0, i] == 1
        
    # Check Laplacian: L[0,0] = degree = 6, and L[0, i] = -1
    assert lap[0, 0] == 6
    for i in range(1, 7):
        assert lap[0, i] == -1

def test_square_binning_aggregation():
    # Create mock data: 3 points sitting in the exact same spatial bin area
    mock_data = pd.DataFrame({
        'x': [10.2, 10.5, 10.8],
        'y': [20.1, 20.4, 20.3],
        'label': pd.Series(['tagA', 'tagB', 'tagA'], dtype='category')
    })
    
    # Run binning with a resolution wide enough to hold all 3 points
    sparse_counts, bin_coords = compute_square_bins(mock_data, resolution=5.0)
    
    # Assertions to verify our array tracking dimensions match expectations
    assert sparse_counts.shape[0] == 1  # All coordinates must collapse to exactly 1 unique bin
    assert sparse_counts[0, 0] == 2    # Count for tagA should equal 2
    assert sparse_counts[0, 1] == 1    # Count for tagB should equal 1

def test_out_of_core_equivalence():
    # 1. Generate test dataset df
    mock_data = pd.DataFrame({
        'x': [12.5, 13.1, 55.0, 14.2, 85.0, 88.0, 102.0],
        'y': [45.2, 44.8, 82.1, 46.0, 81.0, 83.0, 105.0],
        'label': pd.Series(['elk', 'elk', 'mallard', 'columbine', 'elk', 'mallard', 'columbine'], dtype='category')
    })
    
    # Save mock data to temporary file
    csv_file = "test_data_ooc.csv"
    mock_data.to_csv(csv_file, index=False)
    
    # Parameters
    tile_size = 50.0
    resolution = 10.0
    max_dist = 15.0
    
    # 2. In-memory results
    sparse_counts_in_mem, bin_coords_in_mem = compute_square_bins(mock_data, resolution)
    adj_in_mem, lap_in_mem = build_spatial_graph(bin_coords_in_mem, max_dist)
    
    # 3. Out-of-core results
    partition_dir = "tmp_partition_ooc"
    tmp_out_dir = "tmp_out_ooc"
    
    partition_dataset_to_tiles(csv_file, tile_size, partition_dir)
    
    # Identify which tiles exist on disk
    tiles = []
    for f in os.listdir(partition_dir):
        if f.startswith("tile_") and f.endswith(".csv"):
            parts = f[5:-4].split("_")
            tiles.append((int(parts[0]), int(parts[1])))
            
    # Process each tile
    for tx, ty in tiles:
        process_tile_out_of_core(tx, ty, tile_size, resolution, max_dist, partition_dir, tmp_out_dir)
        
    # Stitch
    counts_ooc, centers_ooc, labels_ooc, adj_ooc, lap_ooc = stitch_out_of_core_results(tmp_out_dir, resolution)
    
    # Clean up files
    try:
        os.remove(csv_file)
        shutil.rmtree(partition_dir)
        shutil.rmtree(tmp_out_dir)
    except OSError:
        pass
        
    # 4. Assert mathematical equivalence
    assert counts_ooc.shape[0] == sparse_counts_in_mem.shape[0]
    
    # Sort bins to align comparison since sorting indices might vary
    in_mem_sort = np.lexsort((bin_coords_in_mem[:, 1], bin_coords_in_mem[:, 0]))
    ooc_sort = np.lexsort((centers_ooc[:, 1], centers_ooc[:, 0]))
    
    np.testing.assert_allclose(centers_ooc[ooc_sort], bin_coords_in_mem[in_mem_sort])
    
    # Compare count matrices after sorting rows
    counts_in_mem_sorted = sparse_counts_in_mem[in_mem_sort].toarray()
    counts_ooc_sorted = counts_ooc[ooc_sort].toarray()
    np.testing.assert_array_equal(counts_ooc_sorted, counts_in_mem_sorted)
    
    # Compare Adjacency Matrix
    adj_in_mem_sorted = adj_in_mem[in_mem_sort][:, in_mem_sort].toarray()
    adj_ooc_sorted = adj_ooc[ooc_sort][:, ooc_sort].toarray()
    np.testing.assert_allclose(adj_ooc_sorted, adj_in_mem_sorted)
    
    # Compare Laplacian
    lap_in_mem_sorted = lap_in_mem[in_mem_sort][:, in_mem_sort].toarray()
    lap_ooc_sorted = lap_ooc[ooc_sort][:, ooc_sort].toarray()
    np.testing.assert_allclose(lap_ooc_sorted, lap_in_mem_sorted)

def test_edge_case_empty_frames():
    """Verify binning and graph construction handle empty spatial frames gracefully."""
    x = np.array([])
    y = np.array([])
    res = 10.0
    
    bin_coords_sq, offset_sq = bin_square(x, y, res)
    assert len(bin_coords_sq) == 0
    assert offset_sq == (0.0, 0.0)
    
    bin_coords_kd, offset_kd = bin_kdtree(x, y, res)
    assert len(bin_coords_kd) == 0
    assert offset_kd == (0.0, 0.0)
    
    bin_coords_qt, offset_qt = bin_quadtree(x, y, res)
    assert len(bin_coords_qt) == 0
    assert offset_qt == (0.0, 0.0)
    
    adj, lap = build_spatial_graph(bin_coords_sq, max_distance=15.0)
    assert adj.shape == (0, 0)
    assert lap.shape == (0, 0)

def test_edge_case_extreme_coordinate_ranges():
    """Verify handling of extreme coordinate ranges spanning 10^-9 to 10^9."""
    x = np.array([-1e9, 0.0, 1e9])
    y = np.array([-1e9, 0.0, 1e9])
    res = 1000.0
    
    bin_coords, offset = bin_square(x, y, res)
    assert len(bin_coords) == 3
    assert bin_coords[0, 0] == 0
    assert bin_coords[2, 0] == int((1e9 - (-1e9)) / res)

def test_edge_case_floating_point_precision_limits():
    """Verify sub-nanometer floating point precision handling."""
    x = np.array([1.000000000001, 1.000000000002, 1.000000000009])
    y = np.array([2.0, 2.0, 2.0])
    res = 1e-12
    
    bin_coords, _ = bin_square(x, y, res)
    assert len(bin_coords) == 3
    assert bin_coords[0, 0] != bin_coords[2, 0]

def test_binning_algorithms_tradeoffs_and_benchmarking():
    """Verify correctness and benchmarking of Spatial Hashing vs KDTree vs Quadtree."""
    x = np.array([12.0, 15.0, 25.0, 34.0, 38.0])
    y = np.array([12.0, 15.0, 25.0, 34.0, 38.0])
    res = 10.0
    
    coords_sq, _ = bin_square(x, y, res)
    coords_kd, _ = bin_kdtree(x, y, res)
    coords_qt, _ = bin_quadtree(x, y, res)
    
    np.testing.assert_array_equal(coords_sq, coords_kd)
    np.testing.assert_array_equal(coords_sq, coords_qt)
    
    df_bench = benchmark_binning_algorithms(x, y, res, runs=1)
    assert len(df_bench) >= 3
    assert "Execution Time (ms)" in df_bench.columns
    assert "Peak Memory (MB)" in df_bench.columns


def test_parallel_graph_construction():
    """Verify parallel multi-threading and multi-processing graph construction equivalence."""
    np.random.seed(42)
    coords = np.random.uniform(0, 100, (60, 2))
    max_dist = 20.0
    
    adj_single, lap_single = build_spatial_graph(coords, max_dist, n_jobs=1)
    adj_threading, lap_threading = build_spatial_graph(coords, max_dist, n_jobs=2, backend="threading")
    adj_multi, lap_multi = build_spatial_graph(coords, max_dist, n_jobs=2, backend="multiprocessing")
    
    np.testing.assert_allclose(adj_threading.toarray(), adj_single.toarray())
    np.testing.assert_allclose(lap_threading.toarray(), lap_single.toarray())
    np.testing.assert_allclose(adj_multi.toarray(), adj_single.toarray())
    np.testing.assert_allclose(lap_multi.toarray(), lap_single.toarray())

def test_out_of_core_memory_profiling():
    """Verify exact peak RAM memory profiling for out-of-core operations."""
    file_path = "test_ooc_profile.csv"
    write_synthetic_points(file_path, num_points=2000)
    
    report = profile_out_of_core_execution(
        file_path, 
        tile_size=100.0, 
        resolution=20.0, 
        max_dist=25.0,
        partition_dir="tmp_test_part_prof",
        tmp_out_dir="tmp_test_out_prof"
    )
    
    assert "partitioning" in report
    assert "tile_processing" in report
    assert "stitching" in report
    assert "summary" in report
    assert report["summary"]["max_stage_peak_mb"] > 0
    assert report["summary"]["total_elapsed_sec"] > 0
    
    if os.path.exists(file_path):
        os.remove(file_path)

def test_adaptive_voronoi_binning():
    """Verify adaptive Voronoi spatial binning."""
    np.random.seed(42)
    x = np.random.uniform(0, 100, 50)
    y = np.random.uniform(0, 100, 50)
    
    bin_indices, offset, seeds = bin_voronoi(x, y, num_bins=10)
    assert len(bin_indices) == 50
    assert len(seeds) == 10
    assert len(set(bin_indices)) <= 10

def test_min_counts_thresholding():
    """Verify that min_counts drops sparse bins."""
    mock_data = pd.DataFrame({
        'x': [10.0, 10.0, 10.0, 90.0],
        'y': [10.0, 10.0, 10.0, 90.0],
        'label': pd.Series(['tagA', 'tagA', 'tagB', 'tagA'], dtype='category')
    })
    
    # min_counts = 1 -> keeps both bins
    sparse_1, coords_1 = compute_square_bins(mock_data, resolution=10.0, min_counts=1)
    assert len(coords_1) == 2
    
    # min_counts = 2 -> keeps only the bin at (10, 10) which has 3 counts
    sparse_2, coords_2 = compute_square_bins(mock_data, resolution=10.0, min_counts=2)
    assert len(coords_2) == 1

def test_pyg_data_export():
    """Verify PyTorch Geometric (PyG) format data export."""
    mock_data = pd.DataFrame({
        'x': [10.0, 20.0, 30.0],
        'y': [10.0, 20.0, 30.0],
        'label': pd.Series(['tagA', 'tagB', 'tagA'], dtype='category')
    })
    sparse_counts, bin_coords = compute_square_bins(mock_data, resolution=15.0)
    adj_matrix, _ = build_spatial_graph(bin_coords, max_distance=20.0)
    
    pyg_obj = to_pyg_data(sparse_counts, bin_coords, adj_matrix)
    if isinstance(pyg_obj, dict):
        assert "x" in pyg_obj
        assert "pos" in pyg_obj
        assert "edge_index" in pyg_obj
    else:
        assert hasattr(pyg_obj, "x")
        assert hasattr(pyg_obj, "pos")
        assert hasattr(pyg_obj, "edge_index")

def test_graph_laplacian_spectral_analysis():
    """Verify Graph Laplacian eigenvalue spectrum computation."""
    bin_coords = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0],
        [10.0, 10.0]
    ])
    _, laplacian_matrix = build_spatial_graph(bin_coords, max_distance=15.0)
    evals = compute_laplacian_spectrum(laplacian_matrix, k=3)
    assert len(evals) == 3
    assert np.all(evals >= -1e-5)


