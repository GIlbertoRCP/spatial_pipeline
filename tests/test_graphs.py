import numpy as np
import pytest
from scipy.sparse import csr_matrix

from spatial_pipeline.graphs import (
    build_spatial_graph,
    compute_gearys_c,
    compute_laplacian_spectrum,
    compute_morans_i,
    compute_spatially_variable_genes,
    to_pyg_data,
)


def test_build_spatial_graph_empty():
    coords = np.empty((0, 2))
    adj, lap = build_spatial_graph(coords, max_distance=10.0)
    assert adj.shape == (0, 0)
    assert lap.shape == (0, 0)


def test_build_spatial_graph_single_node():
    coords = np.array([[10.0, 10.0]])
    adj, lap = build_spatial_graph(coords, max_distance=10.0)
    assert adj.shape == (1, 1)
    assert adj.nnz == 0
    assert lap.nnz == 0


def test_build_spatial_graph_connectivity():
    coords = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [20.0, 0.0]
    ])
    adj, lap = build_spatial_graph(coords, max_distance=10.0)
    assert adj.shape == (3, 3)
    # Node 0 and 1 are connected (dist 5 <= 10). Node 2 is isolated.
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    assert adj[0, 2] == 0.0
    assert adj[1, 2] == 0.0
    # Laplacian L = D - A
    assert lap[0, 0] == 1.0
    assert lap[0, 1] == -1.0


def test_build_spatial_graph_backends():
    np.random.seed(42)
    coords = np.random.uniform(0, 100, (60, 2))
    adj_seq, _ = build_spatial_graph(coords, max_distance=15.0, n_jobs=1)
    adj_multi, _ = build_spatial_graph(coords, max_distance=15.0, n_jobs=2, backend="multiprocessing")
    adj_thread, _ = build_spatial_graph(coords, max_distance=15.0, n_jobs=2, backend="threading")
    
    assert (adj_seq != adj_multi).nnz == 0
    assert (adj_seq != adj_thread).nnz == 0


def test_to_pyg_data():
    counts = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    coords = np.array([[0, 0], [5, 0], [10, 0]], dtype=np.float32)
    adj, _ = build_spatial_graph(coords, max_distance=6.0)
    
    pyg_out = to_pyg_data(counts, coords, adj)
    if isinstance(pyg_out, dict):
        assert "x" in pyg_out
        assert "pos" in pyg_out
        assert "edge_index" in pyg_out
        assert pyg_out["x"].shape == (3, 2)
        assert pyg_out["pos"].shape == (3, 2)
        assert pyg_out["edge_index"].shape[0] == 2
    else:
        # PyTorch Geometric Data object
        assert hasattr(pyg_out, "x")
        assert hasattr(pyg_out, "pos")
        assert hasattr(pyg_out, "edge_index")


def test_compute_laplacian_spectrum():
    coords = np.array([[0, 0], [5, 0], [10, 0], [15, 0]])
    _, lap = build_spatial_graph(coords, max_distance=6.0)
    evals = compute_laplacian_spectrum(lap, k=3)
    assert len(evals) == 3
    # Smallest eigenvalue of Graph Laplacian is always 0.0
    assert np.isclose(evals[0], 0.0, atol=1e-5)


def test_spatial_autocorrelation_morans_i():
    # Grid of 4 nodes: 0-(1,0), 1-(2,0), 2-(3,0), 3-(4,0)
    coords = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    adj, _ = build_spatial_graph(coords, max_distance=1.5)
    
    # Highly spatially autocorrelated gene (smooth gradient: 1, 2, 3, 4)
    # Random un-correlated gene (1, 10, 1, 10)
    counts = np.array([
        [1.0, 1.0],
        [2.0, 10.0],
        [3.0, 1.0],
        [4.0, 10.0]
    ])
    
    morans_i, z_scores, p_values = compute_morans_i(counts, adj)
    assert len(morans_i) == 2
    # Gene 0 should have high positive Moran's I
    assert morans_i[0] > 0.0
    # Gene 1 (alternating) should have negative/low Moran's I
    assert morans_i[0] > morans_i[1]


def test_spatial_autocorrelation_gearys_c():
    coords = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    adj, _ = build_spatial_graph(coords, max_distance=1.5)
    counts = np.array([[1.0], [2.0], [3.0], [4.0]])
    
    c_val = compute_gearys_c(counts, adj)
    assert len(c_val) == 1
    # Low Geary's C (< 1.0) indicates positive spatial autocorrelation
    assert c_val[0] < 1.0


def test_compute_spatially_variable_genes():
    coords = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    adj, _ = build_spatial_graph(coords, max_distance=1.5)
    counts = np.array([
        [1.0, 1.0],
        [2.0, 10.0],
        [3.0, 1.0],
        [4.0, 10.0]
    ])
    genes = ["GENE_A", "GENE_B"]
    df_svg = compute_spatially_variable_genes(counts, genes, adj)
    
    assert list(df_svg.columns) == ["gene", "morans_i", "gearys_c", "z_score", "p_value", "fdr_p_value"]
    assert len(df_svg) == 2
    assert df_svg.iloc[0]["gene"] == "GENE_A"
