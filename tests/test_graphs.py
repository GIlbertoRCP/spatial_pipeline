import numpy as np
import pytest
from scipy.sparse import csr_matrix

from spatial_pipeline.graphs import (
    build_knn_graph,
    build_spatial_graph,
    compute_gearys_c,
    compute_laplacian_spectrum,
    compute_morans_i,
    compute_spatially_variable_labels,
    is_gpu_available,
    to_pyg_data,
)


def test_is_gpu_available():
    # Calling is_gpu_available should return a boolean without crashing
    res = is_gpu_available()
    assert isinstance(res, bool)


def test_build_knn_graph_basic():
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [10.0, 10.0]
    ])
    adj, lap = build_knn_graph(coords, n_neighbors=2, backend="auto")
    assert adj.shape == (4, 4)
    assert lap is not None
    assert lap.shape == (4, 4)
    # Node 0 nearest neighbor should include Node 1
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    # Graph is symmetric and undirected
    np.testing.assert_allclose(adj.toarray(), adj.toarray().T)


def test_build_knn_graph_empty():
    coords = np.empty((0, 2))
    adj, lap = build_knn_graph(coords, n_neighbors=5)
    assert adj.shape == (0, 0)
    assert lap is not None
    assert lap.shape == (0, 0)


def test_build_knn_graph_gpu_fallback():
    # If cuml is requested on a non-CUDA host, it should gracefully fall back to CPU
    coords = np.random.uniform(0, 100, (20, 2))
    adj, lap = build_knn_graph(coords, n_neighbors=3, backend="cuml")
    assert adj.shape == (20, 20)
    assert lap is not None


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
    adj_auto, _ = build_spatial_graph(coords, max_distance=15.0, backend="auto")

    assert (adj_seq != adj_multi).nnz == 0
    assert (adj_seq != adj_thread).nnz == 0
    assert (adj_seq != adj_auto).nnz == 0


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
    coords = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    adj, _ = build_spatial_graph(coords, max_distance=1.5)

    counts = np.array([
        [1.0, 1.0],
        [2.0, 10.0],
        [3.0, 1.0],
        [4.0, 10.0]
    ])

    morans_i, _, _ = compute_morans_i(counts, adj)
    assert len(morans_i) == 2
    # Label 0 should have high positive Moran's I
    assert morans_i[0] > 0.0
    # Label 1 (alternating) should have negative/low Moran's I
    assert morans_i[0] > morans_i[1]


def test_spatial_autocorrelation_gearys_c():
    coords = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    adj, _ = build_spatial_graph(coords, max_distance=1.5)
    counts = np.array([[1.0], [2.0], [3.0], [4.0]])

    c_val = compute_gearys_c(counts, adj)
    assert len(c_val) == 1
    # Low Geary's C (< 1.0) indicates positive spatial autocorrelation
    assert c_val[0] < 1.0


def test_compute_spatially_variable_labels():
    coords = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    adj, _ = build_spatial_graph(coords, max_distance=1.5)
    counts = np.array([
        [1.0, 1.0],
        [2.0, 10.0],
        [3.0, 1.0],
        [4.0, 10.0]
    ])
    labels = ["GENE_A", "GENE_B"]
    df_svg = compute_spatially_variable_labels(counts, labels, adj)

    assert list(df_svg.columns) == ["label", "morans_i", "gearys_c", "z_score", "p_value", "fdr_p_value"]
    assert len(df_svg) == 2
    assert df_svg.iloc[0]["label"] == "GENE_A"


def test_fdr_bh_matches_scipy():
    from scipy.stats import false_discovery_control

    from spatial_pipeline.graphs import fdr_bh

    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, size=(40, 3)) ** 3
    expected = np.column_stack([false_discovery_control(p[:, j]) for j in range(3)])
    np.testing.assert_allclose(fdr_bh(p, axis=0), expected, rtol=1e-12)
    np.testing.assert_allclose(fdr_bh(p.T, axis=1), expected.T, rtol=1e-12)
    np.testing.assert_allclose(fdr_bh(p[:, 0]), expected[:, 0], rtol=1e-12)
    assert fdr_bh(np.array([])).shape == (0,)


def test_getis_ord_gi_star_matches_textbook_formula():
    from spatial_pipeline.graphs import build_spatial_graph, compute_getis_ord_gi_star

    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 10, size=(60, 2))
    adj, _ = build_spatial_graph(coords, max_distance=2.0, backend="cpu")
    X = rng.poisson(2.0, size=(60, 4)).astype(float)
    hot = np.hypot(coords[:, 0] - 2.5, coords[:, 1] - 2.5) < 2.0
    X[hot, 0] += 15  # plant a spatially clustered hot spot in column 0

    z, p = compute_getis_ord_gi_star(csr_matrix(X), adj)

    # Reference: Ord & Getis (1995) Gi* with binary weights including self, one bin at a time.
    A = adj.toarray() != 0
    n = len(X)
    for j in range(X.shape[1]):
        x = X[:, j]
        xbar = x.mean()
        s = np.sqrt((x ** 2).mean() - xbar ** 2)
        for i in range(n):
            w = A[i].astype(float)
            w[i] = 1.0
            ref = (w @ x - xbar * w.sum()) / (s * np.sqrt((n * (w ** 2).sum() - w.sum() ** 2) / (n - 1)))
            assert z[i, j] == pytest.approx(ref, rel=1e-9, abs=1e-12)

    assert hot.sum() >= 3
    assert z[hot, 0].mean() > 2.0
    assert z[hot, 0].mean() > z[~hot, 0].mean() + 2.0
    assert np.all((p >= 0) & (p <= 1))

    z1, p1 = compute_getis_ord_gi_star(X[:, 0], adj)
    np.testing.assert_allclose(z1, z[:, 0])
    np.testing.assert_allclose(p1, p[:, 0])


def test_getis_ord_gi_star_constant_column_is_zero():
    from spatial_pipeline.graphs import build_spatial_graph, compute_getis_ord_gi_star

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    adj, _ = build_spatial_graph(coords, max_distance=1.1, backend="cpu")
    z, p = compute_getis_ord_gi_star(np.ones((4, 1)), adj)
    assert np.all(z == 0) and np.all(p == 1)
