import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from spatial_pipeline.graphs import build_spatial_graph
from spatial_pipeline.io import (
    HAS_ANNDATA,
    HAS_PYARROW,
    arrow_table_to_numpy,
    from_anndata,
    generate_vectorized_mock_data,
    load_transcript_arrow,
    load_transcript_data,
    read_anndata,
    save_anndata,
    to_anndata,
)


def test_generate_vectorized_mock_data():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        generate_vectorized_mock_data(tmp_path, num_molecules=500)
        assert os.path.exists(tmp_path)
        df = pd.read_csv(tmp_path)
        assert len(df) == 500
        assert set(df.columns) == {"x", "y", "gene"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_transcript_data():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        generate_vectorized_mock_data(tmp_path, num_molecules=200)
        df = load_transcript_data(tmp_path)
        assert len(df) == 200
        assert df['gene'].dtype.name == 'category'
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_pyarrow_loading():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        generate_vectorized_mock_data(tmp_path, num_molecules=100)
        table = load_transcript_arrow(tmp_path)
        arr_dict = arrow_table_to_numpy(table)
        assert len(arr_dict["x"]) == 100
        assert len(arr_dict["y"]) == 100
        assert len(arr_dict["gene"]) == 100
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.skipif(not HAS_ANNDATA, reason="AnnData not installed")
def test_anndata_roundtrip():
    counts = csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
    coords = np.array([[10.0, 10.0], [20.0, 20.0]])
    genes = np.array(["GAPDH", "ACTB"])
    adj, lap = build_spatial_graph(coords, max_distance=15.0)
    
    adata = to_anndata(
        sparse_counts=counts,
        bin_coords=coords,
        genes=genes,
        adjacency_matrix=adj,
        laplacian_matrix=lap,
        compute_svg=True
    )
    
    assert adata.n_obs == 2
    assert adata.n_vars == 2
    assert "spatial" in adata.obsm
    assert "spatial_connectivities" in adata.obsp
    assert "spatial_laplacian" in adata.obsp
    assert "morans_i" in adata.var
    assert "gearys_c" in adata.var
    
    # Test from_anndata extraction
    extracted = from_anndata(adata)
    assert extracted["counts"].shape == (2, 2)
    assert np.array_equal(extracted["bin_coords"], coords)
    assert np.array_equal(extracted["genes"], genes)
    
    # Test disk save & read
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_h5ad = tmp.name
        
    try:
        save_anndata(adata, tmp_h5ad)
        assert os.path.exists(tmp_h5ad)
        loaded_adata = read_anndata(tmp_h5ad)
        assert loaded_adata.n_obs == 2
        assert loaded_adata.n_vars == 2
    finally:
        if os.path.exists(tmp_h5ad):
            os.remove(tmp_h5ad)
