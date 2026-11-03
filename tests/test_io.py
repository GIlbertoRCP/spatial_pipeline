import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from spatial_pipeline.graphs import build_spatial_graph
from spatial_pipeline.io import (
    HAS_ANNDATA,
    HAS_PYARROW,
    HAS_ZARR,
    arrow_table_to_numpy,
    from_anndata,
    generate_vectorized_mock_data,
    load_transcript_arrow,
    load_transcript_data,
    read_anndata,
    read_zarr,
    save_anndata,
    stream_zarr_anndata,
    to_anndata,
    to_zarr,
)


def test_generate_vectorized_mock_data():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_vectorized_mock_data(tmp_path, num_molecules=500)
        assert os.path.exists(tmp_path)
        df = pd.read_csv(tmp_path)
        assert len(df) == 500
        assert set(df.columns) == {"x", "y", "label"}
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
        assert df["label"].dtype.name == "category"
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
        assert len(arr_dict["label"]) == 100
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.skipif(not HAS_ANNDATA, reason="AnnData not installed")
def test_anndata_roundtrip():
    counts = csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
    coords = np.array([[10.0, 10.0], [20.0, 20.0]])
    labels = np.array(["GAPDH", "ACTB"])
    adj, lap = build_spatial_graph(coords, max_distance=15.0)

    adata = to_anndata(
        sparse_counts=counts,
        bin_coords=coords,
        labels=labels,
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

    extracted = from_anndata(adata)
    assert extracted["counts"].shape == (2, 2)
    assert np.array_equal(extracted["bin_coords"], coords)
    assert np.array_equal(extracted["labels"], labels)

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


@pytest.mark.skipif(not (HAS_ANNDATA and HAS_ZARR), reason="AnnData or Zarr not installed")
def test_zarr_roundtrip_and_streaming():
    counts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    coords = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], dtype=np.float32)
    labels = np.array(["GAPDH", "ACTB"])

    adata = to_anndata(sparse_counts=counts, bin_coords=coords, labels=labels)
    tmp_zarr = tempfile.mkdtemp(suffix=".zarr")

    try:
        to_zarr(adata, tmp_zarr, chunks=(2, 2))
        assert os.path.exists(tmp_zarr)

        loaded_zarr = read_zarr(tmp_zarr)
        assert loaded_zarr.n_obs == 4
        assert loaded_zarr.n_vars == 2

        # Test streaming generator in batches of 2
        streamed_batches = list(stream_zarr_anndata(tmp_zarr, chunk_size=2))
        assert len(streamed_batches) == 2
        for chunk_X, chunk_spatial, var_names in streamed_batches:
            assert chunk_X.shape[0] == 2
            assert chunk_X.shape[1] == 2
    finally:
        if os.path.exists(tmp_zarr):
            shutil.rmtree(tmp_zarr)
