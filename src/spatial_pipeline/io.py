"""
I/O, coordinate ingestion, zero-copy PyArrow ingestion, and AnnData / Zarr storage interop.

This module provides routines for downloading public spatial transcriptomics datasets,
loading high-throughput coordinates via PyArrow / Polars, and bi-directional conversion
between binned matrices, spatial graphs, AnnData (.h5ad), and Zarr stores.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None  # type: ignore
    HAS_POLARS = False

try:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pa_parquet
    HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore
    pa_csv = None  # type: ignore
    pa_parquet = None  # type: ignore
    HAS_PYARROW = False

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    ad = None  # type: ignore
    HAS_ANNDATA = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    zarr = None  # type: ignore
    HAS_ZARR = False


def download_dataset(url: str, output_dir: str) -> str:
    """
    Downloads a public dataset from the given URL and extracts it if it is a zip archive.

    Args:
        url: The HTTP/HTTPS URL of the dataset to download.
        output_dir: The target directory where the downloaded file should be saved and unpacked.

    Returns:
        str: The path to the file or extracted directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = url.split("/")[-1]
    download_path = os.path.join(output_dir, filename)

    if not os.path.exists(download_path):
        print(f"Downloading dataset from {url}...")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )

        with urllib.request.urlopen(req) as response, open(download_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 64  # 64 KB chunks
            count = 0

            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                count += 1
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    percent = min(percent, 100)
                    print(f"Progress: {percent}% completed ({count * block_size}/{total_size} bytes)", end="\r")
                else:
                    print(f"Downloaded {count * block_size} bytes", end="\r")

        print("\nDownload complete.")
    else:
        print(f"Dataset already exists at {download_path}")

    # Extract if zip
    if filename.endswith(".zip"):
        extract_dir = os.path.join(output_dir, filename[:-4])
        if not os.path.exists(extract_dir):
            print(f"Extracting {download_path} to {extract_dir}...")
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Extraction complete.")
        else:
            print(f"Files already extracted at {extract_dir}")
        return extract_dir

    return download_path


def load_coordinates(file_path: str, min_qv: float = 20.0) -> pd.DataFrame:
    """
    Loads spatial coordinates from a transcripts CSV, compressed CSV, or Parquet file.
    Attempts to use Polars for fast loading, falling back to Pandas.

    Args:
        file_path: Path to the transcripts CSV/Parquet file.
        min_qv: Minimum quality value (QV) threshold. Transcripts below this threshold are filtered out.

    Returns:
        pd.DataFrame: DataFrame with standardized columns: ['x', 'y', 'gene'].

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 10x Genomics Xenium transcript column mappings
    column_mapping = {
        "x_location": "x",
        "y_location": "y",
        "feature_name": "gene",
        "x": "x",
        "y": "y",
        "gene": "gene",
        "qv": "qv"
    }

    if HAS_POLARS and pl is not None:
        if file_path.endswith(".parquet"):
            df_pl = pl.read_parquet(file_path)
        else:
            df_pl = pl.read_csv(file_path)

        present_cols = [c for c in column_mapping if c in df_pl.columns]
        df_pl = df_pl.select(present_cols)
        rename_dict = {c: column_mapping[c] for c in present_cols}
        df_pl = df_pl.rename(rename_dict)

        if "qv" in df_pl.columns:
            df_pl = df_pl.filter(pl.col("qv") >= min_qv)
            df_pl = df_pl.drop("qv")

        pdf = df_pl.to_pandas()
    else:
        if file_path.endswith(".parquet"):
            df_pd = pd.read_parquet(file_path)
        else:
            df_pd = pd.read_csv(file_path)

        present_cols = [c for c in column_mapping if c in df_pd.columns]
        df_pd = df_pd[present_cols]
        df_pd = df_pd.rename(columns=column_mapping)

        if "qv" in df_pd.columns:
            df_pd = df_pd[df_pd["qv"] >= min_qv]
            df_pd = df_pd.drop(columns=["qv"])

        pdf = df_pd

    pdf["x"] = pdf["x"].astype("float64")
    pdf["y"] = pdf["y"].astype("float64")
    pdf["gene"] = pdf["gene"].astype("category")
    return pdf


def load_transcript_arrow(file_path: str) -> pa.Table:
    """
    Loads spatial transcriptomics coordinate file into a PyArrow Table for zero-copy memory operations.

    Args:
        file_path: Path to spatial transcripts CSV or Parquet file.

    Returns:
        pa.Table: PyArrow Table containing columns 'x', 'y', 'gene'.

    Raises:
        ImportError: If pyarrow is not installed.
        FileNotFoundError: If the file does not exist.
    """
    if not HAS_PYARROW or pa is None or pa_csv is None or pa_parquet is None:
        raise ImportError("PyArrow is not installed. Install with `pip install pyarrow`.")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No spatial transcript file found at {file_path}")

    str_path = str(path)
    if str_path.endswith(".parquet"):
        table = pa_parquet.read_table(str_path, columns=["x", "y", "gene"])
    else:
        convert_options = pa_csv.ConvertOptions(
            column_types={
                "x": pa.float32(),
                "y": pa.float32(),
                "gene": pa.string()
            },
            include_columns=["x", "y", "gene"]
        )
        table = pa_csv.read_csv(str_path, convert_options=convert_options)
    return table


def arrow_table_to_numpy(table: pa.Table) -> dict[str, np.ndarray]:
    """
    Converts PyArrow Table columns into NumPy arrays using zero-copy memory transfers where possible.

    Args:
        table: PyArrow Table with columns 'x', 'y', 'gene'.

    Returns:
        dict[str, np.ndarray]: Dictionary with keys 'x', 'y', 'gene' mapped to 1D NumPy arrays.
    """
    if not HAS_PYARROW:
        raise ImportError("PyArrow is not installed.")

    x_arr = table["x"].to_numpy(zero_copy_only=False)
    y_arr = table["y"].to_numpy(zero_copy_only=False)
    gene_arr = table["gene"].to_numpy(zero_copy_only=False)
    return {"x": x_arr, "y": y_arr, "gene": gene_arr}


def load_transcript_data(file_path: str, use_arrow: bool = False) -> pd.DataFrame:
    """
    Loads and optimizes spatial transcriptomics coordinate files.

    Args:
        file_path: Path to the CSV file.
        use_arrow: If True, uses PyArrow engine for zero-copy memory loading.

    Returns:
        pd.DataFrame: Loaded DataFrame with 'x', 'y', and 'gene' columns.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No spatial transcript file found at {file_path}")

    if use_arrow and HAS_PYARROW:
        table = load_transcript_arrow(file_path)
        df = table.to_pandas(types_mapper=None)
        df["gene"] = df["gene"].astype("category")
        return df

    df = pd.read_csv(
        path,
        usecols=["x", "y", "gene"],
        dtype={"x": "float32", "y": "float32", "gene": "category"}
    )
    return df


def generate_vectorized_mock_data(file_path: str, num_molecules: int = 15000) -> None:
    """
    Instantly generates thousands of synthetic spatial coordinates and gene labels
    using fully vectorized NumPy functions (no loops), then saves to a CSV.

    Args:
        file_path: Target CSV file path.
        num_molecules: Number of synthetic transcript coordinates to generate.
    """
    np.random.seed(42)
    n_back = num_molecules // 2
    n_clust = num_molecules - n_back

    x_back = np.random.uniform(10.0, 500.0, n_back)
    y_back = np.random.uniform(10.0, 500.0, n_back)

    n_clusters = 5
    centers = np.random.uniform(50.0, 450.0, (n_clusters, 2))
    cluster_ids = np.random.randint(0, n_clusters, size=n_clust)
    assigned_centers = centers[cluster_ids]

    noise = np.random.normal(0.0, 15.0, (n_clust, 2))
    x_clust = assigned_centers[:, 0] + noise[:, 0]
    y_clust = assigned_centers[:, 1] + noise[:, 1]

    x = np.clip(np.concatenate([x_back, x_clust]), 10.0, 500.0)
    y = np.clip(np.concatenate([y_back, y_clust]), 10.0, 500.0)

    gene_list = ["GAPDH", "ACTB", "MALAT1", "EPCAM", "CD3D"]
    genes = np.random.choice(gene_list, size=num_molecules)

    df = pd.DataFrame({
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "gene": genes
    })
    df.to_csv(file_path, index=False)


def to_anndata(
    sparse_counts: Any,
    bin_coords: np.ndarray,
    genes: np.ndarray | None = None,
    adjacency_matrix: Any = None,
    laplacian_matrix: Any = None,
    compute_svg: bool = False,
    metadata: dict[str, Any] | None = None
) -> ad.AnnData:
    """
    Exports spatial binning count matrix, spatial coordinates, and adjacency graphs
    into an AnnData object following Scanpy and Squidpy ecosystem conventions.

    Args:
        sparse_counts: Binned gene expression counts matrix of shape (num_bins, num_genes).
        bin_coords: Array of shape (num_bins, 2) holding Cartesian (x, y) coordinates of bin centers.
        genes: 1D array of gene names (length num_genes).
        adjacency_matrix: Spatial adjacency matrix of shape (num_bins, num_bins). Stored in `obsp['spatial_connectivities']`.
        laplacian_matrix: Graph Laplacian matrix of shape (num_bins, num_bins). Stored in `obsp['spatial_laplacian']`.
        compute_svg: If True, calculates Moran's I & Geary's C spatial autocorrelation metrics and stores in `var`.
        metadata: Pipeline metadata stored in `uns['spatial_pipeline']`.

    Returns:
        ad.AnnData: Standardized AnnData object.
    """
    if not HAS_ANNDATA or ad is None:
        raise ImportError("anndata package is required. Install with `pip install anndata`.")

    adata = ad.AnnData(X=sparse_counts)
    adata.obsm["spatial"] = np.asarray(bin_coords, dtype=np.float32)

    if genes is not None:
        adata.var_names = np.asarray(genes).astype(str)

    if adjacency_matrix is not None:
        adata.obsp["spatial_connectivities"] = adjacency_matrix

        if compute_svg:
            from .graphs import compute_gearys_c, compute_morans_i
            morans_i, _, p_values = compute_morans_i(sparse_counts, adjacency_matrix)
            gearys_c = compute_gearys_c(sparse_counts, adjacency_matrix)
            adata.var["morans_i"] = morans_i
            adata.var["gearys_c"] = gearys_c
            adata.var["morans_pvalue"] = p_values

    if laplacian_matrix is not None:
        adata.obsp["spatial_laplacian"] = laplacian_matrix

    if metadata is not None:
        adata.uns["spatial_pipeline"] = metadata
    else:
        adata.uns["spatial_pipeline"] = {"version": "0.4.0"}

    return adata


def from_anndata(adata: ad.AnnData) -> dict[str, Any]:
    """
    Extracts sparse count matrix, spatial coordinates, gene names, and adjacency graphs from an AnnData object.

    Args:
        adata: Input AnnData object.

    Returns:
        dict[str, Any]: Dictionary containing 'counts', 'bin_coords', 'genes', 'adjacency_matrix', and 'laplacian_matrix'.
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata package is required.")

    counts = adata.X
    bin_coords = adata.obsm.get("spatial", np.empty((adata.n_obs, 2)))
    genes = np.array(adata.var_names)
    adjacency = adata.obsp.get("spatial_connectivities", None)
    laplacian = adata.obsp.get("spatial_laplacian", None)

    return {
        "counts": counts,
        "bin_coords": bin_coords,
        "genes": genes,
        "adjacency_matrix": adjacency,
        "laplacian_matrix": laplacian
    }


def save_anndata(adata: ad.AnnData, file_path: str) -> None:
    """
    Saves AnnData object to disk as an HDF5 .h5ad file.

    Args:
        adata: AnnData object.
        file_path: Target .h5ad file path.
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata package is required.")
    adata.write_h5ad(file_path)


def read_anndata(file_path: str) -> ad.AnnData:
    """
    Reads an AnnData .h5ad file from disk.

    Args:
        file_path: Path to .h5ad file.

    Returns:
        ad.AnnData: Loaded AnnData object.
    """
    if not HAS_ANNDATA or ad is None:
        raise ImportError("anndata package is required.")
    return ad.read_h5ad(file_path)


def to_zarr(
    adata: ad.AnnData,
    store_path: str,
    chunks: tuple[int, int] | None = None
) -> None:
    """
    Exports AnnData object to a chunked Zarr hierarchy on disk for out-of-core streaming.

    Args:
        adata: Input AnnData object.
        store_path: Path to the output directory or .zarr directory.
        chunks: Optional 2D chunk size (n_obs_chunk, n_vars_chunk). Default is (10000, n_vars).
    """
    if not HAS_ANNDATA or ad is None:
        raise ImportError("anndata is required for Zarr export.")

    if hasattr(adata, "write_zarr"):
        adata.write_zarr(store_path, chunks=chunks)
    else:
        # Fallback manual Zarr store
        if not HAS_ZARR or zarr is None:
            raise ImportError("zarr is required. Install with `pip install zarr`.")
        root = zarr.open_group(store_path, mode="w")
        if hasattr(adata.X, "toarray"):
            dense_x = adata.X.toarray()
        else:
            dense_x = np.asarray(adata.X)
        chunk_shape = chunks if chunks is not None else (min(10000, dense_x.shape[0]), dense_x.shape[1])
        root.create_dataset("X", data=dense_x, chunks=chunk_shape)
        if "spatial" in adata.obsm:
            root.create_dataset("obsm/spatial", data=np.asarray(adata.obsm["spatial"]))
        root.create_dataset("var_names", data=np.asarray(adata.var_names, dtype=str))


def read_zarr(store_path: str) -> ad.AnnData:
    """
    Reads an AnnData object from a Zarr store hierarchy on disk.

    Args:
        store_path: Path to the .zarr directory.

    Returns:
        ad.AnnData: Loaded AnnData object.
    """
    if not HAS_ANNDATA or ad is None:
        raise ImportError("anndata is required to read Zarr.")

    if hasattr(ad, "read_zarr"):
        return ad.read_zarr(store_path)

    if not HAS_ZARR or zarr is None:
        raise ImportError("zarr is required.")
    root = zarr.open_group(store_path, mode="r")
    X = np.asarray(root["X"])
    adata = ad.AnnData(X=X)
    if "obsm/spatial" in root:
        adata.obsm["spatial"] = np.asarray(root["obsm/spatial"])
    if "var_names" in root:
        adata.var_names = np.asarray(root["var_names"]).astype(str)
    return adata


def stream_zarr_anndata(
    store_path: str,
    chunk_size: int = 10000
) -> Generator[tuple[np.ndarray, np.ndarray | None, np.ndarray | None], None, None]:
    """
    Streams sub-chunks of an out-of-core Zarr dataset without loading the full matrix into host RAM.

    Args:
        store_path: Path to the .zarr store.
        chunk_size: Number of observation rows to stream per batch.

    Yields:
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
            (chunk_X, chunk_spatial_coords, var_names) for each streaming batch.
    """
    if not HAS_ZARR or zarr is None:
        raise ImportError("zarr package is required for streaming. Install with `pip install zarr`.")

    root = zarr.open_group(store_path, mode="r")
    x_node = root.get("X")
    if x_node is None:
        # Check anndata zarr structure
        x_node = root.get("raw/X") or root
    total_obs = x_node.shape[0]

    spatial_node = root.get("obsm/spatial") or root.get("obsm", {}).get("spatial")
    var_names = None
    if "var" in root and "_index" in root["var"]:
        var_names = np.asarray(root["var"]["_index"])
    elif "var_names" in root:
        var_names = np.asarray(root["var_names"])

    for start_idx in range(0, total_obs, chunk_size):
        end_idx = min(start_idx + chunk_size, total_obs)
        chunk_X = np.asarray(x_node[start_idx:end_idx])
        chunk_spatial = np.asarray(spatial_node[start_idx:end_idx]) if spatial_node is not None else None
        yield chunk_X, chunk_spatial, var_names
