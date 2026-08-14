"""
spatial_pipeline: High-performance spatial transcriptomics data engine.

Provides high-speed binning algorithms, GPU/CPU spatial graph generation (RAPIDS cuGraph/cuML),
Dask/Zarr out-of-core streaming, and AnnData/PyG interoperability.
"""

from __future__ import annotations

from .binning import (
    benchmark_binning_algorithms,
    bin_hexagonal,
    bin_kdtree,
    bin_quadtree,
    bin_square,
    bin_square_memmap,
    bin_voronoi,
    build_sparse_matrix,
    compute_square_bins,
    get_bin_centers,
)
from .graphs import (
    build_knn_graph,
    build_spatial_graph,
    compute_gearys_c,
    compute_gpu_knn_graph_cuml,
    compute_laplacian_spectrum,
    compute_morans_i,
    compute_spatially_variable_genes,
    is_gpu_available,
    to_pyg_data,
)
from .io import (
    arrow_table_to_numpy,
    download_dataset,
    from_anndata,
    generate_vectorized_mock_data,
    load_coordinates,
    load_transcript_arrow,
    load_transcript_data,
    read_anndata,
    read_zarr,
    save_anndata,
    stream_zarr_anndata,
    to_anndata,
    to_zarr,
)
from .out_of_core import (
    dask_bin_transcripts,
    dask_compute_spatial_metrics,
    enforce_memory_limit,
    load_tile_with_halo,
    partition_dataset_to_tiles,
    process_tile_out_of_core,
    process_tiles_parallel_out_of_core,
    profile_memory,
    profile_out_of_core_execution,
    stitch_out_of_core_results,
)

__version__ = "0.4.0"

__all__ = [
    "arrow_table_to_numpy",
    "benchmark_binning_algorithms",
    "bin_hexagonal",
    "bin_kdtree",
    "bin_quadtree",
    "bin_square",
    "bin_square_memmap",
    "bin_voronoi",
    "build_knn_graph",
    "build_sparse_matrix",
    "build_spatial_graph",
    "compute_gearys_c",
    "compute_gpu_knn_graph_cuml",
    "compute_laplacian_spectrum",
    "compute_morans_i",
    "compute_spatially_variable_genes",
    "compute_square_bins",
    "dask_bin_transcripts",
    "dask_compute_spatial_metrics",
    "download_dataset",
    "enforce_memory_limit",
    "from_anndata",
    "generate_vectorized_mock_data",
    "get_bin_centers",
    "is_gpu_available",
    "load_coordinates",
    "load_tile_with_halo",
    "load_transcript_arrow",
    "load_transcript_data",
    "partition_dataset_to_tiles",
    "process_tile_out_of_core",
    "process_tiles_parallel_out_of_core",
    "profile_memory",
    "profile_out_of_core_execution",
    "read_anndata",
    "read_zarr",
    "save_anndata",
    "stitch_out_of_core_results",
    "stream_zarr_anndata",
    "to_anndata",
    "to_pyg_data",
    "to_zarr",
]
